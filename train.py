import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from env.traffic_env import GridEnv
from agent import TrafficDecisionTransformer
import torch.nn as nn
import os

CONTEXT_LEN = 20   # Context window K — must match agent.py
NUM_NODES = 36     # 6x6 grid
MAX_STEPS = 200    # Trajectory length — must match evaluate.py


# ---------------------------------------------------------------------------
# 1a.  LOAD EXPERT DATA (from teammate's expert_dataset.pt)
# ---------------------------------------------------------------------------
def load_expert_trajectories(path='expert_dataset.pt', chunk_len=MAX_STEPS):
    """
    expert_dataset.pt contains:
      states  : (10000, 36, 10)
      actions : (10000, 36)
      rewards : (10000, 36)

    We slice the 10,000-step continuous simulation into non-overlapping
    chunks of chunk_len steps, one trajectory per node per chunk.
    → 10000/200 = 50 chunks × 36 nodes = 1,800 expert trajectories.
    """
    if not os.path.exists(path):
        print(f"  expert_dataset.pt not found — skipping expert data")
        return []

    print(f"  Loading expert data from {path} …")
    data = torch.load(path, map_location='cpu', weights_only=True)
    states  = data['states'].numpy()   # (10000, 36, 10)
    actions = data['actions'].numpy()  # (10000, 36)
    rewards = data['rewards'].numpy()  # (10000, 36)

    total_steps = states.shape[0]
    num_chunks  = total_steps // chunk_len
    trajectories = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_len
        end   = start + chunk_len

        for n in range(NUM_NODES):
            s = states[start:end, n, :]        # (chunk_len, 10)
            a = actions[start:end, n]           # (chunk_len,)
            r = rewards[start:end, n]           # (chunk_len,)

            # Compute Returns-to-Go
            rtg = np.zeros_like(r)
            ret = 0.0
            for t in reversed(range(len(r))):
                ret += r[t]
                rtg[t] = ret

            trajectories.append({
                'states':       s.astype(np.float32),
                'actions':      a.astype(np.int64),
                'rewards':      r.astype(np.float32),
                'returns_to_go': rtg[..., None].astype(np.float32),
                'timesteps':    np.arange(chunk_len, dtype=np.int64),
            })

    print(f"  → {len(trajectories)} expert trajectories "
          f"({num_chunks} chunks × {NUM_NODES} nodes)")
    return trajectories


# ---------------------------------------------------------------------------
# 1b.  GENERATE DIVERSE-QUALITY DATA (full network, uniform policy/episode)
# ---------------------------------------------------------------------------
def generate_network_trajectories(num_episodes=200, max_steps=MAX_STEPS):
    """
    Simulate the full 36-node grid with UNIFORM policy per episode.
    Creates diverse-quality data: some episodes use good switching intervals,
    some use bad ones. This gives the DT low-return examples to contrast
    against the expert data.
    """
    all_trajectories = []

    for ep in range(num_episodes):
        env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)

        frac = ep / max(num_episodes - 1, 1)
        if frac < 0.10:
            policy = 'random'
        elif frac < 0.15:
            policy = ('timed', 999)
        elif frac < 0.25:
            policy = ('timed', 3 + int((frac - 0.15) / 0.10 * 3))
        elif frac < 0.65:
            policy = ('timed', 6 + int((frac - 0.25) / 0.40 * 13))
        elif frac < 0.80:
            policy = ('timed', 20 + int((frac - 0.65) / 0.15 * 21))
        elif frac < 0.90:
            policy = 'adaptive'
        else:
            policy = 'random'

        node_states  = [[] for _ in range(NUM_NODES)]
        node_actions = [[] for _ in range(NUM_NODES)]
        node_rewards = [[] for _ in range(NUM_NODES)]
        timers       = [0] * NUM_NODES

        initial_states = [inter.get_state_vector() for inter in env.intersections]

        for step in range(max_steps):
            actions = []
            for n in range(NUM_NODES):
                inter = env.intersections[n]
                if step == 0:
                    node_states[n].append(initial_states[n])

                if policy == 'random':
                    action = np.random.choice([0, 1], p=[0.75, 0.25])
                elif policy == 'adaptive':
                    g = inter.current_phase
                    r = 1 - g
                    if (inter.queue_lengths[r] > inter.queue_lengths[g] + 2
                            and inter.time_in_phase >= 5):
                        action = 1
                    elif inter.time_in_phase >= 20:
                        action = 1
                    else:
                        action = 0
                else:
                    interval = policy[1]
                    timers[n] += 1
                    if timers[n] >= interval:
                        action = 1
                        timers[n] = 0
                    else:
                        action = 0

                node_actions[n].append(action)
                actions.append(action)

            new_states, rewards = env.step(actions)
            for n in range(NUM_NODES):
                node_rewards[n].append(rewards[n])
                if step < max_steps - 1:
                    node_states[n].append(new_states[n])

        for n in range(NUM_NODES):
            rewards_arr = np.array(node_rewards[n], dtype=np.float32)
            rtg = np.zeros_like(rewards_arr)
            ret = 0.0
            for t in reversed(range(len(rewards_arr))):
                ret += rewards_arr[t]
                rtg[t] = ret

            all_trajectories.append({
                'states':       np.array(node_states[n], dtype=np.float32),
                'actions':      np.array(node_actions[n], dtype=np.int64),
                'rewards':      rewards_arr,
                'returns_to_go': rtg[..., None],
                'timesteps':    np.arange(max_steps, dtype=np.int64),
            })

    return all_trajectories


# ---------------------------------------------------------------------------
# 2.  NORMALIZATION
# ---------------------------------------------------------------------------
def normalize_data(trajectories):
    """
    States  → z-score  (subtract mean, divide by std)
    RTG     → scale-only  (divide by scale, NO mean subtraction)
    """
    all_states = np.concatenate([t['states'] for t in trajectories], axis=0)
    state_mean = all_states.mean(axis=0)
    state_std  = all_states.std(axis=0) + 1e-6

    episode_returns = np.array([t['returns_to_go'][0, 0] for t in trajectories])
    rtg_scale = np.std(episode_returns) + 1e-6

    for t in trajectories:
        t['states']       = ((t['states'] - state_mean) / state_std).astype(np.float32)
        t['returns_to_go'] = (t['returns_to_go'] / rtg_scale).astype(np.float32)

    top_return_raw = float(np.percentile(episode_returns, 90))
    target_rtg     = top_return_raw / rtg_scale

    return trajectories, state_mean, state_std, rtg_scale, target_rtg


# ---------------------------------------------------------------------------
# 3.  PYTORCH DATASET
# ---------------------------------------------------------------------------
class DecisionTransformerDataset(Dataset):
    def __init__(self, trajectories, context_len=CONTEXT_LEN, samples_per_traj=5):
        self.trajectories   = trajectories
        self.context_len    = context_len
        self.samples_per_traj = samples_per_traj

    def __len__(self):
        return len(self.trajectories) * self.samples_per_traj

    def __getitem__(self, idx):
        traj = self.trajectories[idx % len(self.trajectories)]
        traj_len = len(traj['states'])

        start = 0 if traj_len <= self.context_len else np.random.randint(0, traj_len - self.context_len)
        end   = start + self.context_len

        s = traj['states'][start:end]
        a = traj['actions'][start:end]
        r = traj['returns_to_go'][start:end]
        t = traj['timesteps'][start:end]

        cur = s.shape[0]
        if cur < self.context_len:
            pad = self.context_len - cur
            s = np.pad(s, ((0, pad), (0, 0)), 'constant')
            a = np.pad(a, (0, pad),           'constant')
            r = np.pad(r, ((0, pad), (0, 0)), 'constant')
            t = np.pad(t, (0, pad),           'constant')

        return {
            'states':       torch.tensor(s, dtype=torch.float32),
            'actions':      torch.tensor(a, dtype=torch.long),
            'returns_to_go': torch.tensor(r, dtype=torch.float32),
            'timesteps':    torch.tensor(t, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 4.  TRAINING LOOP
# ---------------------------------------------------------------------------
def train():
    # --- Collect all trajectories ---
    print("=== Data Collection ===")

    expert_trajs = load_expert_trajectories('expert_dataset.pt', chunk_len=MAX_STEPS)

    print("  Generating diverse-policy trajectories from GridEnv …")
    diverse_trajs = generate_network_trajectories(num_episodes=200, max_steps=MAX_STEPS)
    print(f"  → {len(diverse_trajs)} diverse trajectories")

    all_trajs = expert_trajs + diverse_trajs
    print(f"  Total: {len(all_trajs)} trajectories")

    # --- Stats ---
    ep_rets = [t['returns_to_go'][0, 0] for t in all_trajs]
    print(f"\n  Episode-return stats: mean={np.mean(ep_rets):.0f}  "
          f"best={np.max(ep_rets):.0f}  worst={np.min(ep_rets):.0f}")
    if expert_trajs:
        expert_rets = [t['returns_to_go'][0, 0] for t in expert_trajs]
        diverse_rets = [t['returns_to_go'][0, 0] for t in diverse_trajs]
        print(f"  Expert returns:  mean={np.mean(expert_rets):.0f}  best={np.max(expert_rets):.0f}")
        print(f"  Diverse returns: mean={np.mean(diverse_rets):.0f}  best={np.max(diverse_rets):.0f}")

    # --- Normalize ---
    all_trajs, s_mean, s_std, rtg_scale, target_rtg = normalize_data(all_trajs)
    print(f"\n  rtg_scale = {rtg_scale:.1f}")
    print(f"  Target RTG (scaled, top-10%) = {target_rtg:.3f}")

    np.savez('norm_stats.npz',
             state_mean=s_mean, state_std=s_std,
             rtg_scale=rtg_scale, target_rtg=target_rtg)
    print("  Saved norm_stats.npz")

    # --- Train ---
    dataset    = DecisionTransformerDataset(all_trajs, CONTEXT_LEN, samples_per_traj=5)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Training ===")
    print(f"  Device: {device},  {len(dataset)} samples/epoch")

    model = TrafficDecisionTransformer(
        state_dim=10, act_dim=2,
        hidden_size=128, max_length=CONTEXT_LEN,
        num_layers=3, num_heads=4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 80
    print(f"  {epochs} epochs …")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            s  = batch['states'].to(device)
            a  = batch['actions'].to(device)
            r  = batch['returns_to_go'].to(device)
            ts = batch['timesteps'].to(device)

            optimizer.zero_grad()
            logits = model(s, a, r, ts)
            loss = criterion(logits.reshape(-1, 2), a.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

    torch.save(model.state_dict(), 'dt_traffic_model.pth')
    print("Saved dt_traffic_model.pth  ✓")


if __name__ == '__main__':
    train()
