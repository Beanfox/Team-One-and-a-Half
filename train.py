import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from env.traffic_env import GridEnv
from agent import TrafficDecisionTransformer
import torch.nn as nn
import os, time

CONTEXT_LEN = 20
NUM_NODES = 36
MAX_STEPS = 200


# ===================================================================
#  ORACLE POLICY — genuinely better than any fixed timer
# ===================================================================
def oracle_action(inter):
    """
    State-aware switching: patient, only switches when there's real pressure.
    Beats fixed timers by not wasting green on empty queues and by reading
    the queue pressure differential.
    """
    g = inter.current_phase
    r = 1 - g
    green_q = inter.queue_lengths[g]
    red_q   = inter.queue_lengths[r]
    t_green = inter.time_in_phase
    ds_cap  = inter.downstream_capacity[g]

    # Rule 1: Green queue empty & red has buildup → switch
    if green_q < 0.5 and red_q > 3 and t_green >= 5:
        return 1

    # Rule 2: Downstream full → green can't flow, give red a turn
    if ds_cap < 1.0 and red_q > 4 and t_green >= 8:
        return 1

    # Rule 3: Red has MUCH more pressure (3x ratio + offset)
    if red_q > green_q * 3.0 + 6 and t_green >= 10:
        return 1

    # Rule 4: Maximum green time to prevent starvation
    if t_green >= 30 and red_q > 2:
        return 1

    # Rule 5: Moderate pressure after long green
    if red_q > green_q + 5 and t_green >= 15:
        return 1

    return 0


# ===================================================================
#  DATA GENERATION — from the full 36-node GridEnv
# ===================================================================
def generate_network_trajectories(num_episodes=300, max_steps=MAX_STEPS):
    """
    UNIFORM policy per episode. Crucially includes the oracle policy
    which is genuinely better than any fixed timer.
    """
    all_trajectories = []

    for ep in range(num_episodes):
        env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)

        # Policy distribution:
        #   40% oracle (the best policy — DT should learn to imitate this)
        #   10% adaptive (decent)
        #   10% timer 8-12  (good fixed timers)
        #   10% timer 13-18 (includes the baseline range)
        #   10% timer 3-6   (bad — too fast)
        #   10% timer 25-50 (bad — too slow)
        #   10% random      (bad)
        frac = ep / max(num_episodes - 1, 1)
        if frac < 0.40:
            policy = 'oracle'
        elif frac < 0.50:
            policy = 'adaptive'
        elif frac < 0.60:
            policy = ('timed', 8 + int((frac - 0.50) / 0.10 * 5))
        elif frac < 0.70:
            policy = ('timed', 13 + int((frac - 0.60) / 0.10 * 6))
        elif frac < 0.80:
            policy = ('timed', 3 + int((frac - 0.70) / 0.10 * 4))
        elif frac < 0.90:
            policy = ('timed', 25 + int((frac - 0.80) / 0.10 * 26))
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

                if policy == 'oracle':
                    action = oracle_action(inter)
                elif policy == 'random':
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


# ===================================================================
#  EXPERT DATA LOADING (optional)
# ===================================================================
def load_expert_trajectories(path='expert_dataset.pt', chunk_len=MAX_STEPS):
    if not os.path.exists(path):
        print("  expert_dataset.pt not found — skipping")
        return []

    print(f"  Loading expert data from {path} …")
    data = torch.load(path, map_location='cpu', weights_only=True)
    states  = data['states'].numpy()
    actions = data['actions'].numpy()
    rewards = data['rewards'].numpy()

    # Only use FIRST 5 chunks (1000 steps) — later chunks have extreme congestion
    total_steps = min(states.shape[0], chunk_len * 5)
    num_chunks  = total_steps // chunk_len
    trajectories = []

    for ci in range(num_chunks):
        s0, s1 = ci * chunk_len, (ci + 1) * chunk_len
        for n in range(NUM_NODES):
            r_arr = rewards[s0:s1, n].astype(np.float32)
            rtg = np.zeros_like(r_arr)
            ret = 0.0
            for t in reversed(range(len(r_arr))):
                ret += r_arr[t]
                rtg[t] = ret
            trajectories.append({
                'states':       states[s0:s1, n, :].astype(np.float32),
                'actions':      actions[s0:s1, n].astype(np.int64),
                'rewards':      r_arr,
                'returns_to_go': rtg[..., None],
                'timesteps':    np.arange(chunk_len, dtype=np.int64),
            })

    print(f"  → {len(trajectories)} expert trajectories ({num_chunks} early chunks × {NUM_NODES} nodes)")
    return trajectories


# ===================================================================
#  NORMALIZATION (scale-only RTG, z-score states)
# ===================================================================
def normalize_data(all_trajectories, reference_trajectories):
    """
    States: z-score (from ALL data)
    RTG: scale-only (from REFERENCE data — the diverse/generated data)
    Target: top 10% of REFERENCE episode returns
    """
    all_states = np.concatenate([t['states'] for t in all_trajectories], axis=0)
    state_mean = all_states.mean(axis=0)
    state_std  = all_states.std(axis=0) + 1e-6

    ref_returns = np.array([t['returns_to_go'][0, 0] for t in reference_trajectories])
    rtg_scale = np.std(ref_returns) + 1e-6

    for t in all_trajectories:
        t['states']       = ((t['states'] - state_mean) / state_std).astype(np.float32)
        t['returns_to_go'] = (t['returns_to_go'] / rtg_scale).astype(np.float32)

    top_return_raw = float(np.percentile(ref_returns, 90))
    target_rtg     = top_return_raw / rtg_scale

    return all_trajectories, state_mean, state_std, rtg_scale, target_rtg


# ===================================================================
#  PYTORCH DATASET
# ===================================================================
class DecisionTransformerDataset(Dataset):
    def __init__(self, trajectories, context_len=CONTEXT_LEN, samples_per_traj=5):
        self.trajectories = trajectories
        self.context_len  = context_len
        self.samples      = samples_per_traj

    def __len__(self):
        return len(self.trajectories) * self.samples

    def __getitem__(self, idx):
        traj = self.trajectories[idx % len(self.trajectories)]
        n = len(traj['states'])
        start = 0 if n <= self.context_len else np.random.randint(0, n - self.context_len)
        end = start + self.context_len

        s, a = traj['states'][start:end], traj['actions'][start:end]
        r, t = traj['returns_to_go'][start:end], traj['timesteps'][start:end]

        c = s.shape[0]
        if c < self.context_len:
            p = self.context_len - c
            s = np.pad(s, ((0,p),(0,0)), 'constant')
            a = np.pad(a, (0,p), 'constant')
            r = np.pad(r, ((0,p),(0,0)), 'constant')
            t = np.pad(t, (0,p), 'constant')

        return {
            'states':       torch.tensor(s, dtype=torch.float32),
            'actions':      torch.tensor(a, dtype=torch.long),
            'returns_to_go': torch.tensor(r, dtype=torch.float32),
            'timesteps':    torch.tensor(t, dtype=torch.long),
        }


# ===================================================================
#  TRAINING
# ===================================================================
def train():
    t0 = time.time()
    print("=" * 60)
    print("  TRAFFIC DECISION TRANSFORMER — NETWORK TRAINING")
    print("=" * 60)

    # ---- Data ----
    print("\n[1/3] Collecting training data …")
    gen_trajs = generate_network_trajectories(num_episodes=300, max_steps=MAX_STEPS)
    expert_trajs = load_expert_trajectories('expert_dataset.pt', chunk_len=MAX_STEPS)
    all_trajs = gen_trajs + expert_trajs

    gen_rets = [t['returns_to_go'][0, 0] for t in gen_trajs]
    print(f"  Generated: {len(gen_trajs)} trajectories")
    print(f"    returns: mean={np.mean(gen_rets):.0f}  best={np.max(gen_rets):.0f}  worst={np.min(gen_rets):.0f}")
    if expert_trajs:
        exp_rets = [t['returns_to_go'][0, 0] for t in expert_trajs]
        print(f"  Expert:    {len(expert_trajs)} trajectories  (best={np.max(exp_rets):.0f})")
    print(f"  Total:     {len(all_trajs)} trajectories")

    # ---- Normalize ----
    print("\n[2/3] Normalizing …")
    all_trajs, s_mean, s_std, rtg_scale, target_rtg = normalize_data(all_trajs, gen_trajs)
    print(f"  rtg_scale  = {rtg_scale:.1f}")
    print(f"  target_rtg = {target_rtg:.3f}  (less negative = better; best={np.max(gen_rets)/rtg_scale:.1f}, worst={np.min(gen_rets)/rtg_scale:.1f})")

    # Save a training_id so evaluate.py can detect model/stats mismatch
    training_id = str(int(time.time()))
    np.savez('norm_stats.npz',
             state_mean=s_mean, state_std=s_std,
             rtg_scale=rtg_scale, target_rtg=target_rtg,
             training_id=training_id)

    # ---- Model ----
    dataset    = DecisionTransformerDataset(all_trajs, CONTEXT_LEN, samples_per_traj=5)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficDecisionTransformer(
        state_dim=10, act_dim=2,
        hidden_size=128, max_length=CONTEXT_LEN,
        num_layers=3, num_heads=4,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ---- Train ----
    epochs = 60
    print(f"\n[3/3] Training on {device} — {len(dataset)} samples/epoch, {epochs} epochs")
    print("      (saves checkpoint every 10 epochs, safe to Ctrl+C after any checkpoint)")

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
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  ({elapsed:.0f}s)")
            # Checkpoint — save model with training_id
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_id': training_id,
                'epoch': epoch + 1,
            }, 'dt_traffic_model.pth')

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_id': training_id,
        'epoch': epochs,
    }, 'dt_traffic_model.pth')
    print(f"\n✓ Saved dt_traffic_model.pth  (training_id={training_id})")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    train()
