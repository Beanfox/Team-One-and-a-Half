import numpy as np
import torch
import matplotlib.pyplot as plt
from env.traffic_env import GridEnv
from agent import TrafficDecisionTransformer

CONTEXT_LEN = 20   # Must match train.py / agent.py
NUM_NODES   = 36   # 6×6 grid
EVAL_STEPS  = 200  # Must match MAX_STEPS in train.py


# ---------------------------------------------------------------------------
# BASELINE — fixed 15-tick timer on every intersection
# ---------------------------------------------------------------------------
def run_baseline_episode(steps=EVAL_STEPS):
    env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)
    wait_times = []
    timers = [0] * NUM_NODES

    for _ in range(steps):
        actions = []
        for n in range(NUM_NODES):
            timers[n] += 1
            if timers[n] >= 15:
                actions.append(1); timers[n] = 0
            else:
                actions.append(0)
        env.step(actions)
        wait_times.append(np.mean([np.mean(i.wait_times) for i in env.intersections]))

    return wait_times


# ---------------------------------------------------------------------------
# AI POLICY — Decision Transformer controlling all 36 nodes
# ---------------------------------------------------------------------------
def run_ai_episode(model_path, steps=EVAL_STEPS):
    env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)
    wait_times = []

    # ---- Load normalization stats ----
    norm       = np.load('norm_stats.npz')
    state_mean = norm['state_mean']
    state_std  = norm['state_std']
    rtg_scale  = float(norm['rtg_scale'])
    target_rtg = float(norm['target_rtg'])    # top-10% episode return, already scaled
    print(f"    target_rtg (scaled) = {target_rtg:.3f}")

    # ---- Load model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TrafficDecisionTransformer(
        state_dim=10, act_dim=2,
        hidden_size=128, max_length=CONTEXT_LEN,
        num_layers=3, num_heads=4,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ---- Per-node autoregressive context buffers ----
    initial_states = [i.get_state_vector() for i in env.intersections]
    buf_s, buf_a, buf_r, buf_t = [], [], [], []

    for n in range(NUM_NODES):
        ns = ((initial_states[n] - state_mean) / state_std).astype(np.float32)
        buf_s.append(torch.tensor(ns, dtype=torch.float32).reshape(1, 1, -1).to(device))
        buf_a.append(torch.zeros(1, 1, dtype=torch.long, device=device))
        buf_r.append(torch.tensor([[[target_rtg]]], dtype=torch.float32, device=device))
        buf_t.append(torch.zeros(1, 1, dtype=torch.long, device=device))

    # ---- Run ----
    for step in range(steps):
        # 1.  Get actions
        actions = []
        for n in range(NUM_NODES):
            a = model.get_action(buf_s[n], buf_a[n], buf_r[n], buf_t[n])
            actions.append(a)

        # 2.  Step env
        new_states, rewards = env.step(actions)
        wait_times.append(np.mean([np.mean(i.wait_times) for i in env.intersections]))

        # 3.  Update buffers
        for n in range(NUM_NODES):
            # Fill in the real action we just took
            buf_a[n][0, -1] = actions[n]

            # Decrement RTG by the (scale-only normalised) reward
            cur_rtg    = buf_r[n][0, -1, 0].item()
            next_rtg   = cur_rtg - rewards[n] / rtg_scale

            if step < steps - 1:
                ns = ((new_states[n] - state_mean) / state_std).astype(np.float32)
                buf_s[n] = torch.cat([buf_s[n],
                    torch.tensor(ns, dtype=torch.float32).reshape(1, 1, -1).to(device)], dim=1)
                buf_a[n] = torch.cat([buf_a[n],
                    torch.zeros(1, 1, dtype=torch.long, device=device)], dim=1)
                buf_r[n] = torch.cat([buf_r[n],
                    torch.tensor([[[next_rtg]]], dtype=torch.float32, device=device)], dim=1)
                buf_t[n] = torch.cat([buf_t[n],
                    torch.tensor([[min(step + 1, CONTEXT_LEN - 1)]],
                                 dtype=torch.long, device=device)], dim=1)

                # Prune to context window
                buf_s[n] = buf_s[n][:, -CONTEXT_LEN:]
                buf_a[n] = buf_a[n][:, -CONTEXT_LEN:]
                buf_r[n] = buf_r[n][:, -CONTEXT_LEN:]
                buf_t[n] = buf_t[n][:, -CONTEXT_LEN:]

    return wait_times


# ---------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------
def evaluate():
    num_runs = 3
    print(f"Running {num_runs} episodes  ({EVAL_STEPS} steps, {NUM_NODES} intersections) …")

    all_bl, all_ai = [], []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}")
        all_bl.append(run_baseline_episode())
        all_ai.append(run_ai_episode('dt_traffic_model.pth'))

    avg_bl = np.mean(all_bl, axis=0)
    avg_ai = np.mean(all_ai, axis=0)

    bl_val = np.mean(avg_bl)
    ai_val = np.mean(avg_ai)
    pct    = (1 - ai_val / bl_val) * 100

    print(f"\nResults (averaged over {num_runs} runs, {NUM_NODES} intersections):")
    print(f"  Baseline Avg Wait: {bl_val:.2f}")
    print(f"  AI Avg Wait:       {ai_val:.2f}")
    print(f"  Improvement:       {pct:.1f}%")

    # ---- Plot ----
    plt.figure(figsize=(12, 6))
    for i in range(num_runs):
        plt.plot(all_bl[i], color='red',  alpha=0.15)
        plt.plot(all_ai[i], color='blue', alpha=0.15)
    plt.plot(avg_bl, label=f'Standard Timed Lights (avg={bl_val:.0f})',
             color='red', linewidth=2)
    plt.plot(avg_ai, label=f'Decision Transformer AI (avg={ai_val:.0f})',
             color='blue', linewidth=2)
    plt.title('Network-Wide Traffic Wait Times: AI vs Timed Lights (6×6 Grid)')
    plt.xlabel('Simulation Step')
    plt.ylabel('Avg Wait Time (36 intersections)')
    plt.legend(fontsize=11)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Graph saved as results.png")


if __name__ == '__main__':
    evaluate()
