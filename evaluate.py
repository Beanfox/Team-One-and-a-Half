import numpy as np
import torch
import matplotlib.pyplot as plt
from env.traffic_env import GridEnv
from agent import TrafficDecisionTransformer

CONTEXT_LEN = 20
NUM_NODES   = 36
EVAL_STEPS  = 200


# ===================================================================
#  BASELINE — fixed 15-tick timer
# ===================================================================
def run_baseline_episode(steps=EVAL_STEPS):
    env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)
    waits = []
    timers = [0] * NUM_NODES
    for _ in range(steps):
        acts = []
        for n in range(NUM_NODES):
            timers[n] += 1
            if timers[n] >= 15:
                acts.append(1); timers[n] = 0
            else:
                acts.append(0)
        env.step(acts)
        waits.append(np.mean([np.mean(i.wait_times) for i in env.intersections]))
    return waits


# ===================================================================
#  AI POLICY — Decision Transformer on all 36 nodes
# ===================================================================
def run_ai_episode(model_path, steps=EVAL_STEPS):
    env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)
    waits = []

    # ---- Load norm stats ----
    norm       = np.load('norm_stats.npz', allow_pickle=True)
    state_mean = norm['state_mean']
    state_std  = norm['state_std']
    rtg_scale  = float(norm['rtg_scale'])
    target_rtg = float(norm['target_rtg'])
    norm_tid   = str(norm['training_id']) if 'training_id' in norm else None

    # ---- Load model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TrafficDecisionTransformer(
        state_dim=10, act_dim=2,
        hidden_size=128, max_length=CONTEXT_LEN,
        num_layers=3, num_heads=4,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model_tid = checkpoint.get('training_id', None)
        model_epoch = checkpoint.get('epoch', '?')
        # Mismatch detection
        if norm_tid and model_tid and norm_tid != model_tid:
            print(f"    ⚠ WARNING: model and norm_stats are from DIFFERENT training runs!")
            print(f"      norm_stats training_id = {norm_tid}")
            print(f"      model training_id      = {model_tid}")
            print(f"      → You MUST re-run train.py to completion before evaluating.")
        print(f"    Model epoch: {model_epoch}")
    else:
        # Legacy format (just state_dict)
        model.load_state_dict(checkpoint)
        print("    ⚠ Legacy model format — no mismatch detection available")

    model.eval()
    print(f"    target_rtg = {target_rtg:.3f}  rtg_scale = {rtg_scale:.1f}")

    # ---- Per-node autoregressive buffers ----
    init = [i.get_state_vector() for i in env.intersections]
    buf_s, buf_a, buf_r, buf_t = [], [], [], []
    for n in range(NUM_NODES):
        ns = ((init[n] - state_mean) / state_std).astype(np.float32)
        buf_s.append(torch.tensor(ns, dtype=torch.float32).reshape(1,1,-1).to(device))
        buf_a.append(torch.zeros(1,1, dtype=torch.long, device=device))
        buf_r.append(torch.tensor([[[target_rtg]]], dtype=torch.float32, device=device))
        buf_t.append(torch.zeros(1,1, dtype=torch.long, device=device))

    for step in range(steps):
        actions = []
        for n in range(NUM_NODES):
            a = model.get_action(buf_s[n], buf_a[n], buf_r[n], buf_t[n])
            actions.append(a)

        new_states, rewards = env.step(actions)
        waits.append(np.mean([np.mean(i.wait_times) for i in env.intersections]))

        for n in range(NUM_NODES):
            buf_a[n][0, -1] = actions[n]
            cur_rtg  = buf_r[n][0, -1, 0].item()
            next_rtg = cur_rtg - rewards[n] / rtg_scale

            if step < steps - 1:
                ns = ((new_states[n] - state_mean) / state_std).astype(np.float32)
                buf_s[n] = torch.cat([buf_s[n],
                    torch.tensor(ns, dtype=torch.float32).reshape(1,1,-1).to(device)], dim=1)
                buf_a[n] = torch.cat([buf_a[n],
                    torch.zeros(1,1, dtype=torch.long, device=device)], dim=1)
                buf_r[n] = torch.cat([buf_r[n],
                    torch.tensor([[[next_rtg]]], dtype=torch.float32, device=device)], dim=1)
                buf_t[n] = torch.cat([buf_t[n],
                    torch.tensor([[min(step+1, CONTEXT_LEN-1)]],
                                 dtype=torch.long, device=device)], dim=1)

                buf_s[n] = buf_s[n][:, -CONTEXT_LEN:]
                buf_a[n] = buf_a[n][:, -CONTEXT_LEN:]
                buf_r[n] = buf_r[n][:, -CONTEXT_LEN:]
                buf_t[n] = buf_t[n][:, -CONTEXT_LEN:]

    return waits


# ===================================================================
#  EVALUATE
# ===================================================================
def evaluate():
    num_runs = 3
    print(f"Evaluating: {num_runs} runs × {EVAL_STEPS} steps × {NUM_NODES} nodes\n")

    all_bl, all_ai = [], []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}")
        all_bl.append(run_baseline_episode())
        all_ai.append(run_ai_episode('dt_traffic_model.pth'))

    avg_bl = np.mean(all_bl, axis=0)
    avg_ai = np.mean(all_ai, axis=0)
    bl_v, ai_v = np.mean(avg_bl), np.mean(avg_ai)
    pct = (1 - ai_v / bl_v) * 100

    print(f"\n{'='*50}")
    print(f"  Baseline Avg Wait: {bl_v:.2f}")
    print(f"  AI Avg Wait:       {ai_v:.2f}")
    print(f"  Improvement:       {pct:.1f}%")
    print(f"{'='*50}")

    plt.figure(figsize=(12, 6))
    for i in range(num_runs):
        plt.plot(all_bl[i], color='red',  alpha=0.15)
        plt.plot(all_ai[i], color='blue', alpha=0.15)
    plt.plot(avg_bl, label=f'Timed Lights (avg={bl_v:.0f})', color='red', lw=2)
    plt.plot(avg_ai, label=f'DT AI (avg={ai_v:.0f})', color='blue', lw=2)
    plt.title('Network Traffic: AI vs Timed Lights (6×6 Grid, 36 intersections)')
    plt.xlabel('Step')
    plt.ylabel('Avg Wait Time')
    plt.legend(fontsize=11)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Graph → results.png")


if __name__ == '__main__':
    evaluate()
