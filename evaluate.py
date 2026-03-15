import argparse
import json
import time
from pathlib import Path
import tempfile

import numpy as np
from env.traffic_env import GridEnv

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CONTEXT_LEN = 20
NUM_NODES   = 36
EVAL_STEPS  = 200
LIVE_UI_SLEEP_SECONDS = 0.5

BRIDGE_PATH = Path('evaluate_bridge.json')
PUBLIC_BRIDGE_PATH = Path('frontend/public/evaluate_bridge.json')
LIVE_UI_LOCK_PATH = Path(tempfile.gettempdir()) / 'trafficbot_evaluate_live_ui.lock'

_LIVE_UI_LOCK_HANDLE = None


def _write_json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, encoding='utf-8') as temp_file:
        json.dump(data, temp_file, indent=2)
        temp_path = Path(temp_file.name)

    try:
        temp_path.replace(path)
    except PermissionError:
        with path.open('w', encoding='utf-8') as fallback_file:
            json.dump(data, fallback_file, indent=2)
        temp_path.unlink(missing_ok=True)


def acquire_live_ui_lock():
    global _LIVE_UI_LOCK_HANDLE

    if _LIVE_UI_LOCK_HANDLE is not None:
        return

    lock_file = LIVE_UI_LOCK_PATH.open('a+', encoding='utf-8')

    try:
        import msvcrt

        lock_file.seek(0, 2)
        if lock_file.tell() == 0:
            lock_file.write('0')
            lock_file.flush()

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        _LIVE_UI_LOCK_HANDLE = lock_file
    except Exception:
        lock_file.close()
        raise RuntimeError(
            'Another evaluate.py --live-ui process is already running. Stop it before starting a new one.'
        )


def write_ui_bridge(ui_data: dict):
    _write_json_atomic(BRIDGE_PATH, ui_data)
    _write_json_atomic(PUBLIC_BRIDGE_PATH, ui_data)


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
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for AI evaluation. Install torch before running evaluate.py.")

    from agent import TrafficDecisionTransformer

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


def run_ai_live_stream(model_path='dt_traffic_model.pth', sleep_seconds=LIVE_UI_SLEEP_SECONDS):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for live AI UI streaming. Install torch before running evaluate.py --live-ui.")

    acquire_live_ui_lock()

    from agent import TrafficDecisionTransformer

    env = GridEnv(num_intersections=NUM_NODES, grid_cols=6)

    norm       = np.load('norm_stats.npz', allow_pickle=True)
    state_mean = norm['state_mean']
    state_std  = norm['state_std']
    rtg_scale  = float(norm['rtg_scale'])
    target_rtg = float(norm['target_rtg'])
    norm_tid   = str(norm['training_id']) if 'training_id' in norm else None

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
        if norm_tid and model_tid and norm_tid != model_tid:
            print('    ⚠ WARNING: model and norm_stats are from DIFFERENT training runs!')
            print(f'      norm_stats training_id = {norm_tid}')
            print(f'      model training_id      = {model_tid}')
            print('      → You MUST re-run train.py to completion before evaluating.')
        print(f'    Model epoch: {model_epoch}')
    else:
        model.load_state_dict(checkpoint)
        print('    ⚠ Legacy model format — no mismatch detection available')

    model.eval()
    print(f'    target_rtg = {target_rtg:.3f}  rtg_scale = {rtg_scale:.1f}')

    init = [i.get_state_vector() for i in env.intersections]
    buf_s, buf_a, buf_r, buf_t = [], [], [], []
    for n in range(NUM_NODES):
        ns = ((init[n] - state_mean) / state_std).astype(np.float32)
        buf_s.append(torch.tensor(ns, dtype=torch.float32).reshape(1, 1, -1).to(device))
        buf_a.append(torch.zeros(1, 1, dtype=torch.long, device=device))
        buf_r.append(torch.tensor([[[target_rtg]]], dtype=torch.float32, device=device))
        buf_t.append(torch.zeros(1, 1, dtype=torch.long, device=device))

    print('Streaming live AI evaluation data for UI... Press Ctrl+C to stop.')

    step = 0
    try:
        while True:
            actions = []
            for n in range(NUM_NODES):
                a = model.get_action(buf_s[n], buf_a[n], buf_r[n], buf_t[n])
                actions.append(a)

            new_states, rewards = env.step(actions)
            ui_data = env.get_ui_data()
            write_ui_bridge(ui_data)

            for n in range(NUM_NODES):
                buf_a[n][0, -1] = actions[n]
                cur_rtg = buf_r[n][0, -1, 0].item()
                next_rtg = cur_rtg - rewards[n] / rtg_scale

                ns = ((new_states[n] - state_mean) / state_std).astype(np.float32)
                buf_s[n] = torch.cat([
                    buf_s[n],
                    torch.tensor(ns, dtype=torch.float32).reshape(1, 1, -1).to(device),
                ], dim=1)
                buf_a[n] = torch.cat([
                    buf_a[n],
                    torch.zeros(1, 1, dtype=torch.long, device=device),
                ], dim=1)
                buf_r[n] = torch.cat([
                    buf_r[n],
                    torch.tensor([[[next_rtg]]], dtype=torch.float32, device=device),
                ], dim=1)
                buf_t[n] = torch.cat([
                    buf_t[n],
                    torch.tensor([[min(step + 1, CONTEXT_LEN - 1)]], dtype=torch.long, device=device),
                ], dim=1)

                buf_s[n] = buf_s[n][:, -CONTEXT_LEN:]
                buf_a[n] = buf_a[n][:, -CONTEXT_LEN:]
                buf_r[n] = buf_r[n][:, -CONTEXT_LEN:]
                buf_t[n] = buf_t[n][:, -CONTEXT_LEN:]

            step += 1
            if step % 5 == 0:
                print(f'Completed {step} live evaluation steps. Latest UI time: {ui_data.get("time")}')

            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print('\nLive evaluation stream stopped by user.')


# ===================================================================
#  EVALUATE
# ===================================================================
def evaluate():
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for evaluate() plotting. Install matplotlib before running offline evaluation.")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-ui', action='store_true', help='Stream AI evaluation output for frontend UI polling')
    parser.add_argument('--model-path', default='dt_traffic_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--sleep', type=float, default=LIVE_UI_SLEEP_SECONDS, help='Seconds between live evaluation steps')
    args = parser.parse_args()

    if args.live_ui:
        run_ai_live_stream(model_path=args.model_path, sleep_seconds=args.sleep)
    else:
        evaluate()
