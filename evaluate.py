import numpy as np
import torch
import matplotlib.pyplot as plt
from SimplifiedIntersection import SimplifiedIntersection
from agent import TrafficDecisionTransformer

CONTEXT_LEN = 20  # Must match train.py and agent.py

def run_baseline_episode(steps=200):
    """Run a fixed-timer baseline: switch every 15 ticks."""
    env = SimplifiedIntersection(1)
    env.reset()
    wait_times = []
    
    timer = 0
    for _ in range(steps):
        timer += 1
        if timer >= 15:
            action = 1
            timer = 0
        else:
            action = 0
            
        _, reward = env.step(action)
        wait_times.append(np.mean(env.wait_times))
        
    return wait_times

def run_ai_episode(model_path, steps=200):
    """Run the trained Decision Transformer policy."""
    env = SimplifiedIntersection(1)
    state = env.reset()
    wait_times = []
    
    # Load normalization stats (saved during training)
    norm = np.load('norm_stats.npz')
    state_mean = norm['state_mean']
    state_std = norm['state_std']
    rtg_mean = float(norm['rtg_mean'])
    rtg_std = float(norm['rtg_std'])
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficDecisionTransformer(
        state_dim=10, 
        act_dim=2, 
        hidden_size=128, 
        max_length=CONTEXT_LEN,
        num_layers=3, 
        num_heads=4
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Target return: we want the BEST possible outcome.
    # In normalized space, a "top trajectory" has a high normalized return.
    # We'll target +2.0 standard deviations above mean (aggressive but realistic)
    target_return_normalized = 2.0
    
    # Normalize the initial state
    norm_state = ((state - state_mean) / state_std).astype(np.float32)
    
    # Initialize auto-regressive sequences
    states_seq = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    actions_seq = torch.zeros((1, 1), dtype=torch.long).to(device)
    returns_seq = torch.tensor([target_return_normalized], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    timesteps_seq = torch.tensor([0], dtype=torch.long).unsqueeze(0).to(device)
    
    for t in range(steps):
        # 1. Get action from model
        action = model.get_action(states_seq, actions_seq, returns_seq, timesteps_seq)
        
        # 2. Step environment
        next_state, reward = env.step(action)
        wait_times.append(np.mean(env.wait_times))
        
        # 3. Update the action we just took
        actions_seq[0, -1] = action
        
        # 4. Calculate the next target return (in normalized space)
        reward_normalized = (reward - rtg_mean / 100.0) / rtg_std  # approximate
        # Simpler: just keep the target return high so the model keeps trying
        next_return = returns_seq[0, -1, 0].item()
        
        if t < steps - 1:
            # Normalize next state
            norm_next = ((next_state - state_mean) / state_std).astype(np.float32)
            
            next_state_tensor = torch.tensor(norm_next, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            states_seq = torch.cat([states_seq, next_state_tensor], dim=1)
            
            next_action_tensor = torch.zeros((1, 1), dtype=torch.long).to(device)
            actions_seq = torch.cat([actions_seq, next_action_tensor], dim=1)
            
            next_return_tensor = torch.tensor([next_return], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            returns_seq = torch.cat([returns_seq, next_return_tensor], dim=1)
            
            next_timestep_tensor = torch.tensor([min(t + 1, CONTEXT_LEN - 1)], dtype=torch.long).unsqueeze(0).to(device)
            timesteps_seq = torch.cat([timesteps_seq, next_timestep_tensor], dim=1)
            
            # Prune to context window
            states_seq = states_seq[:, -CONTEXT_LEN:]
            actions_seq = actions_seq[:, -CONTEXT_LEN:]
            returns_seq = returns_seq[:, -CONTEXT_LEN:]
            timesteps_seq = timesteps_seq[:, -CONTEXT_LEN:]

    return wait_times

def evaluate():
    num_runs = 5
    steps = 200
    
    print(f"Running {num_runs} evaluation episodes ({steps} steps each)...")
    
    all_baseline = []
    all_ai = []
    
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...")
        baseline = run_baseline_episode(steps=steps)
        ai = run_ai_episode(model_path="dt_traffic_model.pth", steps=steps)
        all_baseline.append(baseline)
        all_ai.append(ai)
    
    # Average across runs
    avg_baseline = np.mean(all_baseline, axis=0)
    avg_ai = np.mean(all_ai, axis=0)
    
    print(f"\nResults (averaged over {num_runs} runs):")
    print(f"  Baseline Average Wait Time: {np.mean(avg_baseline):.2f}")
    print(f"  AI Average Wait Time:       {np.mean(avg_ai):.2f}")
    improvement = (1 - np.mean(avg_ai) / np.mean(avg_baseline)) * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Individual runs (faded)
    for i in range(num_runs):
        plt.plot(all_baseline[i], color="red", alpha=0.15)
        plt.plot(all_ai[i], color="blue", alpha=0.15)
    
    # Averages (bold)
    plt.plot(avg_baseline, label=f"Standard Timed Lights (avg={np.mean(avg_baseline):.1f})", color="red", linewidth=2)
    plt.plot(avg_ai, label=f"Decision Transformer AI (avg={np.mean(avg_ai):.1f})", color="blue", linewidth=2)
    
    plt.title("Traffic Wait Times: AI vs Standard Timed Lights")
    plt.xlabel("Simulation Timesteps")
    plt.ylabel("Average Wait Time (Queued Cars)")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig("results.png", dpi=150)
    print("\nGraph saved as 'results.png'")

if __name__ == "__main__":
    evaluate()
