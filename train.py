import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from SimplifiedIntersection import SimplifiedIntersection
from agent import TrafficDecisionTransformer
import torch.nn as nn

CONTEXT_LEN = 20  # The context window K used everywhere

# --- 1. Offline Data Generation ---
def generate_trajectories(num_episodes=2000, max_steps=100):
    """
    Generate diverse offline trajectories using multiple behavior policies.
    The key insight for Decision Transformers: we need trajectories of
    VARYING quality so the model can learn that higher return-to-go
    correlates with better actions.
    """
    trajectories = []
    
    for i in range(num_episodes):
        env = SimplifiedIntersection(1)
        state = env.reset()
        
        states, actions, rewards = [], [], []
        
        # Use a VARIETY of switching intervals to create diverse quality data
        # Some will be good (switch every ~8-12 ticks), some bad (too fast or too slow)
        mode = i % 5
        timer = 0
        
        if mode == 0:
            # Random switching (bad policy)
            switch_interval = None  # purely random
        elif mode == 1:
            # Very fast switching (bad - too much yellow time)
            switch_interval = np.random.randint(3, 6)
        elif mode == 2:
            # Good switching interval
            switch_interval = np.random.randint(8, 15)
        elif mode == 3:
            # Slow switching (bad - one street starves)
            switch_interval = np.random.randint(25, 40)
        else:
            # Adaptive-ish: switch when the red street queue > green queue
            switch_interval = -1  # sentinel for adaptive
        
        for step in range(max_steps):
            states.append(state)
            
            if switch_interval is None:
                # Random policy
                action = np.random.choice([0, 1], p=[0.75, 0.25])
            elif switch_interval == -1:
                # Adaptive: switch if the waiting street has more cars
                green_idx = env.current_phase
                red_idx = 1 - green_idx
                if env.queue_lengths[red_idx] > env.queue_lengths[green_idx] + 2 and env.time_in_phase >= 5:
                    action = 1
                elif env.time_in_phase >= 20:
                    action = 1  # don't let one side starve completely
                else:
                    action = 0
            else:
                # Timed policy with the given interval
                timer += 1
                if timer >= switch_interval:
                    action = 1
                    timer = 0
                else:
                    action = 0
                    
            state, reward = env.step(action)
            actions.append(action)
            rewards.append(reward)
            
        # Calculate Returns-to-Go (sum of future rewards)
        # Normalize rewards to prevent huge magnitudes
        rewards_arr = np.array(rewards, dtype=np.float32)
        
        returns_to_go = np.zeros_like(rewards_arr)
        ret = 0
        for t in reversed(range(len(rewards_arr))):
            ret += rewards_arr[t]
            returns_to_go[t] = ret
        
        trajectories.append({
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'rewards': rewards_arr,
            'returns_to_go': returns_to_go[..., None],  # (T, 1)
            'timesteps': np.arange(max_steps, dtype=np.int64)
        })
        
    return trajectories

def normalize_trajectories(trajectories):
    """
    Normalize states and returns-to-go across the entire dataset so the
    model sees values in a reasonable range.
    """
    # Gather all states and returns
    all_states = np.concatenate([t['states'] for t in trajectories], axis=0)
    all_rtg = np.concatenate([t['returns_to_go'] for t in trajectories], axis=0)
    
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0) + 1e-6  # avoid divide-by-zero
    
    rtg_mean = all_rtg.mean()
    rtg_std = all_rtg.std() + 1e-6
    
    for t in trajectories:
        t['states'] = ((t['states'] - state_mean) / state_std).astype(np.float32)
        t['returns_to_go'] = ((t['returns_to_go'] - rtg_mean) / rtg_std).astype(np.float32)
    
    return trajectories, state_mean, state_std, rtg_mean, rtg_std

# --- 2. PyTorch Dataset ---
class DecisionTransformerDataset(Dataset):
    def __init__(self, trajectories, context_len=CONTEXT_LEN, samples_per_traj=5):
        self.trajectories = trajectories
        self.context_len = context_len
        self.samples_per_traj = samples_per_traj
        
    def __len__(self):
        # Sample multiple windows per trajectory for more gradient steps
        return len(self.trajectories) * self.samples_per_traj
        
    def __getitem__(self, idx):
        traj_idx = idx % len(self.trajectories)
        traj = self.trajectories[traj_idx]
        traj_len = len(traj['states'])
        
        # Sample a random starting index for our context window
        if traj_len <= self.context_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, traj_len - self.context_len)
            
        end_idx = start_idx + self.context_len
        
        states = traj['states'][start_idx:end_idx]
        actions = traj['actions'][start_idx:end_idx]
        returns_to_go = traj['returns_to_go'][start_idx:end_idx]
        timesteps = traj['timesteps'][start_idx:end_idx]
        
        # Pad if needed
        curr_len = states.shape[0]
        if curr_len < self.context_len:
            pad_len = self.context_len - curr_len
            states = np.pad(states, ((0, pad_len), (0, 0)), 'constant')
            actions = np.pad(actions, (0, pad_len), 'constant')
            returns_to_go = np.pad(returns_to_go, ((0, pad_len), (0, 0)), 'constant')
            timesteps = np.pad(timesteps, (0, pad_len), 'constant')

        return {
            'states': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long),
            'returns_to_go': torch.tensor(returns_to_go, dtype=torch.float32),
            'timesteps': torch.tensor(timesteps, dtype=torch.long)
        }

# --- 3. Training Loop ---
def train():
    print("Generating diverse offline trajectories...")
    trajectories = generate_trajectories(num_episodes=2000, max_steps=100)
    
    # Print dataset quality stats
    episode_returns = [t['returns_to_go'][0, 0] for t in trajectories]
    print(f"  Raw return stats: mean={np.mean(episode_returns):.1f}, "
          f"best={np.max(episode_returns):.1f}, worst={np.min(episode_returns):.1f}")
    
    # Normalize
    trajectories, state_mean, state_std, rtg_mean, rtg_std = normalize_trajectories(trajectories)
    
    # Save normalization stats for evaluation
    np.savez('norm_stats.npz', 
             state_mean=state_mean, state_std=state_std,
             rtg_mean=rtg_mean, rtg_std=rtg_std)
    print("  Saved normalization stats to norm_stats.npz")
    
    dataset = DecisionTransformerDataset(trajectories, context_len=CONTEXT_LEN, samples_per_traj=5)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}, {len(dataset)} samples per epoch")
    
    model = TrafficDecisionTransformer(
        state_dim=10, 
        act_dim=2, 
        hidden_size=128, 
        max_length=CONTEXT_LEN,  # Match the context window!
        num_layers=3, 
        num_heads=4
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 50
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            returns_to_go = batch['returns_to_go'].to(device)
            timesteps = batch['timesteps'].to(device)
            
            optimizer.zero_grad()
            
            action_logits = model(states, actions, returns_to_go, timesteps)
            
            loss = criterion(
                action_logits.reshape(-1, 2), 
                actions.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    print("Training complete! Saving model...")
    torch.save(model.state_dict(), "dt_traffic_model.pth")
    print("Saved as dt_traffic_model.pth")

if __name__ == "__main__":
    train()
