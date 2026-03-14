import time
import json
import random

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from env.traffic_env import GridEnv

try:
    from model.agent import TrafficDecisionTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("Warning: model.agent.TrafficDecisionTransformer not found. Using random actions.")

def main():
    # 3. Initialize the GridEnv with 4 intersections.
    env = GridEnv(num_intersections=4)
    
    agent = None
    if HAS_MODEL:
        try:
            agent = TrafficDecisionTransformer()
        except Exception as e:
            print(f"Error initializing agent: {e}")
            
    # 4. Setup a 'State Buffer': Since we are using a Decision Transformer, keep a list of the last 20 states.
    state_buffer = []
    
    print("Starting traffic optimization main orchestrator...")
    
    step_count = 0
    try:
        # 5. The Main Loop:
        while True:
            # 6. Error Handling: If the model file doesn't exist yet, use random actions so the loop doesn't crash.
            actions = [random.randint(0, 1) for _ in range(4)]
            
            if HAS_MODEL and agent is not None and HAS_TORCH:
                if len(state_buffer) > 0:
                    try:
                        # Convert the current 4 intersection states into a torch tensor.
                        # state_buffer is a list of shape (seq_len, 4, state_dim)
                        state_tensor = torch.tensor(state_buffer, dtype=torch.float32)
                        
                        # Pass the sequence of states to the Agent to get 4 actions (one for each node).
                        # We assume the agent has a get_actions or forward method that can handle the tensor.
                        if hasattr(agent, 'get_actions'):
                            model_actions = agent.get_actions(state_tensor)
                        else:
                            model_actions = agent(state_tensor)
                            
                        if model_actions is not None and len(model_actions) == 4:
                            actions = model_actions
                    except Exception as e:
                        # Fallback to random if model fails (e.g., mismatch in dimensions)
                        pass
                        
            # Pass those actions into env.step(actions).
            states, rewards = env.step(actions)
            
            # Update the state buffer to keep only the last 20 states.
            state_buffer.append(states)
            if len(state_buffer) > 20:
                state_buffer.pop(0)
                
            # Capture the 'ui_data' from env.get_ui_data().
            ui_data = env.get_ui_data()
            
            # Write this ui_data to a file named 'data_bridge.json' so the Streamlit UI can read it live.
            with open("data_bridge.json", "w") as f:
                json.dump(ui_data, f, indent=2)
                
            step_count += 1
            if step_count % 5 == 0:
                print(f"Completed {step_count} steps. Latest UI time: {ui_data.get('time')}")
                
            # Sleep to prevent rapid looping and high CPU usage from constant JSON writing
            time.sleep(0.5) 
            
    except KeyboardInterrupt:
        print("\nMain loop stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
