import time
import json
import random
import csv
from pathlib import Path
from collections import deque

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from env.traffic_env import GridEnv

MAX_TELEMETRY_ROWS = 2000

try:
    from model.agent import TrafficDecisionTransformer
    HAS_MODEL = True
except ImportError:
    try:
        from agent import TrafficDecisionTransformer
        HAS_MODEL = True
    except ImportError:
        HAS_MODEL = False
        print("Warning: TrafficDecisionTransformer not found. Using random actions.")


def write_telemetry_csv(csv_path: Path, timestamp: int, system_type: str, ui_data: dict):
    intersections = ui_data.get("intersections", {})
    if not intersections:
        return

    avg_wait_values = []
    total_queue_length = 0
    for inter in intersections.values():
        q1 = int(inter.get("q1", 0))
        q2 = int(inter.get("q2", 0))
        wait1 = float(inter.get("wait1", 0.0))
        wait2 = float(inter.get("wait2", 0.0))

        total_queue_length += q1 + q2
        avg_wait_values.extend([wait1, wait2])

    avg_wait_time_sec = sum(avg_wait_values) / len(avg_wait_values) if avg_wait_values else 0.0

    throughput_samples = []
    for inter in intersections.values():
        phase = int(inter.get("phase", 0))
        active_queue_key = "q2" if phase == 1 else "q1"
        active_queue = int(inter.get(active_queue_key, 0))
        throughput_samples.append(max(0, 5 - active_queue))

    throughput_cars_per_min = float(sum(throughput_samples))

    header = [
        "timestamp",
        "system_type",
        "avg_wait_time_sec",
        "throughput_cars_per_min",
        "total_queue_length",
    ]

    new_row = [
        timestamp,
        system_type,
        round(avg_wait_time_sec, 3),
        round(throughput_cars_per_min, 3),
        total_queue_length,
    ]

    prior_rows: list[list[str]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file_handle:
            reader = csv.reader(file_handle)
            rows = list(reader)

        if rows:
            prior_rows = rows[1:]

    recent_rows = deque(prior_rows, maxlen=MAX_TELEMETRY_ROWS)
    recent_rows.append(new_row)

    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(header)
        writer.writerows(recent_rows)


def main():
    # 3. Initialize the GridEnv with 4 intersections.
    num_nodes = 36
    env = GridEnv(num_intersections=num_nodes)
    csv_path = Path("traffic_metrics.csv")
    bridge_path = Path("data_bridge.json")
    public_bridge_path = Path("frontend/public/data_bridge.json")
    public_bridge_path.parent.mkdir(parents=True, exist_ok=True)
    
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
            actions = [random.randint(0, 1) for _ in range(num_nodes)]
            
            if HAS_MODEL and agent is not None and HAS_TORCH:
                if len(state_buffer) > 0:
                    try:
                        # Convert the current 36 intersection states into a torch tensor.
                        # state_buffer is a list of shape (seq_len, 36, state_dim)
                        state_tensor = torch.tensor(state_buffer, dtype=torch.float32)
                        
                        # Pass the sequence of states to the Agent to get 36 actions (one for each node).
                        # We assume the agent has a get_actions or forward method that can handle the tensor.
                        if hasattr(agent, 'get_actions'):
                            model_actions = agent.get_actions(state_tensor)
                        else:
                            model_actions = agent(state_tensor)
                            
                        if model_actions is not None and len(model_actions) == num_nodes:
                            actions = model_actions
                    except Exception as e:
                        # Fallback to random if model fails (e.g., mismatch in dimensions)
                        pass

            system_type = "ai_optimized" if HAS_MODEL and agent is not None and HAS_TORCH else "random_fallback"
                        
            # Pass those actions into env.step(actions).
            states, rewards = env.step(actions)
            
            # Update the state buffer to keep only the last 20 states.
            state_buffer.append(states)
            if len(state_buffer) > 20:
                state_buffer.pop(0)
                
            # Capture the 'ui_data' from env.get_ui_data().
            ui_data = env.get_ui_data()

            write_telemetry_csv(
                csv_path=csv_path,
                timestamp=int(ui_data.get("time", step_count)),
                system_type=system_type,
                ui_data=ui_data,
            )
            
            # Write this ui_data to a file named 'data_bridge.json' so the Streamlit UI can read it live.
            with bridge_path.open("w", encoding="utf-8") as bridge_file:
                json.dump(ui_data, bridge_file, indent=2)

            with public_bridge_path.open("w", encoding="utf-8") as public_bridge_file:
                json.dump(ui_data, public_bridge_file, indent=2)
                
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
