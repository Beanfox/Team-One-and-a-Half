import numpy as np
import random
import json

class SimplifiedIntersection:
    """
    Simulates a single traffic intersection with main and side streets,
    startup delays, yellow light transitions, and queue dynamics.
    """
    def __init__(self, intersection_id):
        self.id = intersection_id
        
        # Determine if this is a main street (higher volume) based on ID for varying traffic
        self.is_main_street = (intersection_id % 2 == 0)
        
        # State tracking
        self.queue_lengths = np.zeros(2, dtype=np.float32) 
        self.wait_times = np.zeros(2, dtype=np.float32) 
        
        # 0 = North-South Green (Phase 0), 1 = East-West Green (Phase 1)
        self.current_phase = 0
        self.time_in_phase = 0
        
        # Physics delays
        self.startup_delay_remaining = 0
        self.in_transition_delay = 0 
        
        # Upstream/Downstream properties for the neural network state vector
        self.upstream_incoming = np.zeros(2, dtype=np.float32)
        self.downstream_capacity = np.full(2, 50.0, dtype=np.float32) 

    def get_state_vector(self):
        """
        STRICT INTERFACE 1: 
        Must return a flat, 10-element numpy.float32 array for the PyTorch Decision Transformer.
        """
        state = np.concatenate([
            self.queue_lengths,                     # 2 elements
            self.wait_times,                        # 2 elements
            np.array([self.current_phase, self.time_in_phase], dtype=np.float32), # 2 elements
            self.upstream_incoming,                 # 2 elements
            self.downstream_capacity                # 2 elements
        ])
        return state.astype(np.float32)

    def step(self, action):
        """
        Advances the intersection by one tick.
        action = 0 (stay in current phase) or 1 (switch to next phase).
        """
        reward = 0.0

        if self.in_transition_delay > 0:
            # INTERFACE CONSTRAINT: Yellow light processing
            self.in_transition_delay -= 1
            
            # Nobody moves (no discharge), but arrivals still happen
            self._update_arrivals()
            self._update_wait_times()
            
            # REWARD CONSTRAINT: Penalty for being in transition (prevents rapid toggling)
            reward -= 5.0 
        else:
            # Normal green light operation
            if action == 1:
                # Trigger a switch to yellow -> red -> opposite green
                self.in_transition_delay = 3 # 3 ticks of yellow transition
                self.current_phase = 1 if self.current_phase == 0 else 0
                self.time_in_phase = 0
                
                # Sets the startup lost time for when green fully starts
                self.startup_delay_remaining = 2 
                reward -= 5.0 # Initial penalty for deciding to switch
            else:
                self.time_in_phase += 1

            self._update_arrivals()
            if self.in_transition_delay == 0:
                self._update_departures()
            self._update_wait_times()
            
        reward += self._calculate_state_reward()
        return self.get_state_vector(), float(reward)

    def _calculate_state_reward(self):
        """
        REWARD CONSTRAINT:
        Negative signal punishing long queues and starving directions (non-linear wait penalty).
        """
        # Linear penalty for total queue length
        queue_penalty = np.sum(self.queue_lengths) * 0.5
        
        # Exponential/Squared penalty for wait times to heavily penalize starving a side street
        clipped_wait_times = np.clip(self.wait_times, 0, 100)
        wait_penalty = np.sum((clipped_wait_times / 10.0) ** 2)
        
        return -(queue_penalty + wait_penalty)

    def _update_arrivals(self):
        """
        PHYSICS: Simulates cars arriving, biased heavily towards Main Streets.
        """
        p_main = 0.7 if self.is_main_street else 0.3
        p_side = 0.2 if self.is_main_street else 0.4
        
        arrivals = np.array([
            np.random.poisson(p_main * 3), 
            np.random.poisson(p_side * 3)
        ], dtype=np.float32)
        
        self.queue_lengths += arrivals
        self.upstream_incoming = arrivals

    def _update_departures(self):
        """
        PHYSICS: Saturation & Startup Delay.
        Cars discharge slowly at first, then speed up.
        """
        if self.queue_lengths[self.current_phase] > 0:
            if self.startup_delay_remaining > 0:
                # Driver reaction time: low flow capacity
                discharge_rate = 1.0 
                self.startup_delay_remaining -= 1
            else:
                # Full saturation flow capacity
                discharge_rate = 3.0 
                
            actual_discharge = min(self.queue_lengths[self.current_phase], discharge_rate)
            self.queue_lengths[self.current_phase] -= actual_discharge
            self.queue_lengths = np.maximum(self.queue_lengths, 0.0)
            
            # If the queue has been fully cleared out, reset the wait time for this specific lane
            if self.queue_lengths[self.current_phase] == 0:
                 self.wait_times[self.current_phase] = 0.0

    def _update_wait_times(self):
        """
        PHYSICS: Wait Time Accumulation.
        Cars waiting at a red light (or a backed up green light) accumulate delay.
        """
        self.wait_times += self.queue_lengths * 1.0
        # Sanity check: if no cars exist, wait time is exactly 0
        self.wait_times = np.where(self.queue_lengths == 0, 0.0, self.wait_times)


class GridEnv:
    """
    Multi-intersection environment manager orchestrating the individual nodes.
    Scaled to a 6x6 Grid (36 Intersections).
    """
    def __init__(self, num_intersections=36):
        self.num_intersections = num_intersections
        self.intersections = [SimplifiedIntersection(i) for i in range(num_intersections)]
        self.global_time = 0

    def step(self, actions):
        """
        STRICT INTERFACE 2: 
        Accepts a list of N actions, returns N state vectors and N rewards.
        """
        states, rewards = [], []
        
        # Ensure 'actions' list matches number of intersections
        for i, inter in enumerate(self.intersections):
            action = actions[i] if i < len(actions) else 0
            s, r = inter.step(action)
            states.append(s)
            rewards.append(r)
            
        self.global_time += 1
        return states, rewards
        
    def get_ui_data(self):
        """
        STRICT INTERFACE 3:
        Must return pure JSON-serializable Python data for the Streamlit dashboard (No Numpy!).
        CRITICAL UI JSON FORMATTING:
        Returns specific nested objects for "North_South" and "East_West" at each node.
        """
        inter_data = {}
        for idx, inter in enumerate(self.intersections):
            is_transitioning = inter.in_transition_delay > 0
            
            # Light logic: 
            # If current_phase == 0, we're targeting N-S Green.
            # If transitioning INTO phase 0, it means we WERE in phase 1 (E-W green), 
            # so E-W becomes Yellow and N-S is Red.
            if inter.current_phase == 1:
                # Phase 1 is East-West Green, North-South Red
                ns_light = "Yellow" if is_transitioning else "Red"
                ew_light = "Red" if is_transitioning else "Green"
            else:
                # Phase 0 is North-South Green, East-West Red
                ns_light = "Red" if is_transitioning else "Green"
                ew_light = "Yellow" if is_transitioning else "Red"

            inter_data[f"node_{idx}"] = {
                "North_South": {
                    "light": ns_light,
                    "queue": int(inter.queue_lengths[0]),
                    "wait": float(inter.wait_times[0])
                },
                "East_West": {
                    "light": ew_light,
                    "queue": int(inter.queue_lengths[1]),
                    "wait": float(inter.wait_times[1])
                }
            }
            
        return {
            "time": int(self.global_time), 
            "intersections": inter_data
        }

# --- THE EXECUTION BLOCK ---
if __name__ == "__main__":
    # Test setting up the 36-node grid
    env = GridEnv(num_intersections=36)
    print("Environment Live. Running Initialization Test...")
    for step_num in range(5):
        # Pass a list of random actions (0 or 1) for 36 nodes
        res_states, res_rewards = env.step([1 if random.random() < 0.2 else 0 for _ in range(36)])
        print(f"Step {step_num} Rewards Sample (first 5):", res_rewards[:5])
    
    print("\nFinal UI Data Dump (Sample single node):")
    sample_data = env.get_ui_data()
    print(json.dumps({"time": sample_data["time"], "node_0": sample_data["intersections"]["node_0"]}, indent=2))