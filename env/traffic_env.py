import numpy as np
import random
import json

class SimplifiedIntersection:
    def __init__(self, intersection_id):
        self.id = intersection_id
        self.queue_lengths = np.zeros(2) 
        self.wait_times = np.zeros(2) 
        self.current_phase = 0
        self.time_in_phase = 0
        self.upstream_incoming = np.zeros(2)
        self.downstream_capacity = np.full(2, 50) 
        self.in_transition_delay = 0 

    def get_state_vector(self):
        return np.concatenate([
            self.queue_lengths,
            self.wait_times,
            [self.current_phase, self.time_in_phase],
            self.upstream_incoming,
            self.downstream_capacity
        ], dtype=np.float32)

    def step(self, action):
        if self.in_transition_delay > 0:
            self.in_transition_delay -= 1
            self._update_physics_during_yellow()
            return self.get_state_vector(), self._calculate_reward()

        if action == 1:
            self.in_transition_delay = 3
            self.current_phase = 1 if self.current_phase == 0 else 0
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1

        self._update_physics_normal()
        return self.get_state_vector(), self._calculate_reward()

    def _calculate_reward(self):
        local_penalty = np.sum(self.wait_times)
        network_penalty = np.sum(np.where(self.downstream_capacity < 5, 10, 0))
        return -(local_penalty + (0.2 * network_penalty))

    def _update_physics_normal(self):
        self.queue_lengths += np.random.randint(0, 3, size=2)
        if self.queue_lengths[self.current_phase] > 0:
            self.queue_lengths[self.current_phase] -= np.random.randint(1, 4)
            self.queue_lengths = np.maximum(self.queue_lengths, 0)
        red_phase = 1 if self.current_phase == 0 else 0
        self.wait_times[red_phase] += self.queue_lengths[red_phase] * 1.5 
        self.wait_times[self.current_phase] = 0 
        self.upstream_incoming = np.random.randint(0, 5, size=2)
        self.downstream_capacity = np.random.randint(20, 50, size=2)

    def _update_physics_during_yellow(self):
        self.queue_lengths += np.random.randint(0, 2, size=2)
        self.wait_times += self.queue_lengths * 1.5 

class GridEnv:
    def __init__(self, num_intersections=4):
        self.num_intersections = num_intersections
        self.intersections = [SimplifiedIntersection(i) for i in range(num_intersections)]
        self.global_time = 0

    def step(self, actions):
        states, rewards = [], []
        for i, inter in enumerate(self.intersections):
            s, r = inter.step(actions[i])
            states.append(s)
            rewards.append(r)
        self.global_time += 1
        return states, rewards
        
    def get_ui_data(self):
        inter_data = {}
        for idx, inter in enumerate(self.intersections):
            inter_data[f"node_{idx}"] = {
                "phase": int(inter.current_phase),
                "q1": int(inter.queue_lengths[0]), "q2": int(inter.queue_lengths[1]),
                "wait1": float(inter.wait_times[0]), "wait2": float(inter.wait_times[1])
            }
        return {"time": self.global_time, "intersections": inter_data}

# --- THE EXECUTION BLOCK ---
if __name__ == "__main__":
    env = GridEnv(num_intersections=4)
    print("Environment Live. Running Test...")
    for _ in range(3):
        res = env.step([random.randint(0, 1) for _ in range(4)])
    print(json.dumps(env.get_ui_data(), indent=2))