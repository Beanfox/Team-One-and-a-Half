import numpy as np

class SimplifiedIntersection:
    def __init__(self, intersection_id):
        self.id = intersection_id
        
        # 1. LOCAL STATE (2 intersecting one-way streets)
        self.queue_lengths = np.zeros(2)  # [Street 1 Queue, Street 2 Queue]
        self.wait_times = np.zeros(2)     # [Street 1 Wait Time, Street 2 Wait Time]
        
        # Phase is strictly binary now: 0 = Street 1 Green, 1 = Street 2 Green
        self.current_phase = 0
        self.time_in_phase = 0

        # 2. NETWORK STATE ("Radar")
        self.upstream_incoming = np.zeros(2)
        self.downstream_capacity = np.full(2, 50) 

        # 3. ENVIRONMENT CONTROLS (Kept hidden from the AI)
        self.in_transition_delay = 0  # Counter for the forced yellow/red clearance phase

    def reset(self):
        self.queue_lengths = np.zeros(2)
        self.wait_times = np.zeros(2)
        self.current_phase = 0
        self.time_in_phase = 0
        self.upstream_incoming = np.zeros(2)
        self.downstream_capacity = np.full(2, 50) 
        self.in_transition_delay = 0
        return self.get_state_vector()

    def get_state_vector(self):
        """
        Outputs a lean 10-value, 1D numpy array.
        This is the exact token Person B will feed into the PyTorch model.
        """
        state = np.concatenate([
            self.queue_lengths,           # 2 values
            self.wait_times,              # 2 values
            [self.current_phase],         # 1 value (0 or 1)
            [self.time_in_phase],         # 1 value
            self.upstream_incoming,       # 2 values
            self.downstream_capacity      # 2 values
        ], dtype=np.float32)
        
        return state 

    def step(self, action):
        """
        Action Space: 0 = Keep current light, 1 = Switch lights
        """
        # --- 1. HANDLE FORCED DELAYS ---
        if self.in_transition_delay > 0:
            self.in_transition_delay -= 1
            # The AI is locked out. Lights are effectively yellow/all-red here.
            self._update_physics_during_yellow()
            return self.get_state_vector(), self._calculate_reward()

        # --- 2. HANDLE AI ACTION ---
        if action == 1:
            # AI wants to switch. Lock it out for 3 ticks to safely clear the box.
            self.in_transition_delay = 3
            self.current_phase = 1 if self.current_phase == 0 else 0
            self.time_in_phase = 0
        else:
            # AI chooses to keep the light green
            self.time_in_phase += 1

        self._update_physics_normal()

        return self.get_state_vector(), self._calculate_reward()

    def _update_physics_during_yellow(self):
        # Cars arrive but cannot pass the intersection
        arrivals = np.random.poisson(0.5, size=2)
        self.queue_lengths += arrivals
        # Increase wait times for all queued cars
        self.wait_times += self.queue_lengths

    def _update_physics_normal(self):
        # Cars arrive
        arrivals = np.random.poisson(0.5, size=2)
        self.queue_lengths += arrivals
        
        # Green light street lets cars through (up to 3 per tick, if capacity allows)
        green_idx = self.current_phase
        max_flow = min(3, self.downstream_capacity[green_idx])
        dispatched = min(self.queue_lengths[green_idx], max_flow)
        
        self.queue_lengths[green_idx] -= dispatched
        
        # Simple simulation of downstream filling/emptying
        self.downstream_capacity -= arrivals
        self.downstream_capacity = np.clip(self.downstream_capacity + 1, 0, 50)
        
        # Increase wait times for all queued cars
        self.wait_times += self.queue_lengths
        
        # Reset wait time slightly when cars flow to simulate throughput being rewarded
        if dispatched > 0:
            self.wait_times[green_idx] = max(0, self.wait_times[green_idx] - (dispatched * 2))

    def _calculate_reward(self):
        """
        Negative reward based strictly on local wait times to avoid conflicting signals.
        """
        local_penalty = np.sum(self.wait_times)
        return -local_penalty