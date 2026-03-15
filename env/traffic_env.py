import numpy as np
import random
import json




class SimplifiedIntersection:
   """
   Simulates a single traffic intersection with main and side streets,
   startup delays, yellow light transitions, and queue dynamics.
   Designed to plug into a connected 6x6 grid via GridEnv.
   """


   # Max queue before spillback fully blocks upstream discharge
   MAX_QUEUE_CAPACITY = 60.0


   def __init__(self, intersection_id, grid_cols=6):
       self.id = intersection_id
       self.row = intersection_id // grid_cols
       self.col = intersection_id % grid_cols


       # Determine if this is a main street (higher volume) based on ID
       self.is_main_street = (intersection_id % 2 == 0)


       # State tracking — index 0 = North/South direction, index 1 = East/West direction
       self.queue_lengths = np.zeros(2, dtype=np.float32)
       self.wait_times = np.zeros(2, dtype=np.float32)


       # 0 = North-South Green (Phase 0), 1 = East-West Green (Phase 1)
       self.current_phase = 0
       self.time_in_phase = 0


       # Physics delays
       self.startup_delay_remaining = 0
       self.in_transition_delay = 0


       # Network-aware properties for the neural network state vector
       # upstream_incoming: cars arriving from the network this tick
       # downstream_capacity: available space in the downstream neighbors
       self.upstream_incoming = np.zeros(2, dtype=np.float32)
       self.downstream_capacity = np.full(2, MAX_QUEUE_CAPACITY, dtype=np.float32)


       # Cars discharged this tick, to be routed by GridEnv
       self.discharged_this_tick = np.zeros(2, dtype=np.float32)


       # Global step counter (set by GridEnv)
       self.global_step = 0


   def get_state_vector(self):
       """
       STRICT INTERFACE 1:
       Returns a flat, 10-element numpy.float32 array for the PyTorch Decision Transformer.
       """
       state = np.concatenate([
           self.queue_lengths,                                                      # 2
           self.wait_times,                                                         # 2
           np.array([self.current_phase, self.time_in_phase], dtype=np.float32),     # 2
           self.upstream_incoming,                                                   # 2
           self.downstream_capacity                                                 # 2
       ])
       return state.astype(np.float32)


   def step(self, action, downstream_cap):
       """
       Advances the intersection by one tick.
       action = 0 (stay in current phase) or 1 (switch to next phase).
       downstream_cap: np.array([ns_cap, ew_cap]) — available space in neighbors.
       """
       self.downstream_capacity = downstream_cap.copy()
       self.discharged_this_tick[:] = 0.0
       reward = 0.0


       if self.in_transition_delay > 0:
           # Yellow light processing — no discharge, arrivals still happen
           self.in_transition_delay -= 1
           self._update_arrivals()
           self._update_wait_times()
           reward -= 5.0
       else:
           if action == 1:
               # Trigger yellow -> switch (costly!)
               self.in_transition_delay = 5        # longer yellow phase
               self.current_phase = 1 if self.current_phase == 0 else 0
               self.time_in_phase = 0
               self.startup_delay_remaining = 3    # longer startup
               reward -= 15.0                       # heavy switch penalty
           else:
               self.time_in_phase += 1


           self._update_arrivals()
           if self.in_transition_delay == 0:
               self._update_departures()
           self._update_wait_times()


       reward += self._calculate_state_reward()
       return self.get_state_vector(), float(reward)


   def add_routed_arrivals(self, direction_idx, count):
       """Called by GridEnv to inject cars routed from an upstream neighbor."""
       self.queue_lengths[direction_idx] += count


   def _calculate_state_reward(self):
       """
       Negative signal punishing long queues and starving directions.
       No clipping — the full wait-time signal is preserved so the model
       can distinguish moderate from severe congestion.
       """
       queue_penalty = np.sum(self.queue_lengths) * 0.5
       wait_penalty = np.sum((self.wait_times / 10.0) ** 1.5)
       return -(queue_penalty + wait_penalty)


   def _update_arrivals(self):
       """
       PHYSICS: Simulates *external* cars arriving (from outside the grid).
       Network-routed cars are added separately via add_routed_arrivals().
       Edge nodes get more external arrivals; interior nodes get fewer.
       """
       grid_rows = 6
       is_edge = (self.row == 0 or self.row == grid_rows - 1
                  or self.col == 0 or self.col == grid_rows - 1)


       # Strongly asymmetric traffic: N/S direction is 3-4x heavier than E/W
       # This creates a clear optimization opportunity:
       #   - A fixed timer wastes ~50% of green on the nearly-empty E/W
       #   - A smart policy gives more green to the busy N/S direction
       # Time-varying: a "rush hour" pulse makes it even more pronounced
       rush = 1.3 if (self.global_step % 100) < 50 else 0.7


       if is_edge:
           ns_rate = 0.40 * rush   # Tuned for ~15% oracle advantage
       else:
           ns_rate = 0.09 * rush


       # Enforce directional asymmetry (N/S receives ~4x E/W demand)
       ew_rate = ns_rate / 4.0


       arrivals = np.array([
           np.random.poisson(ns_rate * 3),
           np.random.poisson(ew_rate * 3)
       ], dtype=np.float32)


       self.queue_lengths += arrivals
       self.upstream_incoming = arrivals


   def _update_departures(self):
       """
       PHYSICS: Saturation & Startup Delay + Backpressure.
       Discharge is capped by the downstream neighbor's available capacity.
       """
       active = self.current_phase  # direction index with green light
       if self.queue_lengths[active] > 0:
           if self.startup_delay_remaining > 0:
               discharge_rate = 1.0
               self.startup_delay_remaining -= 1
           else:
               discharge_rate = 4.0


           # Backpressure: cap discharge by downstream capacity
           cap = max(self.downstream_capacity[active], 0.0)
           discharge_rate = min(discharge_rate, cap)


           actual_discharge = min(self.queue_lengths[active], discharge_rate)
           self.queue_lengths[active] -= actual_discharge
           self.queue_lengths = np.maximum(self.queue_lengths, 0.0)
           self.discharged_this_tick[active] = actual_discharge


           if self.queue_lengths[active] == 0:
               self.wait_times[active] = 0.0


   def _update_wait_times(self):
       """
       PHYSICS: Wait Time Accumulation (linear).
       """
       self.wait_times += self.queue_lengths * 1.0
       self.wait_times = np.where(self.queue_lengths == 0, 0.0, self.wait_times)




# Constant used by the class above (module-level for clarity)
MAX_QUEUE_CAPACITY = SimplifiedIntersection.MAX_QUEUE_CAPACITY




class GridEnv:
   """
   Multi-intersection environment manager for a connected 6x6 grid.
   Handles spatial topology, discharge routing, and backpressure.
   """


   def __init__(self, num_intersections=36, grid_cols=6):
       self.num_intersections = num_intersections
       self.grid_cols = grid_cols
       self.grid_rows = num_intersections // grid_cols
       self.intersections = [
           SimplifiedIntersection(i, grid_cols=grid_cols)
           for i in range(num_intersections)
       ]
       self.global_time = 0


       # Precompute neighbor maps for each node
       # neighbors[i] = {"north": id|None, "south": id|None, "east": id|None, "west": id|None}
       self._neighbors = self._build_neighbor_map()


   def _build_neighbor_map(self):
       """Build adjacency for a rows x cols grid."""
       neighbors = {}
       for node_id in range(self.num_intersections):
           row = node_id // self.grid_cols
           col = node_id % self.grid_cols
           neighbors[node_id] = {
               "north": node_id - self.grid_cols if row > 0 else None,
               "south": node_id + self.grid_cols if row < self.grid_rows - 1 else None,
               "west":  node_id - 1 if col > 0 else None,
               "east":  node_id + 1 if col < self.grid_cols - 1 else None,
           }
       return neighbors


   def _get_downstream_capacity(self, node_id):
       """
       Returns np.array([ns_cap, ew_cap]) representing how much room
       the downstream neighbors have for cars flowing out of this node.
       NS green routes cars north/south, EW green routes cars east/west.
       """
       nb = self._neighbors[node_id]


       # N/S capacity: average available space at north and south neighbors
       ns_caps = []
       for direction in ("north", "south"):
           nid = nb[direction]
           if nid is not None:
               q = float(self.intersections[nid].queue_lengths[0])  # NS queue of neighbor
               ns_caps.append(max(MAX_QUEUE_CAPACITY - q, 0.0))
           else:
               # Edge — infinite capacity (cars leave the grid)
               ns_caps.append(MAX_QUEUE_CAPACITY)
       ns_cap = min(ns_caps)  # bottleneck is the most constrained neighbor


       # E/W capacity: average available space at east and west neighbors
       ew_caps = []
       for direction in ("east", "west"):
           nid = nb[direction]
           if nid is not None:
               q = float(self.intersections[nid].queue_lengths[1])  # EW queue of neighbor
               ew_caps.append(max(MAX_QUEUE_CAPACITY - q, 0.0))
           else:
               ew_caps.append(MAX_QUEUE_CAPACITY)
       ew_cap = min(ew_caps)


       return np.array([ns_cap, ew_cap], dtype=np.float32)


   def step(self, actions):
       """
       STRICT INTERFACE 2:
       Accepts a list of N actions, returns N state vectors and N rewards.
       Handles discharge routing between connected nodes.
       """
       states, rewards = [], []


       # Phase 1: Compute downstream capacities, then step each intersection
       for i, inter in enumerate(self.intersections):
           action = actions[i] if i < len(actions) else 0
           inter.global_step = self.global_time
           ds_cap = self._get_downstream_capacity(i)
           s, r = inter.step(action, ds_cap)
           states.append(s)
           rewards.append(r)


       # Phase 2: Route discharged cars to downstream neighbors
       for i, inter in enumerate(self.intersections):
           nb = self._neighbors[i]


           # Cars discharged along N/S direction (index 0) -> split to north & south.
           # If one side is missing (edge), that portion leaves the grid.
           ns_discharged = inter.discharged_this_tick[0]
           if ns_discharged > 0:
               per_direction = ns_discharged / 2.0
               for dir_key in ["north", "south"]:
                   nid = nb[dir_key]
                   if nid is not None:
                       self.intersections[nid].add_routed_arrivals(0, per_direction)
                   # else: cars leave the grid on this edge


           # Cars discharged along E/W direction (index 1) -> split to east & west.
           # If one side is missing (edge), that portion leaves the grid.
           ew_discharged = inter.discharged_this_tick[1]
           if ew_discharged > 0:
               per_direction = ew_discharged / 2.0
               for dir_key in ["east", "west"]:
                   nid = nb[dir_key]
                   if nid is not None:
                       self.intersections[nid].add_routed_arrivals(1, per_direction)
                   # else: cars leave the grid on this edge


       self.global_time += 1
       return states, rewards


   def get_ui_data(self):
       """
       STRICT INTERFACE 3:
       Must return pure JSON-serializable Python data for the Streamlit dashboard.
       Uses the explicit directional North_South / East_West structure.
       """
       inter_data = {}
       for idx, inter in enumerate(self.intersections):
           is_transitioning = inter.in_transition_delay > 0


           if inter.current_phase == 1:
               # Phase 1: E-W is (or will be) Green, N-S is Red
               ns_light = "Yellow" if is_transitioning else "Red"
               ew_light = "Red" if is_transitioning else "Green"
           else:
               # Phase 0: N-S is (or will be) Green, E-W is Red
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
   env = GridEnv(num_intersections=36)
   print("Environment Live (6x6 Connected Grid). Running Initialization Test...")
   for step_num in range(10):
       res_states, res_rewards = env.step(
           [1 if random.random() < 0.2 else 0 for _ in range(36)]
       )
       print(f"Step {step_num} | Reward sample (nodes 0-4): "
             f"{[round(r, 1) for r in res_rewards[:5]]}")


   print("\nFinal UI Data Dump (node_0 sample):")
   sample = env.get_ui_data()
   print(json.dumps({
       "time": sample["time"],
       "node_0": sample["intersections"]["node_0"]
   }, indent=2))
