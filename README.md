# 6IX STREETS: Urban Traffic Optimization via Decision Transformers

## 🚦 Inspiration
The current landscape of AI is heavily saturated with LLM-based applications, but we wanted to push the boundaries of what machine learning can do for physical infrastructure. We were inspired by the challenge of **Urban Optimization**—specifically, the fact that traffic congestion isn't just a data problem, it’s a community problem. Every minute a car spends idling is time taken away from families and carbon added to our local environment. Our goal was to move away from text-generation and build a functional **Decision Transformer** capable of managing the fluid, high-stakes dynamics of a city grid to improve the daily lives of those in our community.

## 🏙️ What it does
**6IX STREETS** is a real-time traffic optimization engine designed currently for a 36-node (6x6) connected grid. Using an offline Reinforcement Learning approach, the system analyzes a high-dimensional state space—including queue lengths, wait times, and downstream road capacities—to optimize light phases across the entire network. The system aims to move beyond reactive, sensor-based timers and instead treat the city as a single, interconnected organism, prioritizing the "flushing" of high-density corridors to prevent gridlock before it starts.

## 🏗️ How we built it
We prioritized building a custom, high-fidelity environment to ensure our results were grounded in realistic physics:
* **The Physics Engine:** Developed a custom `GridEnv` in Python that models 36 nodes. Unlike simple simulations, this environment accounts for backpressure (downstream congestion stopping upstream flow) and saturation flow rates.
* **The Brain:** Implemented a **Decision Transformer** architecture using PyTorch. By treating Reinforcement Learning as a sequence-modeling problem, we utilized **Returns-to-Go** \\((R_t)\\) to allow the model to learn long-term strategies from complex datasets.
* **The Data Pipeline:** To train the model effectively, we generated a comprehensive dataset of **10,000 steps** (360,000 data points) using a mathematical heuristic. This provided the necessary "expert trajectories" for the Transformer to learn the optimal relationship between traffic states and actions.
* **The Tech Stack:** Built with a **Python/PyTorch** backend for the heavy lifting, visualized through a modern **React and Tailwind CSS** dashboard that maps out the grid's density in real-time.

## 🚧 Challenges we ran into
The primary challenge was managing the **Density Saturation** of a 36-node network. We discovered that in a high-volume "Rush Hour" simulation, the roads often hit their physical capacity. Achieving a **2.5% improvement** over our baseline was a significant technical hurdle; it required multiple iterations of our reward function to ensure the model didn't collapse when wait times grew exponentially. We had to balance the "local" needs of a single intersection with the "global" needs of the entire 6x6 grid, which often led to conflicting gradients during the training process.

## 🏆 Accomplishments that we're proud of
* **Architectural Depth:** We successfully moved beyond simple heuristics to implement a custom **Decision Transformer**, a sophisticated approach to offline Reinforcement Learning.
* **Network Routing Logic:** Engineering the logic for cars to "flow" through 36 nodes—where the exit of one node is the entrance of another—was a complex backend feat that we executed successfully.
* **Problem Pivoting:** We were proud of our ability to analyze our 2.5% improvement results and realize that in a saturated system, even small gains represent a massive shift in overall grid stability.

## 🎓 What we learned
Building 6IX STREETS gave us deep insights into **Offline Reinforcement Learning** and the intricacies of high-density traffic physics. We learned how to structure state vectors for a 36-node system and the importance of **data normalization** when dealing with large-scale rewards. Most importantly, we learned that true urban optimization requires a holistic view of the network; you cannot solve traffic at Node A without understanding the capacity at Node B.

## 🚀 What's next for 6IX STREETS
* **Multi-Agent Scaling:** Transitioning to a multi-agent transformer setup where each node functions as an autonomous agent communicating with its neighbors.
* **Real-World Topographies:** Porting the engine to support irregular grid layouts based on actual GIS data from major metropolitan areas.
* **Multi-Modal Traffic:** Integrating public transit and emergency vehicle prioritization into the AI’s decision-making matrix.
