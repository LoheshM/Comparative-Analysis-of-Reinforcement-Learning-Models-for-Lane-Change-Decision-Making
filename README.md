Abstract

This research focuses on developing an intelligent lane-changing system for autonomous vehicles using reinforcement learning (RL) techniques. The goal is to enable vehicles to make optimal lane-changing decisions based on surrounding traffic conditions. The system is implemented and evaluated in the CARLA simulator, simulating realistic road networks and traffic scenarios.

The research formulates the lane-changing problem as an RL task, training an agent to learn a policy based on the vehicle's state and surrounding traffic. The state includes factors like distances and velocities of neighboring vehicles and the vehicle's own speed. The RL agent interacts with the environment, selecting actions (e.g., stay, change left, change right) and receiving rewards based on performance. Training involves iterative learning, aiming to maximize cumulative rewards over time for safe and efficient lane-changing behavior.

Introduction

Reinforcement learning (RL) offers a promising approach for enhancing autonomous driving, particularly in lane-changing maneuvers. Traditional rule-based methods often lack adaptability to handle dynamic traffic scenarios, prompting interest in RL techniques.

This study explores RL's application in developing an intelligent lane-changing system. Using the CARLA simulator, the goal is to train RL agents to make informed decisions on lane changes, considering safety, efficiency, and traffic rules.

Methodology and Implementation

The methodology involves several steps:

Environment Setup: Initializing simulation variables and configurations.
![image](https://github.com/LoheshM/Comparative-Analysis-of-Reinforcement-Learning-Models-for-Lane-Change-Decision-Making/assets/116341584/60e59f5e-bc55-4f44-bcc8-5ee53235ebaf)

Grid Creation and Actor Analysis: Representing the environment and analyzing surrounding traffic.
State Representation: Extracting relevant information for decision-making.
Traffic Manager Initialization: Configuring traffic scenarios.
Action Execution and Reward Calculation: Selecting actions and computing rewards.

Dynamic Vehicle Spawning: Introducing variability in traffic conditions.
Termination and Episode Handling: Monitoring episode completion criteria.

Position Calculation: Determining the vehicle's position.
Learning and Decision-Making: Training RL agents.
Evaluation and Performance Metrics: Assessing agent performance using various metrics.
The implementation involves developing DQN, Double DQN, and Dueling Double DQN models using Python, TensorFlow, and Keras. Each model undergoes iterative training and evaluation in the CARLA simulator environment.

Tools Used

Key tools utilized include CARLA simulator, Python, TensorFlow, Keras, OpenCV, and Matplotlib. These tools enable simulation, model development, training, and evaluation.

Results

The results compare the performance of DQN, Double DQN, and Dueling Double DQN models. The Double DQN model demonstrated consistent and smooth lane changes, while the Dueling Double DQN model surpassed others in challenging scenarios.
DQN Model
![image](https://github.com/LoheshM/Comparative-Analysis-of-Reinforcement-Learning-Models-for-Lane-Change-Decision-Making/assets/116341584/69c617d8-3455-40e7-bf09-8acfa2ead2f7)

Double DQN Model
![image](https://github.com/LoheshM/Comparative-Analysis-of-Reinforcement-Learning-Models-for-Lane-Change-Decision-Making/assets/116341584/72bfb2ae-bd3d-413b-a6c8-2d20056ede6b)

Dueling Double DQN (D3QN) Model
![image](https://github.com/LoheshM/Comparative-Analysis-of-Reinforcement-Learning-Models-for-Lane-Change-Decision-Making/assets/116341584/464eed43-cc38-41d0-a58a-f597518d9209)

Conclusion

The research highlights the effectiveness of RL in improving autonomous vehicle lane-changing behavior. The Dueling Double DQN model shows significant promise for real-world implementation, offering adaptability and intelligence in handling complex traffic situations. This study contributes to advancing autonomous driving technology, paving the way for safer and more efficient self-driving vehicles.
