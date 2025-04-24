# ESE559-Project-2

# Deep Reinforcement Learning for Robot Navigation with Unknown Goals and Obstacles

This is an implementation of the solution for Project 2, Problem 3 of ESE 559: Special Topics in Systems and Control. The project involves training a DQN agent to navigate a robot through unknown environments towards unknown goal locations.

## Problem Description

We are training a robot to navigate in environments with the following characteristics:
- The robot starts from a fixed initial area (bottom-left corner)
- Goal locations can be in one of two regions:
  - Region 1: x ∈ [0.8, 1.0] and y ∈ [0.8, 1.0]
  - Region 2: x ∈ [-1.1, -0.9] and y ∈ [0.8, 1.0]
- The environment contains 3 cylindrical obstacles with radius r ∈ [0.16, 0.20]
- The robot's goal is to reach the target position while avoiding obstacles
- The robot is considered to have reached the goal if it arrives within a radius of 0.08 units

The key challenge in Problem 3 is that:
1. The goal location is unknown before deployment (but is provided during deployment)
2. The obstacle configuration is unknown before deployment
3. The agent must learn a policy that generalizes to any goal in the specified regions

## Project Structure

- `environment.py`: Implements the robot environment with obstacle and goal management
- `dqn_agent.py`: Implements the Deep Q-Network agent with goal-conditioned policy
- `train.py`: Contains the training loop for the DQN agent
- `test.py`: Contains functions to test the trained agent on unseen environments
- `main.py`: Main entry point for training or testing the agent

## Dependencies

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm
- project2 wheel file (provided with assignment)

## Installation

1. Install Python dependencies:
```bash
pip install torch numpy matplotlib tqdm
```

2. Install the project2 wheel file:
```bash
pip install /path/to/project2-0.1.0-py3-none-any.whl
```

## Usage

### Training

To train the agent, run:

```bash
python main.py --mode train --episodes 2000 --seed 42 --render
```

Key training parameters:
- `--episodes`: Number of training episodes (default: 2000)
- `--batch-size`: Batch size for experience replay (default: 64)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-min`: Minimum exploration rate (default: 0.01)
- `--epsilon-decay`: Exploration decay rate (default: 0.995)
- `--hidden-dim`: Hidden dimension of DQN (default: 128)
- `--additional-envs`: Number of additional random environments to add during training (default: 2)
- `--render`: Enable environment rendering during training

### Testing

To test a trained agent, run:

```bash
python main.py --mode test --model-path results/train_YYYYMMDD_HHMMSS/dqn_model_final.pth --seed 123 --render
```

Key testing parameters:
- `--model-path`: Path to trained model file (required)
- `--num-envs`: Number of test environments to generate (default: 3)
- `--goals-per-env`: Number of goals to test per environment (default: 3)
- `--num-trials`: Number of trials per environment-goal combination (default: 10)
- `--max-steps`: Maximum steps per trial (default: 150)
- `--render`: Enable environment rendering during testing

## Solution Approach

### State Representation

We represent the state as a 5-dimensional vector combining:
- Robot position and orientation (px, py, φ)
- Goal position (gx, gy)

This goal-conditioned state representation allows the agent to learn a policy that can generalize to different goal locations.

### DQN Architecture

The DQN architecture consists of:
- Input layer: 5 neurons (state representation)
- Hidden layers: 128 → 128 → 64 neurons with ReLU activation
- Output layer: 23 neurons (Q-values for each action)

### Training Strategy

To train a policy that generalizes to different goals and obstacle configurations, we:
1. Train on the three environments provided in the problem statement
2. Generate additional random environments during training
3. Randomly sample goals from both specified regions during training
4. Use experience replay to learn from diverse experiences
5. Implement an epsilon-greedy exploration strategy with decaying epsilon

### Reward Function

We use a dense reward function to guide the learning:
- Step penalty (-0.1) to encourage shorter paths
- Progress reward (0.3 * distance_improvement) for moving toward the goal
- Large penalty (-100.0) for collisions and out-of-bounds
- Large reward (20.0) for reaching the goal
- Small penalty (-0.2) for staying still

## Results

The performance of the trained agent is evaluated based on:
1. Success rate: Percentage of trials where the robot reaches the goal without collisions
2. Average steps to goal: Average number of steps taken to reach the goal in successful trials

The expected results should show the agent's ability to navigate to various goal locations in unseen environments, demonstrating generalization to both unknown goals and unknown obstacle configurations.

## Visualization

The training and testing scripts generate visualizations of:
- Training rewards over time
- Test environments with obstacles and goals
- Robot trajectories during test trials
- Summary of test results across different environments and goals

These visualizations are saved in the results directory for analysis.