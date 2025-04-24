# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import argparse
# import os
# import json
# from tqdm import tqdm
# from datetime import datetime
# import time

# from environment import RobotEnv
# from dqn_agent import DQNAgent

# def test_single_environment(agent, obstacles, goal, num_trials=10, max_steps=150, 
#                           render=True, output_dir=None, env_name=""):
#     """
#     Test agent on a single environment with a specific goal.
    
#     Args:
#         agent (DQNAgent): The trained DQN agent.
#         obstacles (list): List of obstacle dictionaries.
#         goal (tuple): Goal position (x, y).
#         num_trials (int): Number of trials to run.
#         max_steps (int): Maximum steps per trial.
#         render (bool): Whether to render the environment.
#         output_dir (str): Directory to save results.
#         env_name (str): Name of the environment for saving.
        
#     Returns:
#         tuple: (success_rate, avg_steps_to_goal, success_count, total_steps, successful_trials)
#     """
#     # Initialize result tracking
#     success_count = 0
#     total_steps = 0
#     successful_trials = 0
    
#     # Initialize figure for visualization if rendering
#     if render:
#         fig, ax = plt.subplots(figsize=(10, 10))
        
#         # Plot obstacles
#         for obs in obstacles:
#             circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
#             ax.add_patch(circle)
        
#         # Plot goal
#         goal_circle = plt.Circle(goal, 0.08, color='green', alpha=0.3)
#         ax.add_patch(goal_circle)
#         ax.plot(goal[0], goal[1], 'g*', markersize=15)
        
#         # Set plot limits
#         ax.set_xlim([-1.5, 1.5])
#         ax.set_ylim([-1.5, 1.5])
#         ax.set_xlabel("X Position")
#         ax.set_ylabel("Y Position")
#         ax.grid(True)
    
#     # Run trials
#     for trial in range(num_trials):
#         # Sample random initial state within the bottom-left corner
#         px0 = np.random.uniform(-1.3, -1.2)
#         py0 = np.random.uniform(-1.3, -1.2)
#         phi0 = np.random.uniform(-np.pi, np.pi)
#         initial_state = (px0, py0, phi0)
        
#         # Create environment
#         env = RobotEnv(obstacles, goal, initial_state)
        
#         # Reset environment
#         state = env.reset()
#         done = False
#         step = 0
        
#         # For rendering
#         trajectory = [state]
        
#         # Run trial
#         while not done and step < max_steps:
#             # Select action (no exploration during testing)
#             action = agent.select_action(state, goal, epsilon=0)
            
#             # Take action
#             next_state, reward, done = env.step(state, action)
            
#             # Update state
#             state = next_state
            
#             # Record trajectory
#             trajectory.append(state)
            
#             # Update step counter
#             step += 1
            
#             # Check if goal reached
#             if env.check_goal(state):
#                 success_count += 1
#                 total_steps += step
#                 successful_trials += 1
#                 break
        
#         # Plot trajectory if rendering
#         if render:
#             xs = [s[0] for s in trajectory]
#             ys = [s[1] for s in trajectory]
            
#             # Different color and transparency for each trajectory
#             color = plt.cm.jet(trial / num_trials)
#             ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1.5)
            
#             # Plot final position
#             if env.check_goal(trajectory[-1]):
#                 ax.plot(xs[-1], ys[-1], 'go', markersize=5)
#             else:
#                 ax.plot(xs[-1], ys[-1], 'ro', markersize=5)
    
#     # Calculate success rate and average steps
#     success_rate = success_count / num_trials
#     avg_steps = total_steps / successful_trials if successful_trials > 0 else float('inf')
    
#     # Save visualization if rendering
#     if render and output_dir:
#         title = f"{env_name} - Success Rate: {success_rate:.2f}, Avg Steps: {avg_steps:.1f}"
#         ax.set_title(title)
#         plt.savefig(os.path.join(output_dir, f"{env_name}_test_results.png"))
#         plt.close(fig)
    
#     return success_rate, avg_steps, success_count, total_steps, successful_trials

# def test_multiple_environments(agent, test_envs, num_trials=10, max_steps=150,
#                              render=True, output_dir=None):
#     """
#     Test agent on multiple environments with different goals.
    
#     Args:
#         agent (DQNAgent): The trained DQN agent.
#         test_envs (list): List of test environment dictionaries.
#         num_trials (int): Number of trials per environment.
#         max_steps (int): Maximum steps per trial.
#         render (bool): Whether to render the environment.
#         output_dir (str): Directory to save results.
        
#     Returns:
#         dict: Dictionary of test results.
#     """
#     # Initialize result tracking
#     all_results = {}
#     overall_success_count = 0
#     overall_total_trials = 0
#     overall_total_steps = 0
#     overall_successful_trials = 0
    
#     # Create output directory if specified
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
    
#     # Test each environment
#     for env_idx, env_config in enumerate(test_envs):
#         obstacles = env_config['obstacles']
        
#         # For each environment, test multiple goals
#         for goal_idx, goal in enumerate(env_config['goals']):
#             env_name = f"env{env_idx+1}_goal{goal_idx+1}"
#             print(f"\nTesting {env_name}")
            
#             # Test on this environment with this goal
#             success_rate, avg_steps, success_count, total_steps, successful_trials = test_single_environment(
#                 agent=agent,
#                 obstacles=obstacles,
#                 goal=goal,
#                 num_trials=num_trials,
#                 max_steps=max_steps,
#                 render=render,
#                 output_dir=output_dir,
#                 env_name=env_name
#             )
            
#             # Store results
#             all_results[env_name] = {
#                 'success_rate': success_rate,
#                 'avg_steps': avg_steps,
#                 'success_count': success_count,
#                 'total_trials': num_trials,
#                 'successful_trials': successful_trials
#             }
            
#             # Update overall statistics
#             overall_success_count += success_count
#             overall_total_trials += num_trials
#             overall_total_steps += total_steps
#             overall_successful_trials += successful_trials
            
#             print(f"  Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
#             print(f"  Success Rate: {success_rate:.2f}")
#             print(f"  Average Steps to Goal: {avg_steps:.1f}")
    
#     # Calculate overall results
#     overall_success_rate = overall_success_count / overall_total_trials
#     overall_avg_steps = overall_total_steps / overall_successful_trials if overall_successful_trials > 0 else float('inf')
    
#     # Store overall results
#     all_results['overall'] = {
#         'success_rate': overall_success_rate,
#         'avg_steps': overall_avg_steps,
#         'success_count': overall_success_count,
#         'total_trials': overall_total_trials,
#         'successful_trials': overall_successful_trials
#     }
    
#     # Save results to file if output directory specified
#     if output_dir:
#         with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
#             json.dump(all_results, f, indent=4)
        
#         # Create a summary plot
#         plt.figure(figsize=(12, 6))
        
#         # Remove overall from plotting
#         plot_results = {k: v for k, v in all_results.items() if k != 'overall'}
        
#         # Sort by environment and goal
#         env_names = sorted(plot_results.keys())
#         success_rates = [plot_results[env]['success_rate'] for env in env_names]
#         avg_steps_list = [plot_results[env]['avg_steps'] if plot_results[env]['avg_steps'] != float('inf') else 0 for env in env_names]
        
#         # Create bar plot
#         x = np.arange(len(env_names))
#         width = 0.35
        
#         fig, ax1 = plt.subplots(figsize=(14, 8))
#         ax2 = ax1.twinx()
        
#         rects1 = ax1.bar(x - width/2, success_rates, width, label='Success Rate', color='blue', alpha=0.7)
#         rects2 = ax2.bar(x + width/2, avg_steps_list, width, label='Avg Steps', color='green', alpha=0.7)
        
#         # Add labels and legend
#         ax1.set_xlabel('Environment-Goal')
#         ax1.set_ylabel('Success Rate')
#         ax2.set_ylabel('Average Steps to Goal')
#         ax1.set_title('Test Results Across Environments and Goals')
#         ax1.set_xticks(x)
#         ax1.set_xticklabels(env_names, rotation=45, ha='right')
#         ax1.set_ylim([0, 1.1])
#         ax2.set_ylim([0, max(max(avg_steps_list) * 1.1, 150)])
        
#         # Add a line for overall success rate
#         ax1.axhline(y=all_results['overall']['success_rate'], color='red', linestyle='-', 
#                    label=f"Overall Success Rate: {all_results['overall']['success_rate']:.2f}")
        
#         # Add legend
#         lines1, labels1 = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'test_results_summary.png'))
#         plt.close()
    
#     # Print overall results
#     print("\nOverall Test Results:")
#     print(f"  Success Rate: {overall_success_rate:.2f}")
#     print(f"  Average Steps to Goal: {overall_avg_steps:.1f}")
    
#     return all_results

# def generate_test_environments(num_envs=3, goals_per_env=3):
#     """
#     Generate random test environments with multiple goals per environment.
    
#     Args:
#         num_envs (int): Number of environments to generate.
#         goals_per_env (int): Number of goals per environment.
        
#     Returns:
#         list: List of test environment dictionaries.
#     """
#     test_envs = []
    
#     for _ in range(num_envs):
#         # Generate random obstacles
#         obstacles = RobotEnv.generate_random_environment(num_obstacles=3)
        
#         # Generate random goals from specified regions
#         goals = []
#         for _ in range(goals_per_env):
#             goals.append(RobotEnv.sample_random_goal())
        
#         # Add environment configuration
#         test_envs.append({
#             'obstacles': obstacles,
#             'goals': goals
#         })
    
#     return test_envs

# def create_released_test_environment():
#     """
#     Create the test environment released by the instructors.
#     This should be updated when the instructors release the test environment.
    
#     Returns:
#         dict: Test environment configuration.
#     """
#     # This is a placeholder. Replace with actual released test environment.
#     obstacles = [
#         {'x': 0.3, 'y': 0.3, 'r': 0.18},
#         {'x': -0.3, 'y': 0.5, 'r': 0.19},
#         {'x': 0.5, 'y': -0.7, 'r': 0.16}
#     ]
    
#     goals = [
#         (0.9, 0.85),     # Region 1
#         (-1.0, 0.9)      # Region 2
#     ]
    
#     return {
#         'obstacles': obstacles,
#         'goals': goals
#     }

# def visualize_environments(envs, output_dir):
#     """
#     Visualize all test environments.
    
#     Args:
#         envs (list): List of environment configurations.
#         output_dir (str): Directory to save visualizations.
#     """
#     for env_idx, env_config in enumerate(envs):
#         obstacles = env_config['obstacles']
#         goals = env_config['goals']
        
#         # Create figure
#         fig, ax = plt.subplots(figsize=(10, 10))
        
#         # Plot obstacles
#         for obs in obstacles:
#             circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
#             ax.add_patch(circle)
        
#         # Plot goals
#         for goal_idx, goal in enumerate(goals):
#             goal_circle = plt.Circle(goal, 0.08, color='green', alpha=0.3)
#             ax.add_patch(goal_circle)
#             ax.plot(goal[0], goal[1], 'g*', markersize=15, label=f"Goal {goal_idx+1}")
        
#         # Plot initial region
#         init_rect = plt.Rectangle((-1.3, -1.3), 0.1, 0.1, color='blue', alpha=0.3, label='Initial Region')
#         ax.add_patch(init_rect)
        
#         # Set plot limits
#         ax.set_xlim([-1.5, 1.5])
#         ax.set_ylim([-1.5, 1.5])
#         ax.set_xlabel("X Position")
#         ax.set_ylabel("Y Position")
#         ax.set_title(f"Test Environment {env_idx+1}")
#         ax.grid(True)
#         ax.legend()
        
#         # Save figure
#         plt.savefig(os.path.join(output_dir, f"test_env{env_idx+1}.png"))
#         plt.close(fig)

# def main():
#     """Main function to parse arguments and start testing."""
#     parser = argparse.ArgumentParser(description='Test DQN agent for robot navigation.')
#     parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
#     parser.add_argument('--num-envs', type=int, default=3, help='Number of test environments')
#     parser.add_argument('--goals-per-env', type=int, default=3, help='Number of goals per environment')
#     parser.add_argument('--num-trials', type=int, default=10, help='Number of trials per environment')
#     parser.add_argument('--max-steps', type=int, default=150, help='Maximum steps per trial')
#     parser.add_argument('--render', action='store_true', help='Render test environments')
#     parser.add_argument('--output-dir', type=str, default='test_results', help='Directory to save results')
#     parser.add_argument('--released-env', action='store_true', help='Test on released environment')
#     parser.add_argument('--seed', type=int, default=123, help='Random seed')
#     parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
#     args = parser.parse_args()
    
#     # Set random seed
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
    
#     # Set device
#     device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
#     print(f"Using device: {device}")
    
#     # Create output directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save test parameters
#     with open(os.path.join(output_dir, 'test_params.json'), 'w') as f:
#         json.dump(vars(args), f, indent=4)
    
#     # Create agent
#     agent = DQNAgent(device=device)
    
#     # Load trained model
#     agent.load(args.model_path)
#     print(f"Loaded model from {args.model_path}")
    
#     # Generate test environments
#     if args.released_env:
#         # Test on released environment
#         released_env = create_released_test_environment()
#         test_envs = [released_env]
#         print("Testing on released environment")
#     else:
#         # Generate random test environments
#         test_envs = generate_test_environments(args.num_envs, args.goals_per_env)
#         print(f"Generated {args.num_envs} test environments with {args.goals_per_env} goals each")
    
#     # Visualize test environments
#     visualize_environments(test_envs, output_dir)
    
#     # Start testing
#     print("\nStarting testing...")
#     start_time = time.time()
    
#     # Test on all environments
#     results = test_multiple_environments(
#         agent=agent,
#         test_envs=test_envs,
#         num_trials=args.num_trials,
#         max_steps=args.max_steps,
#         render=args.render,
#         output_dir=output_dir
#     )
    
#     # Print total testing time
#     test_time = time.time() - start_time
#     print(f"\nTesting completed in {test_time:.2f} seconds")
    
#     return results

# if __name__ == "__main__":
#     main()
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime
import time

from environment import RobotEnv
from dqn_agent import DQNAgent

def test_problem1(agent, num_trials=10, max_steps=150, render=True, output_dir=None):
    """
    Test agent according to Problem 1 requirements.
    
    Args:
        agent (DQNAgent): The trained DQN agent.
        num_trials (int): Number of trials to run (10 as specified in problem).
        max_steps (int): Maximum steps per trial (150 as specified in problem).
        render (bool): Whether to render the environment.
        output_dir (str): Directory to save results.
        
    Returns:
        tuple: (success_rate, avg_steps_to_goal)
    """
    # Initialize result tracking
    success_count = 0
    total_steps = 0
    successful_trials = 0
    
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize figure for visualization if rendering
    if render:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create environment to get obstacle information
        env = RobotEnv()
        
        # Plot obstacles
        for obs in env.obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
            ax.add_patch(circle)
        
        # Plot goal
        goal_pos = env.goal_pos
        goal_circle = plt.Circle(goal_pos, 0.08, color='green', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15)
        
        # Set plot limits
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
    
    # Run trials
    for trial in range(num_trials):
        # Sample random initial state from the specified square area
        px0 = np.random.uniform(-1.3, -1.2)
        py0 = np.random.uniform(-1.3, -1.2)
        phi0 = np.random.uniform(-np.pi, np.pi)
        initial_state = (px0, py0, phi0)
        
        # Create environment
        env = RobotEnv(initial_state)
        
        # Reset environment
        state = env.reset()
        done = False
        step = 0
        
        # For rendering
        trajectory = [state]
        
        # Run trial
        while not done and step < max_steps:
            # Select action (no exploration during testing)
            action = agent.select_action(state, epsilon=0)
            
            # Take action
            next_state, reward, done = env.step(state, action)
            
            # Update state
            state = next_state
            
            # Record trajectory
            trajectory.append(state)
            
            # Update step counter
            step += 1
            
            # Check if goal reached
            if env.check_goal(state):
                success_count += 1
                total_steps += step
                successful_trials += 1
                break
        
        # Plot trajectory if rendering
        if render:
            xs = [s[0] for s in trajectory]
            ys = [s[1] for s in trajectory]
            
            # Different color and transparency for each trajectory
            color = plt.cm.jet(trial / num_trials)
            ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1.5)
            
            # Plot final position
            if env.check_goal(trajectory[-1]):
                ax.plot(xs[-1], ys[-1], 'go', markersize=5)
            else:
                ax.plot(xs[-1], ys[-1], 'ro', markersize=5)
    
    # Calculate success rate and average steps
    success_rate = success_count / num_trials
    avg_steps = total_steps / successful_trials if successful_trials > 0 else float('inf')
    
    # Save visualization if rendering
    if render and output_dir:
        title = f"Problem 1 - Success Rate: {success_rate:.2f}, Avg Steps: {avg_steps:.1f}"
        ax.set_title(title)
        plt.savefig(os.path.join(output_dir, "problem1_test_results.png"))
        plt.close(fig)
    
    # Print results
    print(f"Problem 1 Test Results:")
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Average Steps to Goal: {avg_steps:.1f}")
    
    return success_rate, avg_steps

def main():
    """Main function to parse arguments and start testing."""
    parser = argparse.ArgumentParser(description='Test DQN agent for Problem 1.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num-trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--max-steps', type=int, default=150, help='Maximum steps per trial')
    parser.add_argument('--render', action='store_true', help='Render test environments')
    parser.add_argument('--output-dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test parameters
    with open(os.path.join(output_dir, 'test_params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create agent
    agent = DQNAgent(device=device)
    
    # Load trained model
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Start testing
    print("\nStarting testing...")
    start_time = time.time()
    
    # Test according to Problem 1 requirements
    success_rate, avg_steps = test_problem1(
        agent=agent,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        render=args.render,
        output_dir=output_dir
    )
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'success_rate': success_rate,
            'avg_steps': float(avg_steps) if avg_steps != float('inf') else "inf"
        }, f, indent=4)
    
    # Print total testing time
    test_time = time.time() - start_time
    print(f"\nTesting completed in {test_time:.2f} seconds")
    
    return success_rate, avg_steps

if __name__ == "__main__":
    main()