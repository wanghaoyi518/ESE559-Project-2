import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime
import time

from environment import *
from dqn_agent import DQNAgent

# Import the test cases from test_case.py
try:
    from test_case import problem2_test, problem3_test1, problem3_test_2
    HAS_TEST_CASES = True
except ImportError:
    HAS_TEST_CASES = False
    print("Warning: test_case.py not found. Will not use released test cases.")

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
        env = RobotEnv(initial_state=initial_state)
        
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

def test_problem2(agent, num_trials=10, max_steps=150, render=True, output_dir=None):
    """
    Test agent according to Problem 2 requirements.
    
    Args:
        agent (EnhancedDQNAgent): The trained DQN agent.
        num_trials (int): Number of trials to run per test environment.
        max_steps (int): Maximum steps per trial.
        render (bool): Whether to render the environment.
        output_dir (str): Directory to save results.
        
    Returns:
        dict: Dictionary containing success rates and average steps for each test environment.
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate test environments
    test_envs = []
    
    # Add 3 random test environments
    for i in range(3):
        obstacles = Problem2Env.generate_random_test_environment()
        test_envs.append(obstacles)
        
        # Visualize each test environment
        if render:
            env = Problem2Env(obstacles=obstacles)
            fig = env.visualize(title=f"Random Test Environment {i+1}")
            plt.savefig(os.path.join(output_dir, f"random_test_env{i+1}.png"))
            plt.close(fig)
    
    # Add the released test environment if available
    if HAS_TEST_CASES:
        # Convert the dictionary format to list format
        released_obstacles = [
            {'x': problem2_test['O1']['x'], 'y': problem2_test['O1']['y'], 'r': problem2_test['O1']['r']},
            {'x': problem2_test['O2']['x'], 'y': problem2_test['O2']['y'], 'r': problem2_test['O2']['r']},
            {'x': problem2_test['O3']['x'], 'y': problem2_test['O3']['y'], 'r': problem2_test['O3']['r']}
        ]
        test_envs.append(released_obstacles)
        
        # Visualize released test environment
        if render:
            env = Problem2Env(obstacles=released_obstacles)
            fig = env.visualize(title="Released Test Environment")
            plt.savefig(os.path.join(output_dir, "released_test_env.png"))
            plt.close(fig)
    
    # Initialize results tracking
    results = {}
    
    # Test on each environment
    for env_idx, obstacles in enumerate(test_envs):
        env_name = f"released_env" if (HAS_TEST_CASES and env_idx == len(test_envs) - 1) else f"random_env{env_idx+1}"
        print(f"\nTesting on {env_name}...")
        
        # Initialize tracking for this environment
        success_count = 0
        total_steps = 0
        successful_trials = 0
        
        # Initialize figure for visualization if rendering
        if render:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot obstacles
            for obs in obstacles:
                circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
                ax.add_patch(circle)
            
            # Plot goal
            goal_pos = (1.2, 1.2)  # Fixed goal for Problem 2
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
            env = Problem2Env(obstacles=obstacles, initial_state=initial_state)
            
            # Reset environment
            state = env.reset()
            done = False
            step = 0
            
            # For rendering
            trajectory = [state]
            
            # Run trial
            while not done and step < max_steps:
                # Select action (no exploration during testing)
                action = agent.select_action(state, obstacles, epsilon=0)
                
                # Take action
                next_state, reward, done = env.step(state, action)
                
                # Update state
                state = next_state
                
                # Record trajectory for rendering
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
            title = f"{env_name} - Success Rate: {success_rate:.2f}, Avg Steps: {avg_steps:.1f}"
            ax.set_title(title)
            plt.savefig(os.path.join(output_dir, f"{env_name}_results.png"))
            plt.close(fig)
        
        # Store results
        results[env_name] = {
            'success_rate': success_rate,
            'avg_steps': float(avg_steps) if avg_steps != float('inf') else "inf",
            'success_count': success_count,
            'total_trials': num_trials,
            'successful_trials': successful_trials
        }
        
        # Print results
        print(f"  Success Rate: {success_rate:.2f}")
        avg_steps_str = f"{avg_steps:.1f}" if avg_steps != float('inf') else "inf"
        print(f"  Average Steps to Goal: {avg_steps_str}")
    
    # Calculate overall statistics
    overall_success_count = sum(results[env]['success_count'] for env in results.keys())
    overall_total_trials = sum(results[env]['total_trials'] for env in results.keys())
    overall_successful_trials = sum(results[env]['successful_trials'] for env in results.keys())
    
    overall_success_rate = overall_success_count / overall_total_trials
    
    # Calculate overall average steps (handling "inf" cases)
    total_steps_sum = 0
    for env in results.keys():
        if results[env]["avg_steps"] != "inf":
            total_steps_sum += results[env]["successful_trials"] * results[env]["avg_steps"]
    
    overall_avg_steps = total_steps_sum / overall_successful_trials if overall_successful_trials > 0 else float('inf')
    
    results['overall'] = {
        'success_rate': overall_success_rate,
        'avg_steps': float(overall_avg_steps) if overall_avg_steps != float('inf') else "inf",
        'success_count': overall_success_count,
        'total_trials': overall_total_trials,
        'successful_trials': overall_successful_trials
    }
    
    # Save results to file
    if output_dir:
        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print overall results
    print("\nOverall Results:")
    print(f"  Success Rate: {overall_success_rate:.2f}")
    overall_avg_steps_str = f"{overall_avg_steps:.1f}" if overall_avg_steps != float('inf') else "inf"
    print(f"  Average Steps to Goal: {overall_avg_steps_str}")
    
    return results

def test_problem3(agent, num_trials=10, max_steps=150, render=True, output_dir=None):
    """
    Test agent according to Problem 3 requirements.
    
    Args:
        agent (GoalConditionedDQNAgent): The trained goal-conditioned DQN agent.
        num_trials (int): Number of trials to run per test environment-goal pair.
        max_steps (int): Maximum steps per trial.
        render (bool): Whether to render the environment.
        output_dir (str): Directory to save results.
        
    Returns:
        dict: Dictionary containing success rates and average steps for each test environment-goal pair.
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate test environments
    test_envs = []
    # Generate 3 random test environments
    for i in range(3):
        obstacles = Problem2Env.generate_random_test_environment()
        test_envs.append(obstacles)
        
        # Visualize each test environment
        if render:
            env = Problem3Env(obstacles=obstacles)
            fig = env.visualize(title=f"Random Test Environment {i+1}")
            plt.savefig(os.path.join(output_dir, f"random_test_env{i+1}.png"))
            plt.close(fig)
    
    # Add the released test environments if available
    released_test_envs = []
    if HAS_TEST_CASES:
        # Convert test case 1
        released_obstacles1 = [
            {'x': problem3_test1['O1']['x'], 'y': problem3_test1['O1']['y'], 'r': problem3_test1['O1']['r']},
            {'x': problem3_test1['O2']['x'], 'y': problem3_test1['O2']['y'], 'r': problem3_test1['O2']['r']},
            {'x': problem3_test1['O3']['x'], 'y': problem3_test1['O3']['y'], 'r': problem3_test1['O3']['r']}
        ]
        released_test_envs.append((released_obstacles1, problem3_test1['x_goal'], problem3_test1['y_goal']))
        
        # Convert test case 2
        released_obstacles2 = [
            {'x': problem3_test_2['O1']['x'], 'y': problem3_test_2['O1']['y'], 'r': problem3_test_2['O1']['r']},
            {'x': problem3_test_2['O2']['x'], 'y': problem3_test_2['O2']['y'], 'r': problem3_test_2['O2']['r']},
            {'x': problem3_test_2['O3']['x'], 'y': problem3_test_2['O3']['y'], 'r': problem3_test_2['O3']['r']}
        ]
        released_test_envs.append((released_obstacles2, problem3_test_2['x_goal'], problem3_test_2['y_goal']))
        
        # Visualize released test environments
        for i, (obstacles, x_goal, y_goal) in enumerate(released_test_envs):
            # Sample a specific goal position within the range
            goal_x = np.random.uniform(x_goal[0], x_goal[1])
            goal_y = np.random.uniform(y_goal[0], y_goal[1])
            goal_pos = (goal_x, goal_y)
            
            env = Problem3Env(obstacles=obstacles, goal_position=goal_pos)
            fig = env.visualize(title=f"Released Test Environment {i+1}")
            plt.savefig(os.path.join(output_dir, f"released_test_env{i+1}.png"))
            plt.close(fig)
    
    # Generate test goals for each environment
    test_env_goal_pairs = []
    
    # For random environments
    for env_idx, obstacles in enumerate(test_envs):
        for goal_idx in range(3):  # Generate 3 test goals per environment
            # Use the updated method that checks for obstacle conflicts
            goal_pos = Problem3Env.generate_random_goal(obstacles=obstacles)
            test_env_goal_pairs.append(("random", env_idx, obstacles, goal_pos))
            
            # Visualize each environment-goal pair
            if render:
                env = Problem3Env(obstacles=obstacles, goal_position=goal_pos)
                fig = env.visualize(title=f"Random Test Environment {env_idx+1}, Goal {goal_idx+1}: {goal_pos}")
                plt.savefig(os.path.join(output_dir, f"random_test_env{env_idx+1}_goal{goal_idx+1}.png"))
                plt.close(fig)
    
    # For released test environments
    if HAS_TEST_CASES:
        for rel_idx, (obstacles, x_goal, y_goal) in enumerate(released_test_envs):
            # Generate 3 goals per released environment within the specified range
            for goal_idx in range(3):
                goal_x = np.random.uniform(x_goal[0], x_goal[1])
                goal_y = np.random.uniform(y_goal[0], y_goal[1])
                goal_pos = (goal_x, goal_y)
                
                test_env_goal_pairs.append(("released", rel_idx, obstacles, goal_pos))
                
                # Visualize each released environment-goal pair
                if render:
                    env = Problem3Env(obstacles=obstacles, goal_position=goal_pos)
                    fig = env.visualize(title=f"Released Test Environment {rel_idx+1}, Goal {goal_idx+1}: {goal_pos}")
                    plt.savefig(os.path.join(output_dir, f"released_test_env{rel_idx+1}_goal{goal_idx+1}.png"))
                    plt.close(fig)
    
    # Initialize results tracking
    results = {}
    
    # Test on each environment-goal pair
    for pair_idx, (env_type, env_idx, obstacles, goal_pos) in enumerate(test_env_goal_pairs):
        pair_name = f"{env_type}_env{env_idx+1}_goal{pair_idx % 3 + 1}"
        print(f"\nTesting on {pair_name}, goal position {goal_pos}...")
        
        # Initialize tracking for this environment-goal pair
        success_count = 0
        total_steps = 0
        successful_trials = 0
        
        # Initialize figure for visualization if rendering
        if render:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot obstacles
            for obs in obstacles:
                circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
                ax.add_patch(circle)
            
            # Plot goal
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
            env = Problem3Env(obstacles=obstacles, goal_position=goal_pos, initial_state=initial_state)
            
            # Reset environment
            state = env.reset()
            done = False
            step = 0
            
            # For rendering
            trajectory = [state]
            
            # Run trial
            while not done and step < max_steps:
                # Select action (no exploration during testing)
                action = agent.select_action(state, obstacles, goal_pos, epsilon=0)
                
                # Take action
                next_state, reward, done = env.step(state, action)
                
                # Update state
                state = next_state
                
                # Record trajectory for rendering
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
            title = f"{pair_name}, Goal ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}) - Success: {success_rate:.2f}, Avg Steps: {avg_steps:.1f}"
            ax.set_title(title)
            plt.savefig(os.path.join(output_dir, f"{pair_name}_results.png"))
            plt.close(fig)
        
        # Store results
        results[pair_name] = {
            'env_type': env_type,
            'env_idx': env_idx,
            'goal_position': goal_pos,
            'success_rate': success_rate,
            'avg_steps': float(avg_steps) if avg_steps != float('inf') else "inf",
            'success_count': success_count,
            'total_trials': num_trials,
            'successful_trials': successful_trials
        }
        
        # Print results
        print(f"  Success Rate: {success_rate:.2f}")
        avg_steps_str = f"{avg_steps:.1f}" if avg_steps != float('inf') else "inf"
        print(f"  Average Steps to Goal: {avg_steps_str}")
    
    # Calculate overall statistics
    overall_success_count = sum(results[key]["success_count"] for key in results)
    overall_total_trials = sum(results[key]["total_trials"] for key in results)
    overall_successful_trials = sum(results[key]["successful_trials"] for key in results)
    
    overall_success_rate = overall_success_count / overall_total_trials
    
    # Calculate overall average steps (handling "inf" cases)
    total_steps_sum = 0
    for key in results:
        if results[key]["avg_steps"] != "inf":
            total_steps_sum += results[key]["successful_trials"] * results[key]["avg_steps"]
    
    overall_avg_steps = total_steps_sum / overall_successful_trials if overall_successful_trials > 0 else float('inf')
    
    # Calculate stats for random and released environments separately
    if HAS_TEST_CASES:
        random_results = {k: v for k, v in results.items() if v['env_type'] == 'random'}
        released_results = {k: v for k, v in results.items() if v['env_type'] == 'released'}
        
        # Random environments stats
        random_success_count = sum(v["success_count"] for v in random_results.values())
        random_total_trials = sum(v["total_trials"] for v in random_results.values())
        random_successful_trials = sum(v["successful_trials"] for v in random_results.values())
        random_success_rate = random_success_count / random_total_trials if random_total_trials > 0 else 0
        
        # Released environments stats
        released_success_count = sum(v["success_count"] for v in released_results.values())
        released_total_trials = sum(v["total_trials"] for v in released_results.values())
        released_successful_trials = sum(v["successful_trials"] for v in released_results.values())
        released_success_rate = released_success_count / released_total_trials if released_total_trials > 0 else 0
        
        # Add to results
        results['random_overall'] = {
            'success_rate': random_success_rate,
            'success_count': random_success_count,
            'total_trials': random_total_trials,
            'successful_trials': random_successful_trials
        }
        
        results['released_overall'] = {
            'success_rate': released_success_rate,
            'success_count': released_success_count,
            'total_trials': released_total_trials,
            'successful_trials': released_successful_trials
        }
    
    results['overall'] = {
        'success_rate': overall_success_rate,
        'avg_steps': float(overall_avg_steps) if overall_avg_steps != float('inf') else "inf",
        'success_count': overall_success_count,
        'total_trials': overall_total_trials,
        'successful_trials': overall_successful_trials
    }
    
    # Save results to file
    if output_dir:
        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print overall results
    print("\nOverall Results:")
    print(f"  Success Rate: {overall_success_rate:.2f}")
    overall_avg_steps_str = f"{overall_avg_steps:.1f}" if overall_avg_steps != float('inf') else "inf"
    print(f"  Average Steps to Goal: {overall_avg_steps_str}")
    
    # Print released test case results if available
    if HAS_TEST_CASES:
        print("\nReleased Test Cases Results:")
        print(f"  Success Rate: {released_success_rate:.2f}")
        overall_avg_steps_str = f"{overall_avg_steps:.1f}" if overall_avg_steps != float('inf') else "inf"
        print(f"  Average Steps to Goal: {overall_avg_steps_str}")
        
    return results