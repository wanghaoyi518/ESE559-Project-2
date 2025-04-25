import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import json
from datetime import datetime
from collections import deque

from environment import *
from dqn_agent import DQNAgent

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_rewards(rewards, window_size=100, filename='training_rewards.png'):
    """
    Plot training rewards with a moving average.
    
    Args:
        rewards (list): List of rewards for each episode.
        window_size (int): Size of the window for moving average.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.4, label='Raw Rewards')
    
    # Plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f'Moving Average ({window_size} episodes)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train(agent, num_episodes=2000, batch_size=64, 
          save_freq=100, log_freq=10, render=False, results_dir='results',
          update_target_freq=10, max_steps=150):
    """
    Train the DQN agent for Problem 1 (fixed environment and goal).
    
    Args:
        agent (DQNAgent): The DQN agent to train.
        num_episodes (int): Number of episodes to train for.
        batch_size (int): Batch size for experience replay.
        save_freq (int): Frequency of saving the model.
        log_freq (int): Frequency of logging training progress.
        render (bool): Whether to render the environment during training.
        results_dir (str): Directory to save results.
        update_target_freq (int): Number of episodes after which to update target network.
        max_steps (int): Maximum steps per episode.
        
    Returns:
        list: List of rewards for each episode.
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save training parameters to JSON file for reproducibility
    training_params = {
        'problem_type': 'Problem 1',
        'num_episodes': num_episodes,
        'batch_size': batch_size,
        'save_freq': save_freq,
        'log_freq': log_freq,
        'render': render,
        'update_target_freq': update_target_freq,
        'max_steps': max_steps,
        'agent_config': {
            'state_dim': agent.state_dim,
            'action_dim': agent.action_dim,
            'hidden_dim': agent.hidden_dim,
            'learning_rate': agent.learning_rate,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'epsilon_min': agent.epsilon_min,
            'epsilon_decay': agent.epsilon_decay,
            'memory_size': agent.memory_size,
            'device': str(agent.device)
        },
        'timestamp': timestamp,
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(run_dir, 'training_parameters.json'), 'w') as f:
        json.dump(training_params, f, indent=4)
    
    # Initialize list to store rewards
    rewards = []
    
    # Initialize list to store success rates
    success_rates = []
    
    # Create a fixed environment for Problem 1
    env = RobotEnv(initial_state=(-1.2, -1.2, 0))
    
    # Create a progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    
    # For visualizing exploration
    epsilons = []
    
    for episode in pbar:
        # Occasionally start from a random position within the bottom-left corner
        # This helps with exploration and generalization
        if episode % 5 == 0:
            px0 = np.random.uniform(-1.3, -1.2)
            py0 = np.random.uniform(-1.3, -1.2)
            phi0 = np.random.uniform(-np.pi, np.pi)
            initial_state = (px0, py0, phi0)
            env = RobotEnv(initial_state=initial_state)
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # For rendering
        if render and episode % log_freq == 0:
            trajectory = [state]
        
        # Run episode
        while not done and step < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done = env.step(state, action)
            
            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            step += 1
            
            # Record trajectory for rendering
            if render and episode % log_freq == 0:
                trajectory.append(state)
            
            # Train the agent if enough samples in memory
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Update target network periodically
        if episode % update_target_freq == 0:
            agent.update_target_network()
            
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Record episode reward
        rewards.append(total_reward)
        
        # Record epsilon
        epsilons.append(agent.epsilon)
        
        # Track success (reaching goal)
        success = env.check_goal(state)
        success_rates.append(1 if success else 0)
        
        # Update progress bar
        recent_success_rate = np.mean(success_rates[-min(100, len(success_rates)):]) * 100
        pbar.set_postfix({
            'reward': f"{total_reward:.2f}", 
            'epsilon': f"{agent.epsilon:.4f}",
            'step': step,
            'success': f"{recent_success_rate:.1f}%"
        })
        
        # Save model periodically
        if episode > 0 and episode % save_freq == 0:
            agent.save(os.path.join(run_dir, f"dqn_model_ep{episode}.pth"))
            
            # Plot rewards
            plot_rewards(rewards, filename=os.path.join(run_dir, 'training_rewards.png'))
            
            # Plot epsilon decay
            plt.figure(figsize=(10, 5))
            plt.plot(epsilons)
            plt.title('Epsilon Decay')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.grid(True)
            plt.savefig(os.path.join(run_dir, 'epsilon_decay.png'))
            plt.close()
            
            # Save rewards to file
            with open(os.path.join(run_dir, 'rewards.json'), 'w') as f:
                json.dump(rewards, f)
        
        # Render environment and trajectory periodically
        if render and episode % log_freq == 0:
            fig = env.visualize(trajectory=trajectory, 
                             title=f"Episode {episode}, Reward: {total_reward:.2f}, Success: {success}")
            plt.savefig(os.path.join(run_dir, f"episode_{episode}.png"))
            plt.close(fig)
        
        # Early stopping if consistently successful
        if len(success_rates) >= 100 and np.mean(success_rates[-100:]) > 0.95:
            print(f"\nEarly stopping at episode {episode}: Success rate above 95% for 100 episodes")
            break
    
    # Save final model
    agent.save(os.path.join(run_dir, "dqn_model_final.pth"))
    
    # Plot final rewards
    plot_rewards(rewards, filename=os.path.join(run_dir, 'training_rewards_final.png'))
    
    # Save final rewards to file
    with open(os.path.join(run_dir, 'rewards_final.json'), 'w') as f:
        json.dump(rewards, f)
    
    # Calculate and save final success rate
    final_success_rate = np.mean(success_rates[-min(100, len(success_rates))]) * 100
    with open(os.path.join(run_dir, 'final_stats.json'), 'w') as f:
        json.dump({
            'final_success_rate': final_success_rate,
            'final_epsilon': agent.epsilon,
            'num_episodes': episode + 1,  # Include the current episode
            'actual_episodes_trained': episode + 1
        }, f)
    
    print(f"\nTraining completed. Final success rate: {final_success_rate:.2f}%")
    
    return rewards, run_dir  # Return the run directory for reference

def train_problem2(agent, num_episodes=2000, batch_size=64, 
                  save_freq=100, log_freq=10, render=False, results_dir='results',
                  update_target_freq=10, max_steps=150, additional_envs=20):
    """
    Train the DQN agent for Problem 2 with multiple environments including random ones.
    
    Args:
        agent (EnhancedDQNAgent): The enhanced DQN agent to train.
        num_episodes (int): Number of episodes to train for.
        batch_size (int): Batch size for experience replay.
        save_freq (int): Frequency of saving the model.
        log_freq (int): Frequency of logging training progress.
        render (bool): Whether to render the environment during training.
        results_dir (str): Directory to save results.
        update_target_freq (int): Number of episodes after which to update target network.
        max_steps (int): Maximum steps per episode.
        additional_envs (int): Number of additional random environments to generate.
        
    Returns:
        tuple: (rewards, run_dir) - List of rewards and the directory where results are saved.
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save training parameters to JSON file for reproducibility
    training_params = {
        'problem_type': 'Problem 2',
        'num_episodes': num_episodes,
        'batch_size': batch_size,
        'save_freq': save_freq,
        'log_freq': log_freq,
        'render': render,
        'update_target_freq': update_target_freq,
        'max_steps': max_steps,
        'additional_envs': additional_envs,
        'total_environments': 3 + additional_envs,  # 3 predefined + additional random envs
        'agent_config': {
            'state_dim': agent.state_dim,
            'action_dim': agent.action_dim,
            'hidden_dim': agent.hidden_dim,
            'learning_rate': agent.learning_rate,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'epsilon_min': agent.epsilon_min,
            'epsilon_decay': agent.epsilon_decay,
            'memory_size': agent.memory_size,
            'device': str(agent.device)
        },
        'timestamp': timestamp,
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(run_dir, 'training_parameters.json'), 'w') as f:
        json.dump(training_params, f, indent=4)
    
    # Initialize tracking
    rewards = []
    success_rates = []
    epsilons = []
    env_success_tracking = {}  # Track success by environment
    
    # Get training environments with additional random ones
    training_envs = Problem2Env.get_training_environments(additional_envs=additional_envs)
    
    # Create a directory for environment visualizations
    env_viz_dir = os.path.join(run_dir, "environments")
    os.makedirs(env_viz_dir, exist_ok=True)
    
    # Visualize each training environment
    for i, obstacles in enumerate(training_envs):
        env = Problem2Env(obstacles=obstacles)
        if i < 3:
            title = f"Predefined Environment {i+1}"
        else:
            title = f"Random Environment {i-2}"
            
        fig = env.visualize(title=title)
        plt.savefig(os.path.join(env_viz_dir, f"env_{i+1}.png"))
        plt.close(fig)
        
        # Initialize environment success tracking
        env_success_tracking[i] = {
            'attempts': 0,
            'successes': 0,
            'last_100_attempts': deque(maxlen=100),
            'success_rate': 0.0
        }
    
    # Create a grid visualization of all environments
    env = Problem2Env()
    fig = env.visualize_environments(training_envs[:min(len(training_envs), 20)], 
                                   "Sample of Training Environments")
    plt.savefig(os.path.join(run_dir, "training_environments_sample.png"))
    plt.close(fig)
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        # Randomly select a training environment
        env_idx = np.random.randint(0, len(training_envs))
        obstacles = training_envs[env_idx]
        
        # Randomly sample initial orientation
        phi0 = np.random.uniform(-np.pi, np.pi)
        initial_state = (-1.2, -1.2, phi0)
        
        # Create environment
        env = Problem2Env(obstacles=obstacles, initial_state=initial_state)
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # For rendering
        if render and episode % log_freq == 0:
            trajectory = [state]
        
        # Run episode
        while not done and step < max_steps:
            # Select action
            action = agent.select_action(state, obstacles)
            
            # Take action
            next_state, reward, done = env.step(state, action)
            
            # Store experience in replay memory
            agent.remember(state, obstacles, action, reward, next_state, obstacles, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            step += 1
            
            # Record trajectory for rendering
            if render and episode % log_freq == 0:
                trajectory.append(state)
            
            # Train the agent if enough samples in memory
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Update target network periodically
        if episode % update_target_freq == 0:
            agent.update_target_network()
            
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Record data
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        success = env.check_goal(state)
        success_rates.append(1 if success else 0)
        
        # Update environment-specific success tracking
        env_success_tracking[env_idx]['attempts'] += 1
        env_success_tracking[env_idx]['successes'] += 1 if success else 0
        env_success_tracking[env_idx]['last_100_attempts'].append(1 if success else 0)
        env_success_tracking[env_idx]['success_rate'] = (
            sum(env_success_tracking[env_idx]['last_100_attempts']) / 
            len(env_success_tracking[env_idx]['last_100_attempts'])
        )
        
        # Update progress bar
        recent_success_rate = np.mean(success_rates[-min(100, len(success_rates)):]) * 100
        pbar.set_postfix({
            'reward': f"{total_reward:.2f}", 
            'epsilon': f"{agent.epsilon:.4f}",
            'step': step,
            'success': f"{recent_success_rate:.1f}%",
            'env': f"{env_idx+1}/{len(training_envs)}"
        })
        
        # Save model and plots periodically
        if episode > 0 and episode % save_freq == 0:
            agent.save(os.path.join(run_dir, f"dqn_model_ep{episode}.pth"))
            
            # Plot rewards
            plot_rewards(rewards, filename=os.path.join(run_dir, 'training_rewards.png'))
            
            # Save rewards to file
            with open(os.path.join(run_dir, 'rewards.json'), 'w') as f:
                json.dump(rewards, f)
                
            # Plot environment-specific success rates
            plt.figure(figsize=(12, 6))
            env_indices = sorted(list(env_success_tracking.keys()))
            success_rates_by_env = [env_success_tracking[i]['success_rate'] for i in env_indices]
            
            plt.bar(range(len(env_indices)), success_rates_by_env)
            plt.xlabel('Environment Index')
            plt.ylabel('Success Rate (last 100 attempts)')
            plt.title(f'Success Rate by Environment (Episode {episode})')
            plt.xticks(range(len(env_indices)), [str(i+1) for i in env_indices])
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'env_success_rates_ep{episode}.png'))
            plt.close()
        
        # Render environment and trajectory periodically
        if render and episode % log_freq == 0:
            fig = env.visualize(trajectory=trajectory, 
                             title=f"Episode {episode}, Env {env_idx+1}, Reward: {total_reward:.2f}")
            plt.savefig(os.path.join(run_dir, f"episode_{episode}_env{env_idx+1}.png"))
            plt.close(fig)
        
        # Early stopping if consistently successful across all environments
        if episode >= 1000 and episode % 100 == 0:
            # Check if the agent has a high success rate across all environments
            env_success_rates = [data['success_rate'] for data in env_success_tracking.values() 
                               if len(data['last_100_attempts']) >= 20]  # Only check environments with sufficient attempts
            
            if env_success_rates and np.mean(env_success_rates) > 0.9 and min(env_success_rates) > 0.7:
                print(f"\nEarly stopping at episode {episode}: High success rates across all environments")
                print(f"Average: {np.mean(env_success_rates):.2f}, Min: {min(env_success_rates):.2f}")
                break
    
    # Save final model
    agent.save(os.path.join(run_dir, "dqn_model_final.pth"))
    
    # Save final environment success rates
    env_success_data = {}
    for env_idx, data in env_success_tracking.items():
        env_success_data[f"env_{env_idx+1}"] = {
            'attempts': data['attempts'],
            'successes': data['successes'],
            'success_rate': data['success_rate'] if len(data['last_100_attempts']) > 0 else 0.0
        }
    
    with open(os.path.join(run_dir, 'environment_success_rates.json'), 'w') as f:
        json.dump(env_success_data, f, indent=4)
    
    # Plot final environment success rates
    plt.figure(figsize=(12, 6))
    env_indices = sorted(list(env_success_tracking.keys()))
    success_rates_by_env = [env_success_tracking[i]['success_rate'] 
                           if len(env_success_tracking[i]['last_100_attempts']) > 0 else 0.0 
                           for i in env_indices]
    
    plt.bar(range(len(env_indices)), success_rates_by_env)
    plt.xlabel('Environment Index')
    plt.ylabel('Success Rate (last 100 attempts)')
    plt.title(f'Final Success Rate by Environment')
    plt.xticks(range(len(env_indices)), [str(i+1) for i in env_indices], rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'final_env_success_rates.png'))
    plt.close()
    
    # Calculate and save final success rate
    final_success_rate = np.mean(success_rates[-min(300, len(success_rates))]) * 100
    with open(os.path.join(run_dir, 'final_stats.json'), 'w') as f:
        json.dump({
            'final_success_rate': final_success_rate,
            'final_epsilon': agent.epsilon,
            'num_episodes': episode + 1,
            'actual_episodes_trained': episode + 1,
            'num_environments': len(training_envs)
        }, f)
    
    print(f"\nTraining completed. Final success rate: {final_success_rate:.2f}%")
    
    return rewards, run_dir
