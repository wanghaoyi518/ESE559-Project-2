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
                  update_target_freq=10, max_steps=150):
    """
    Train the DQN agent for Problem 2 with multiple environments.
    
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
        
    Returns:
        tuple: (rewards, run_dir) - List of rewards and the directory where results are saved.
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize tracking
    rewards = []
    success_rates = []
    epsilons = []
    
    # Get training environments
    training_envs = Problem2Env.get_training_environments()
    
    # Visualize training environments
    env = Problem2Env()
    fig = env.visualize_environments(training_envs, "Training Environments")
    plt.savefig(os.path.join(run_dir, "training_environments.png"))
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
        
        # Update progress bar
        recent_success_rate = np.mean(success_rates[-min(100, len(success_rates)):]) * 100
        pbar.set_postfix({
            'reward': f"{total_reward:.2f}", 
            'epsilon': f"{agent.epsilon:.4f}",
            'step': step,
            'success': f"{recent_success_rate:.1f}%",
            'env': f"{env_idx+1}"
        })
        
        # Save model and plots periodically
        if episode > 0 and episode % save_freq == 0:
            agent.save(os.path.join(run_dir, f"dqn_model_ep{episode}.pth"))
            
            # Plot rewards
            plot_rewards(rewards, filename=os.path.join(run_dir, 'training_rewards.png'))
            
            # Save rewards to file
            with open(os.path.join(run_dir, 'rewards.json'), 'w') as f:
                json.dump(rewards, f)
        
        # Render environment and trajectory periodically
        if render and episode % log_freq == 0:
            fig = env.visualize(trajectory=trajectory, 
                             title=f"Episode {episode}, Env {env_idx+1}, Reward: {total_reward:.2f}")
            plt.savefig(os.path.join(run_dir, f"episode_{episode}_env{env_idx+1}.png"))
            plt.close(fig)
        
        # Early stopping if consistently successful across all environments
        if len(success_rates) >= 300 and np.mean(success_rates[-300:]) > 0.95:
            print(f"\nEarly stopping at episode {episode}: Success rate above 95% for 300 episodes")
            break
    
    # Save final model
    agent.save(os.path.join(run_dir, "dqn_model_final.pth"))
    
    # Plot final rewards
    plot_rewards(rewards, filename=os.path.join(run_dir, 'training_rewards_final.png'))
    
    # Save final rewards to file
    with open(os.path.join(run_dir, 'rewards_final.json'), 'w') as f:
        json.dump(rewards, f)
    
    # Calculate and save final success rate
    final_success_rate = np.mean(success_rates[-min(300, len(success_rates))]) * 100
    with open(os.path.join(run_dir, 'final_stats.json'), 'w') as f:
        json.dump({
            'final_success_rate': final_success_rate,
            'final_epsilon': agent.epsilon,
            'num_episodes': episode + 1,
            'actual_episodes_trained': episode + 1
        }, f)
    
    print(f"\nTraining completed. Final success rate: {final_success_rate:.2f}%")
    
    return rewards, run_dir

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train DQN agent for robot navigation (Problem 1).')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for experience replay')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.001, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.99, help='Exploration decay rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension of DQN')
    parser.add_argument('--target-update-freq', type=int, default=10, help='Target network update frequency (episodes)')
    parser.add_argument('--memory-size', type=int, default=100000, help='Size of replay memory')
    parser.add_argument('--save-freq', type=int, default=100, help='Model saving frequency')
    parser.add_argument('--log-freq', type=int, default=10, help='Logging frequency')
    parser.add_argument('--render', action='store_true', help='Render environment during training')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--max-steps', type=int, default=150, help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Create agent
    agent = DQNAgent(
        state_dim=3,  # [px, py, phi]
        action_dim=23,  # 23 discrete actions
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        memory_size=args.memory_size,
        device=device
    )
    
    # Print training setup
    print(f"Starting training with {args.episodes} episodes")
    
    # Start training
    start_time = time.time()
    rewards, run_dir = train(
        agent=agent,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        render=args.render,
        results_dir=args.results_dir,
        update_target_freq=args.target_update_freq,
        max_steps=args.max_steps
    )
    
    # Print training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Results saved in: {run_dir}")

if __name__ == "__main__":
    main()