import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from environment import *
from dqn_agent import *
from train import *
from test import *
from utils import *

def main():
    """Main function to parse arguments and start training or testing."""
    parser = argparse.ArgumentParser(description='DQN for Robot Navigation - ESE 559 Project 2')
    
    # Problem selection
    parser.add_argument('--problem', type=int, default=1, choices=[1, 2], 
                        help='Problem number to solve (1 or 2)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Mode: train or test')
    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--render', action='store_true', help='Render environment')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for experience replay')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.85, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.001, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.99, help='Exploration decay rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for DQN')
    parser.add_argument('--target-update-freq', type=int, default=10, help='Target network update frequency')
    parser.add_argument('--memory-size', type=int, default=100000, help='Size of replay memory')
    parser.add_argument('--save-freq', type=int, default=1000, help='Model saving frequency')
    parser.add_argument('--log-freq', type=int, default=100000, help='Logging frequency')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory for saving results')
    parser.add_argument('--additional-envs', type=int, default=27, help='Number of additional random environments for Problem 2')
    
    # Testing parameters
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model for testing')
    parser.add_argument('--num-trials', type=int, default=10, help='Number of trials for testing')
    parser.add_argument('--max-steps', type=int, default=150, help='Maximum steps per trial')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"problem{args.problem}_{args.mode}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    if args.problem == 1:
        # Problem 1: Fixed environment and goal
        print(f"Running Problem 1: Fixed environment with goal at (1.2, 1.2)")
        
        # Create agent for Problem 1
        agent = DQNAgent(
            state_dim=5,  # [px, py, phi, dist_to_goal, angle_diff]
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
        
        if args.mode == 'train':
            # Train agent for Problem 1
            print(f"Starting training with {args.episodes} episodes")
            
            # Train agent (pass all relevant parameters)
            rewards, _ = train(
                agent=agent,
                num_episodes=args.episodes,
                batch_size=args.batch_size,
                save_freq=args.save_freq,
                log_freq=args.log_freq,
                render=args.render,
                results_dir=results_dir,
                update_target_freq=args.target_update_freq,
                max_steps=args.max_steps
            )
            
            # After training, save the final model
            final_model_path = os.path.join(results_dir, "dqn_model_final.pth")
            agent.save(final_model_path)
            
            # Create evaluation directory
            eval_dir = os.path.join(results_dir, "test_evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Test the trained agent
            print("\nEvaluating trained agent...")
            success_rate, avg_steps = test_problem1(
                agent=agent,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                render=args.render,
                output_dir=eval_dir
            )
            
        elif args.mode == 'test':
            # Check if model path is provided
            if args.model_path is None:
                print("Error: Model path must be provided for testing mode")
                return
            
            # Load trained model
            success = agent.load(args.model_path)
            if not success:
                print(f"Error: Failed to load model from {args.model_path}")
                return
            
            print(f"Loaded model from {args.model_path}")
            
            # Test on problem 1 environment
            success_rate, avg_steps = test_problem1(
                agent=agent,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                render=args.render,
                output_dir=results_dir
            )
    
    elif args.problem == 2:
        # Problem 2: Multiple environments with fixed goal
        print(f"Running Problem 2: Multiple environments with goal at (1.2, 1.2)")
        
        # Create agent for Problem 2
        agent = EnhancedDQNAgent(
            state_dim=12,  # Enhanced state including obstacle information
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
        
        if args.mode == 'train':
            # Train agent for Problem 2
            print(f"Starting training with {args.episodes} episodes on multiple environments")
            
            # Train agent on multiple environments
            rewards, _ = train_problem2(
                agent=agent,
                num_episodes=args.episodes,
                batch_size=args.batch_size,
                save_freq=args.save_freq,
                log_freq=args.log_freq,
                render=args.render,
                results_dir=results_dir,
                update_target_freq=args.target_update_freq,
                max_steps=args.max_steps,
                additional_envs=args.additional_envs  # Pass the parameter here
            )
            
            # After training, save the final model
            final_model_path = os.path.join(results_dir, "dqn_model_final.pth")
            agent.save(final_model_path)
            
            # Create evaluation directory
            eval_dir = os.path.join(results_dir, "test_evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Test the trained agent on random environments
            print("\nEvaluating trained agent on random test environments...")
            test_results = test_problem2(
                agent=agent,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                render=args.render,
                output_dir=eval_dir
            )
            
        elif args.mode == 'test':
            # Check if model path is provided
            if args.model_path is None:
                print("Error: Model path must be provided for testing mode")
                return
            
            # Load trained model
            success = agent.load(args.model_path)
            if not success:
                print(f"Error: Failed to load model from {args.model_path}")
                return
            
            print(f"Loaded model from {args.model_path}")
            
            # Test on random environments
            test_results = test_problem2(
                agent=agent,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                render=args.render,
                output_dir=results_dir
            )

if __name__ == "__main__":
    main()