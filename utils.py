import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from matplotlib.patches import Circle, Rectangle

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_obstacles_and_goal(ax, obstacles, goal_pos=(1.2, 1.2), goal_radius=0.08):
    """Plot obstacles and goal on the given axes."""
    # Plot obstacles
    for obs in obstacles:
        circle = Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
        ax.add_patch(circle)
    
    # Plot goal
    goal_circle = Circle(goal_pos, goal_radius, color='green', alpha=0.3)
    ax.add_patch(goal_circle)
    ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=10)
    
    # Plot initial position region
    init_rect = Rectangle((-1.3, -1.3), 0.1, 0.1, color='blue', alpha=0.3)
    ax.add_patch(init_rect)
    
    # Set plot limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    
    return ax

def summarize_results(results, title="Test Results"):
    """Create a summary plot of test results."""
    # Extract environment names and success rates
    env_names = [key for key in results.keys() if key != 'overall']
    success_rates = [results[env]['success_rate'] for env in env_names]
    
    # Handle infinite avg_steps values
    avg_steps = []
    for env in env_names:
        if results[env]['avg_steps'] == "inf":
            avg_steps.append(0)  # Use 0 for visualization
        else:
            avg_steps.append(results[env]['avg_steps'])
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot success rates and average steps
    x = np.arange(len(env_names))
    width = 0.35
    
    ax1.bar(x - width/2, success_rates, width, label='Success Rate', color='blue', alpha=0.7)
    ax2.bar(x + width/2, avg_steps, width, label='Avg Steps', color='green', alpha=0.7)
    
    # Add overall line
    if 'overall' in results:
        ax1.axhline(y=results['overall']['success_rate'], color='red', linestyle='-', 
                   label=f"Overall: {results['overall']['success_rate']:.2f}")
    
    # Customize plot
    ax1.set_xlabel('Environment')
    ax1.set_ylabel('Success Rate')
    ax2.set_ylabel('Average Steps to Goal')
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_names)
    ax1.set_ylim([0, 1.1])
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig