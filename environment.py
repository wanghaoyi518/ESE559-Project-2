import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from project2 import move  # Import the provided move function

class RobotEnv:
    """
    Environment class for the robot navigation task.
    
    Implements the environment for Problem 1 of the ESE 559 Project 2, where
    the robot needs to navigate to a fixed goal location in a known environment.
    """
    
    def __init__(self, initial_state=(-1.2, -1.2, 0)):
        """
        Initialize the robot environment for Problem 1.
        
        Args:
            initial_state (tuple): Initial robot state as (px, py, phi).
        """
        # Fixed obstacles for Problem 1
        self.obstacles = [
            {'x': -0.4, 'y': -0.4, 'r': 0.16},
            {'x': 0.1, 'y': -0.4, 'r': 0.16},
            {'x': -0.4, 'y': 0.1, 'r': 0.17}
        ]
        
        # Fixed goal position for Problem 1
        self.goal_pos = (1.2, 1.2)
        
        self.initial_state = initial_state
        self.state = initial_state
        self.goal_radius = 0.08  # Goal tolerance radius as specified in the problem
        self.robot_radius = 0.08  # Assumed robot radius for collision detection
        
        # Define discount factor for reward shaping
        self.gamma = 0.99
        
        # Define the state space boundaries
        self.state_space = {
            'x_min': -1.3, 'x_max': 1.3,
            'y_min': -1.3, 'y_max': 1.3,
            'phi_min': -np.pi, 'phi_max': np.pi
        }
        
        # Load action map from pickle file if available
        try:
            with open('action_map_project_2.pkl', 'rb') as f:
                self.action_map = pickle.load(f)
        except FileNotFoundError:
            # If file not found, define the action map manually as specified in the problem
            self.action_map = {
                0: (-0.25, -1.82), 1: (-0.25, -1.46), 2: (-0.25, -1.09),
                3: (-0.25, -0.73), 4: (-0.25, -0.36), 5: (-0.25, 0),
                6: (-0.25, 0.36), 7: (-0.25, 0.73), 8: (-0.25, 1.09),
                9: (-0.25, 1.46), 10: (-0.25, 1.82), 11: (0.25, -1.82),
                12: (0.25, -1.46), 13: (0.25, -1.09), 14: (0.25, -0.73),
                15: (0.25, -0.36), 16: (0.25, 0), 17: (0.25, 0.36),
                18: (0.25, 0.73), 19: (0.25, 1.09), 20: (0.25, 1.46),
                21: (0.25, 1.82), 22: (0, 0)
            }
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            tuple: The initial state (px, py, phi).
        """
        self.state = self.initial_state
        return self.state
    
    def step(self, state, action_idx):
        """
        Take a step in the environment by applying an action.
        
        Args:
            state (tuple): Current state (px, py, phi).
            action_idx (int): Index of the action to take.
            
        Returns:
            tuple: Next state (px, py, phi).
            float: Reward received.
            bool: Whether the episode is done.
        """
        # Get action parameters from action map
        v, w = self.action_map[action_idx]
        
        # Apply action using the provided 'move' function
        next_state = move(state, [v, w])
        
        # Ensure the angles stay within [-π, π]
        next_state = (next_state[0], next_state[1], self.normalize_angle(next_state[2]))
        
        # Check for collision
        collision = self.check_collision(next_state)
        
        # Check if goal reached
        reached_goal = self.check_goal(next_state)
        
        # Check if out of bounds
        out_of_bounds = self.check_out_of_bounds(next_state)
        
        # Calculate reward and done status
        reward, done = self.get_reward(state, next_state, action_idx, collision, reached_goal, out_of_bounds)
        
        # Always terminate the episode if collision happens or goal is reached
        if collision or reached_goal:
            done = True
        
        # Update current state if no collision and not out of bounds
        if not collision and not out_of_bounds:
            self.state = next_state
        
        return next_state, reward, done
    
    def check_collision(self, state):
        """
        Check if the robot collides with any obstacle.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            
        Returns:
            bool: True if collision, False otherwise.
        """
        px, py, _ = state
        
        # Check collision with each obstacle
        for obs in self.obstacles:
            # Calculate distance to obstacle center
            dist = np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2)
            
            # Check if robot is inside obstacle (accounting for robot radius)
            if dist < obs['r'] + self.robot_radius:
                return True
        
        return False
    
    def check_goal(self, state):
        """
        Check if the robot has reached the goal.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            
        Returns:
            bool: True if goal reached, False otherwise.
        """
        px, py, _ = state
        
        # Calculate distance to goal
        dist = np.sqrt((px - self.goal_pos[0])**2 + (py - self.goal_pos[1])**2)
        
        # Check if robot is within goal tolerance
        return dist <= self.goal_radius
    
    def check_out_of_bounds(self, state):
        """
        Check if the robot is out of the state space bounds.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            
        Returns:
            bool: True if out of bounds, False otherwise.
        """
        px, py, _ = state
        
        # Check if position is within bounds
        if (px < self.state_space['x_min'] or px > self.state_space['x_max'] or
            py < self.state_space['y_min'] or py > self.state_space['y_max']):
            return True
        
        return False
    
    def get_reward(self, state, next_state, action_idx, collision, reached_goal, out_of_bounds):
        """
        Calculate the reward for a state transition.
        
        Args:
            state (tuple): Current state (px, py, phi).
            next_state (tuple): Next state (px, py, phi).
            action_idx (int): Index of the action taken.
            collision (bool): Whether the robot collided with an obstacle.
            reached_goal (bool): Whether the robot reached the goal.
            out_of_bounds (bool): Whether the robot went out of bounds.
            
        Returns:
            float: Reward value.
            bool: Whether the episode is done.
        """
        # Extract positions
        px, py, _ = state
        px_next, py_next, _ = next_state
        gx, gy = self.goal_pos
        
        # Calculate distances to goal
        prev_dist = np.sqrt((px - gx)**2 + (py - gy)**2)
        curr_dist = np.sqrt((px_next - gx)**2 + (py_next - gy)**2)
        
        # Base reward: smaller penalty
        reward = -0.05
        
        # Stronger progress reward
        dist_diff = prev_dist - curr_dist
        reward += 0.8 * dist_diff
        
        # Add potential-based shaping reward
        potential_prev = -prev_dist
        potential_curr = -curr_dist
        reward += 0.8 * (potential_curr - potential_prev * self.gamma)
        
        # Collision penalty
        if collision:
            reward = -10.0  # Reduced to avoid too harsh penalties
            return reward, True
        
        if out_of_bounds:
            reward = -10.0
            return reward, True
    
        # Goal reward
        if reached_goal:
            reward = 20.0
            return reward, True
        
        # Penalize staying still (action index 22 is (0,0))
        if action_idx == 22 and not reached_goal:
            reward -= 0.2
        
        # Episode is not done
        return reward, False
    
    def normalize_angle(self, angle):
        """
        Normalize angle to the range [-π, π].
        
        Args:
            angle (float): Angle in radians.
            
        Returns:
            float: Normalized angle in radians.
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def visualize(self, trajectory=None, title=None):
        """
        Visualize the environment with obstacles, goal, and optionally a trajectory.
        
        Args:
            trajectory (list, optional): List of states representing a trajectory.
            title (str, optional): Title for the plot.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot obstacles
        for obs in self.obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
            ax.add_patch(circle)
        
        # Plot goal area
        goal_circle = plt.Circle(self.goal_pos, self.goal_radius, color='green', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'g*', markersize=10)
        
        # Plot initial position
        ax.plot(self.initial_state[0], self.initial_state[1], 'bo', markersize=8)
        
        # Plot trajectory if provided
        if trajectory:
            # Extract positions from trajectory
            xs = [state[0] for state in trajectory]
            ys = [state[1] for state in trajectory]
            
            # Plot trajectory line
            ax.plot(xs, ys, 'b-', alpha=0.7, linewidth=2)
            
            # Plot final position
            ax.plot(xs[-1], ys[-1], 'ro' if not self.check_goal((xs[-1], ys[-1], 0)) else 'go', markersize=8)
        
        # Set plot limits to match state space
        ax.set_xlim([self.state_space['x_min'] - 0.2, self.state_space['x_max'] + 0.2])
        ax.set_ylim([self.state_space['y_min'] - 0.2, self.state_space['y_max'] + 0.2])
        
        # Set title if provided
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Robot Environment - Goal: {self.goal_pos}")
        
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create the environment
    env = RobotEnv()
    
    # Visualize environment
    fig = env.visualize(title="Problem 1 Environment")
    plt.show()