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
        reward = -0.1
        
        # Stronger progress reward
        dist_diff = prev_dist - curr_dist
        reward += 0.8 * dist_diff
        
        # # Add potential-based shaping reward
        # potential_prev = -prev_dist
        # potential_curr = -curr_dist
        # reward += 0.8 * (potential_curr - potential_prev * self.gamma)
        
        # Collision penalty
        if collision:
            reward = -10.0  # Reduced to avoid too harsh penalties
            return reward, True
        
        if out_of_bounds:
            reward = -10.0
            return reward, True
    
        # Goal reward
        if reached_goal:
            reward = 30.0
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

class Problem2Env(RobotEnv):
    """Environment class for Problem 2 with multiple training environments."""
    
    def __init__(self, obstacles=None, goal_position=(1.2, 1.2), initial_state=(-1.2, -1.2, 0)):
        """Initialize environment with custom obstacles configuration."""
        super().__init__(initial_state)
        
        if obstacles is not None:
            self.obstacles = obstacles
        
        self.goal_pos = goal_position
    
    @staticmethod
    def get_training_environments(additional_envs=20, hard_test_envs=10):
        """
        Return training environments for Problem 2 with additional random environments
        and environments hard test cases.
        
        Args:
            additional_envs (int): Number of additional random environments to generate
                                   beyond the predefined ones.
            hard_test_envs (int): Number of environments hard test cases to generate.
        
        Returns:
            list: List of obstacle configurations for training.
        """
        # Predefined environments from the problem statement
        predefined_envs = [
            # Case 1
            [
                {'x': -0.4, 'y': -0.4, 'r': 0.16},
                {'x': 0.1, 'y': -0.4, 'r': 0.16},
                {'x': -0.4, 'y': 0.1, 'r': 0.17}
            ],
            # Case 2
            [
                {'x': -0.8, 'y': -0.4, 'r': 0.16},
                {'x': -0.1, 'y': -0.4, 'r': 0.17},
                {'x': 0.5, 'y': -0.4, 'r': 0.17}
            ],
            # Case 3
            [
                {'x': -0.6, 'y': 1.0, 'r': 0.17},
                {'x': -0.6, 'y': -1.0, 'r': 0.16},
                {'x': 1.0, 'y': -0.6, 'r': 0.17}
            ]
        ]
        
        # Generate environments hard test cases
        hard_envs = Problem2Env.generate_hard_test_environments(hard_test_envs)
        
        # Generate additional random environments
        random_envs = []
        for _ in range(additional_envs):
            random_envs.append(Problem2Env.generate_random_training_environment())
        
        # Combine predefined and random environments
        all_environments = predefined_envs + hard_envs + random_envs
        
        print(f"Created {len(all_environments)} training environments "
              f"({len(predefined_envs)} predefined, {len(hard_envs)} hard test, "
              f"{len(random_envs)} random)")
        
        return all_environments
    
    @staticmethod
    def generate_hard_test_environments(num_environments=10, variation=0.1):
        """
        Generate environments that are hard the test environments from test_case.py.
        
        Args:
            num_environments (int): Number of environments to generate.
            variation (float): Maximum variation in obstacle positions from the test case.
            
        Returns:
            list: List of obstacle configurations.
        """
        hard_envs = []
        
        # Try to import test cases
        try:
            from test_case import problem2_test
            has_test_cases = True
        except ImportError:
            has_test_cases = False
            print("Warning: test_case.py not found. Using predefined template for hard environments.")
        
        if has_test_cases:
            # Use the actual test case as a template
            template = [
                {'x': problem2_test['O1']['x'], 'y': problem2_test['O1']['y'], 'r': problem2_test['O1']['r']},
                {'x': problem2_test['O2']['x'], 'y': problem2_test['O2']['y'], 'r': problem2_test['O2']['r']},
                {'x': problem2_test['O3']['x'], 'y': problem2_test['O3']['y'], 'r': problem2_test['O3']['r']}
            ]
        else:
            # Predefined template hard the test case
            template = [
                {'x': -0.6, 'y': 0.5, 'r': 0.2},
                {'x': -0.5, 'y': -0.5, 'r': 0.2},
                {'x': 0.5, 'y': 0.5, 'r': 0.2}
            ]
        
        # Create variations of the template
        for _ in range(num_environments):
            new_env = []
            for obs in template:
                # Create a variation of the obstacle
                new_obs = {
                    'x': obs['x'] + np.random.uniform(-variation, variation),
                    'y': obs['y'] + np.random.uniform(-variation, variation),
                    'r': np.random.uniform(0.18, 0.2)  # Use large radii (harder)
                }
                new_env.append(new_obs)
            
            # Ensure the obstacles don't overlap and aren't too close to the goal or start
            if Problem2Env._validate_environment(new_env):
                hard_envs.append(new_env)
        
        return hard_envs

    @staticmethod
    def _validate_environment(obstacles, min_dist_between_obstacles=0.1, 
                             min_dist_to_goal=0.3, min_dist_to_start=0.3,
                             goal_pos=(1.2, 1.2), start_pos=(-1.2, -1.2)):
        """
        Validate that obstacles don't overlap and aren't too close to goal or start.
        
        Args:
            obstacles (list): List of obstacle dictionaries.
            min_dist_between_obstacles (float): Minimum distance between obstacles.
            min_dist_to_goal (float): Minimum distance from any obstacle to goal.
            min_dist_to_start (float): Minimum distance from any obstacle to start.
            goal_pos (tuple): Goal position.
            start_pos (tuple): Start position.
            
        Returns:
            bool: True if environment is valid, False otherwise.
        """
        # Check distance between obstacles
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                obs1, obs2 = obstacles[i], obstacles[j]
                dist = np.sqrt((obs1['x'] - obs2['x'])**2 + (obs1['y'] - obs2['y'])**2)
                if dist < (obs1['r'] + obs2['r'] + min_dist_between_obstacles):
                    return False
        
        # Check distance to goal
        for obs in obstacles:
            dist_to_goal = np.sqrt((obs['x'] - goal_pos[0])**2 + (obs['y'] - goal_pos[1])**2)
            if dist_to_goal < (obs['r'] + min_dist_to_goal):
                return False
        
        # Check distance to start
        for obs in obstacles:
            dist_to_start = np.sqrt((obs['x'] - start_pos[0])**2 + (obs['y'] - start_pos[1])**2)
            if dist_to_start < (obs['r'] + min_dist_to_start):
                return False
        
        return True
    
    @staticmethod
    def generate_random_training_environment(num_obstacles=3):
        """
        Generate a random environment for training with diverse obstacle configurations.
        
        This method creates more diverse configurations than test environments
        to improve generalization.
        
        Args:
            num_obstacles (int): Number of obstacles to generate.
            
        Returns:
            list: List of obstacle dictionaries.
        """
        obstacles = []
        
        # Minimum distance between obstacles (smaller than in test environments
        # to create more challenging scenarios)
        min_distance = 0.3
        
        # Obstacle placement patterns to ensure diversity
        placement_pattern = 'random'  # Options: 'random', 'grid', 'clustered'
        
        if placement_pattern == 'random':
            # Fully random placement
            for _ in range(num_obstacles):
                valid_position = False
                attempts = 0
                
                while not valid_position and attempts < 50:
                    attempts += 1
                    
                    # Generate random position
                    x = np.random.uniform(-1.0, 1.0)
                    y = np.random.uniform(-1.0, 1.0)
                    
                    # Generate random radius from specified range
                    r = np.random.uniform(0.16, 0.20)
                    
                    # Check distance to other obstacles
                    valid_position = True
                    for obs in obstacles:
                        dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
                        if dist < (r + obs['r'] + min_distance):
                            valid_position = False
                            break
                    
                    # Check distance to goal and initial position
                    goal_dist = np.sqrt((x - 1.2)**2 + (y - 1.2)**2)
                    init_dist = np.sqrt((x + 1.2)**2 + (y + 1.2)**2)
                    
                    if goal_dist < (r + 0.25) or init_dist < (r + 0.25):
                        valid_position = False
                
                if valid_position:
                    obstacles.append({'x': x, 'y': y, 'r': r})
        return obstacles       
        
    @staticmethod
    def generate_random_test_environment(num_obstacles=3):
        """Generate random test environments with obstacles."""
        obstacles = []
        
        # Minimum distance between obstacles
        min_distance = 0.4
        
        for _ in range(num_obstacles):
            valid_position = False
            
            while not valid_position:
                # Generate random position
                x = np.random.uniform(-1.0, 1.0)
                y = np.random.uniform(-1.0, 1.0)
                
                # Generate random radius from specified range
                r = np.random.uniform(0.16, 0.20)
                
                # Check distance to other obstacles
                valid_position = True
                for obs in obstacles:
                    dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
                    if dist < (r + obs['r'] + min_distance):
                        valid_position = False
                        break
                
                # Check distance to goal and initial position
                goal_dist = np.sqrt((x - 1.2)**2 + (y - 1.2)**2)
                init_dist = np.sqrt((x + 1.2)**2 + (y + 1.2)**2)
                
                if goal_dist < (r + 0.3) or init_dist < (r + 0.3):
                    valid_position = False
            
            obstacles.append({'x': x, 'y': y, 'r': r})
        
        return obstacles
        
    def visualize_environments(self, environments, title="Training Environments"):
        """Visualize multiple environments in a single figure."""
        rows = int(np.ceil(len(environments) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for i, obstacles in enumerate(environments):
            if i < len(axes):
                ax = axes[i]
                
                # Plot obstacles
                for obs in obstacles:
                    circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
                    ax.add_patch(circle)
                
                # Plot goal
                goal_circle = plt.Circle(self.goal_pos, self.goal_radius, color='green', alpha=0.3)
                ax.add_patch(goal_circle)
                ax.plot(self.goal_pos[0], self.goal_pos[1], 'g*', markersize=10)
                
                # Plot initial position area
                rect = plt.Rectangle((-1.3, -1.3), 0.1, 0.1, color='blue', alpha=0.3)
                ax.add_patch(rect)
                
                ax.set_xlim([self.state_space['x_min'] - 0.2, self.state_space['x_max'] + 0.2])
                ax.set_ylim([self.state_space['y_min'] - 0.2, self.state_space['y_max'] + 0.2])
                ax.set_title(f"Environment {i+1}")
                ax.grid(True)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

class Problem3Env(Problem2Env):
    """Environment class for Problem 3 with multiple training environments and variable goal locations."""
    
    def __init__(self, obstacles=None, goal_position=None, initial_state=(-1.2, -1.2, 0)):
        """Initialize environment with custom obstacles and goal configuration."""
        super().__init__(initial_state=initial_state)
        
        if obstacles is not None:
            self.obstacles = obstacles
        
        if goal_position is not None:
            self.goal_pos = goal_position
    
    @staticmethod
    def generate_random_goal(obstacles=None, min_obstacle_distance=0.25):
        """
        Generate a random goal location based on the specified ranges for Problem 3.
        Ensures the goal does not conflict with any obstacles.
        
        Args:
            obstacles (list): List of obstacle dictionaries to check for conflicts.
            min_obstacle_distance (float): Minimum allowed distance between goal and any obstacle.
            
        Returns:
            tuple: Generated goal position (x_g, y_g).
        """
        if obstacles is None:
            obstacles = []
            
        # Goal radius for collision checking
        goal_radius = 0.08
        
        # Try up to 50 times to generate a valid goal
        for _ in range(50):
            if np.random.random() < 0.5:
                # First condition: x_g ∈ [0.8, 1] and y_g ∈ [0.8, 1]
                x_g = np.random.uniform(0.8, 1.0)
                y_g = np.random.uniform(0.8, 1.0)
            else:
                # Second condition: x_g ∈ [-1.1, -0.9] and y_g ∈ [0.8, 1]
                x_g = np.random.uniform(-1.1, -0.9)
                y_g = np.random.uniform(0.8, 1.0)
            
            # Check if goal conflicts with any obstacle
            valid_goal = True
            for obs in obstacles:
                # Calculate distance between goal and obstacle center
                dist = np.sqrt((x_g - obs['x'])**2 + (y_g - obs['y'])**2)
                
                # Check if goal is too close to obstacle (considering both radii)
                if dist < (obs['r'] + goal_radius + min_obstacle_distance):
                    valid_goal = False
                    break
            
            # If goal is valid, return it
            if valid_goal:
                return (x_g, y_g)
        
        # If we couldn't find a valid goal after several attempts,
        # return a position that matches problem constraints but might be close to obstacles
        # This is a fallback that should rarely be used
        if np.random.random() < 0.5:
            return (np.random.uniform(0.8, 1.0), np.random.uniform(0.8, 1.0))
        else:
            return (np.random.uniform(-1.1, -0.9), np.random.uniform(0.8, 1.0))
    
    @staticmethod
    def get_training_environments_with_goals(additional_envs=20, hard_test_envs=10):
        """
        Return training environments and goals for Problem 3 with environments hard test cases.
        
        Args:
            additional_envs (int): Number of additional random environments to generate
                                   beyond the predefined ones.
            hard_test_envs (int): Number of environments hard test cases to generate.
        
        Returns:
            list: List of (obstacle configuration, goal position) pairs for training.
        """
        # Predefined environments and goals from the problem statement
        predefined_envs_goals = [
            # Case 1
            ([
                {'x': -0.4, 'y': -0.4, 'r': 0.16},
                {'x': 0.1, 'y': -0.4, 'r': 0.16},
                {'x': -0.4, 'y': 0.1, 'r': 0.17}
            ], (0.9, 0.9)),
            
            # Case 2
            ([
                {'x': -0.8, 'y': -0.4, 'r': 0.16},
                {'x': -0.1, 'y': -0.4, 'r': 0.17},
                {'x': 0.5, 'y': -0.4, 'r': 0.17}
            ], (0.82, 0.95)),
            
            # Case 3
            ([
                {'x': -0.6, 'y': 1.0, 'r': 0.17},
                {'x': -0.6, 'y': -1.0, 'r': 0.16},
                {'x': 1.0, 'y': -0.6, 'r': 0.17}
            ], (-1.0, 0.9))
        ]
        
        # Generate environments and goals hard test cases
        hard_envs_goals = Problem3Env.generate_hard_test_environments_with_goals(hard_test_envs)
        
        # Generate additional random environments with random goals
        random_envs_goals = []
        for _ in range(additional_envs):
            obstacles = Problem2Env.generate_random_training_environment()
            # Generate a goal that doesn't conflict with the obstacles
            goal = Problem3Env.generate_random_goal(obstacles=obstacles)
            random_envs_goals.append((obstacles, goal))
        
        # Combine all environments with goals
        all_environments_goals = predefined_envs_goals + hard_envs_goals + random_envs_goals
        
        print(f"Created {len(all_environments_goals)} training environments with goals "
              f"({len(predefined_envs_goals)} predefined, {len(hard_envs_goals)} hard test, "
              f"{len(random_envs_goals)} random)")
        
        return all_environments_goals

    @staticmethod
    def generate_hard_test_environments_with_goals(num_environments=10, variation=0.1):
        """
        Generate environments and goals that are hard the test environments from test_case.py.
        
        Args:
            num_environments (int): Number of environments to generate.
            variation (float): Maximum variation in obstacle positions from the test case.
            
        Returns:
            list: List of (obstacle configuration, goal position) pairs.
        """
        hard_envs_goals = []
        
        # Try to import test cases
        try:
            from test_case import problem3_test1, problem3_test_2
            has_test_cases = True
        except ImportError:
            has_test_cases = False
            print("Warning: test_case.py not found. Using predefined templates for hard environments.")
        
        if has_test_cases:
            # Use the actual test cases as templates
            templates = [
                # Template 1
                ([
                    {'x': problem3_test1['O1']['x'], 'y': problem3_test1['O1']['y'], 'r': problem3_test1['O1']['r']},
                    {'x': problem3_test1['O2']['x'], 'y': problem3_test1['O2']['y'], 'r': problem3_test1['O2']['r']},
                    {'x': problem3_test1['O3']['x'], 'y': problem3_test1['O3']['y'], 'r': problem3_test1['O3']['r']}
                ], problem3_test1['x_goal'], problem3_test1['y_goal']),
                
                # Template 2
                ([
                    {'x': problem3_test_2['O1']['x'], 'y': problem3_test_2['O1']['y'], 'r': problem3_test_2['O1']['r']},
                    {'x': problem3_test_2['O2']['x'], 'y': problem3_test_2['O2']['y'], 'r': problem3_test_2['O2']['r']},
                    {'x': problem3_test_2['O3']['x'], 'y': problem3_test_2['O3']['y'], 'r': problem3_test_2['O3']['r']}
                ], problem3_test_2['x_goal'], problem3_test_2['y_goal'])
            ]
        else:
            # Predefined templates hard the test cases
            templates = [
                # Template 1 - hard problem3_test1
                ([
                    {'x': 1.0, 'y': 0, 'r': 0.2},
                    {'x': 0.3, 'y': 1.0, 'r': 0.2},
                    {'x': 0.3, 'y': 0.5, 'r': 0.2}
                ], [0.8, 1], [0.8, 1]),
                
                # Template 2 - hard problem3_test_2
                ([
                    {'x': -1, 'y': 0.3, 'r': 0.2},
                    {'x': -1, 'y': 1.3, 'r': 0.2},
                    {'x': -0.2, 'y': 0.9, 'r': 0.2}
                ], [-1.1, -0.9], [0.8, 1])
            ]
        
        # Create variations of the templates
        for template_idx in range(len(templates)):
            for _ in range(num_environments // len(templates) + 1):  # Ensure we get enough environments
                template_obstacles, x_goal_range, y_goal_range = templates[template_idx]
                
                # Create a variation of the obstacle configuration
                new_obstacles = []
                for obs in template_obstacles:
                    new_obs = {
                        'x': obs['x'] + np.random.uniform(-variation, variation),
                        'y': obs['y'] + np.random.uniform(-variation, variation),
                        'r': np.random.uniform(0.18, 0.2)  # Use large radii (harder)
                    }
                    new_obstacles.append(new_obs)
                
                # Generate a random goal position within the specified range
                goal_x = np.random.uniform(x_goal_range[0], x_goal_range[1])
                goal_y = np.random.uniform(y_goal_range[0], y_goal_range[1])
                goal_pos = (goal_x, goal_y)
                
                # Ensure the obstacles don't overlap and aren't too close to the goal or start
                if Problem3Env._validate_environment_with_goal(new_obstacles, goal_pos):
                    hard_envs_goals.append((new_obstacles, goal_pos))
                
                # If we have enough environments, break
                if len(hard_envs_goals) >= num_environments:
                    return hard_envs_goals
        
        return hard_envs_goals

    @staticmethod
    def _validate_environment_with_goal(obstacles, goal_pos, min_dist_between_obstacles=0.1, 
                                       min_dist_to_goal=0.2, min_dist_to_start=0.3,
                                       start_pos=(-1.2, -1.2)):
        """
        Validate that obstacles don't overlap and aren't too close to goal or start.
        
        Args:
            obstacles (list): List of obstacle dictionaries.
            goal_pos (tuple): Goal position.
            min_dist_between_obstacles (float): Minimum distance between obstacles.
            min_dist_to_goal (float): Minimum distance from any obstacle to goal.
            min_dist_to_start (float): Minimum distance from any obstacle to start.
            start_pos (tuple): Start position.
            
        Returns:
            bool: True if environment is valid, False otherwise.
        """
        # Check distance between obstacles
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                obs1, obs2 = obstacles[i], obstacles[j]
                dist = np.sqrt((obs1['x'] - obs2['x'])**2 + (obs1['y'] - obs2['y'])**2)
                if dist < (obs1['r'] + obs2['r'] + min_dist_between_obstacles):
                    return False
        
        # Check distance to goal
        for obs in obstacles:
            dist_to_goal = np.sqrt((obs['x'] - goal_pos[0])**2 + (obs['y'] - goal_pos[1])**2)
            if dist_to_goal < (obs['r'] + min_dist_to_goal):
                return False
        
        # Check distance to start
        for obs in obstacles:
            dist_to_start = np.sqrt((obs['x'] - start_pos[0])**2 + (obs['y'] - start_pos[1])**2)
            if dist_to_start < (obs['r'] + min_dist_to_start):
                return False
        
        return True
        
    def visualize_environments_with_goals(self, env_goal_pairs, title="Training Environments and Goals"):
        """Visualize multiple environments with their goals in a single figure."""
        rows = int(np.ceil(len(env_goal_pairs) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for i, (obstacles, goal_pos) in enumerate(env_goal_pairs):
            if i < len(axes):
                ax = axes[i]
                
                # Plot obstacles
                for obs in obstacles:
                    circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.5)
                    ax.add_patch(circle)
                
                # Plot goal
                goal_circle = plt.Circle(goal_pos, self.goal_radius, color='green', alpha=0.3)
                ax.add_patch(goal_circle)
                ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=10)
                
                # Plot initial position area
                rect = plt.Rectangle((-1.3, -1.3), 0.1, 0.1, color='blue', alpha=0.3)
                ax.add_patch(rect)
                
                ax.set_xlim([self.state_space['x_min'] - 0.2, self.state_space['x_max'] + 0.2])
                ax.set_ylim([self.state_space['y_min'] - 0.2, self.state_space['y_max'] + 0.2])
                ax.set_title(f"Environment-Goal {i+1}")
                ax.grid(True)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig