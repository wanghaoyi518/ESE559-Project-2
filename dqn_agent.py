import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import os

class DQN(nn.Module):
    """
    Deep Q-Network model for approximating the Q-function.
    
    This model takes a state representation that includes robot state,
    and outputs Q-values for each possible action.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        # A deeper network with skip connections for better learning
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, action_dim)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
    def forward(self, x):
        x1 = F.relu(self.input_layer(x))
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))
        # Add skip connection
        x3 = x3 + x1
        x4 = F.relu(self.hidden3(x3))
        return self.output_layer(x4)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

class DQNAgent:
    """
    Deep Q-Learning agent for robot navigation task.
    
    This agent learns a policy for Problem 1 with fixed goal and environment.
    """
    
    def __init__(self, state_dim=3, action_dim=23, hidden_dim=128, 
                 learning_rate=0.0005, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.001, epsilon_decay=0.998, 
                 target_update_freq=10, memory_size=100000, 
                 device=None, goal_position=(1.2, 1.2)):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (int): Dimension of the state.
            action_dim (int): Number of possible actions.
            hidden_dim (int): Dimension of hidden layers in the DQN.
            learning_rate (float): Learning rate for optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for exploration.
            target_update_freq (int): Frequency of target network updates.
            memory_size (int): Maximum size of replay memory.
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
            goal_position (tuple): Fixed goal position for the problem.
        """
        # Set device (GPU or CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Agent parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        self.goal_position = goal_position
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Use a learning rate scheduler to reduce LR over time
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        
        # Initialize replay memory with prioritized experience
        self.memory = deque(maxlen=memory_size)
        
        # Initialize step counter for target network update
        self.steps = 0
        
        # For tracking training progress
        self.loss_history = []
    
    def get_state_representation(self, state):
        """Enhanced state representation for Problem 1"""
        # Extract robot state
        px, py, phi = state
        
        # Calculate goal-relative features
        gx, gy = self.goal_position
        
        # Distance to goal
        dist_to_goal = np.sqrt((px - gx)**2 + (py - gy)**2)
        
        # Angle to goal relative to current orientation
        angle_to_goal = np.arctan2(gy - py, gx - px)
        angle_diff = self.normalize_angle(angle_to_goal - phi)
        
        # Returns 5 dimensions: [px, py, phi, dist_to_goal, angle_diff]
        return np.array([px, py, phi, dist_to_goal, angle_diff], dtype=np.float32)
    
    def normalize_angle(self, angle):
        """
        Normalize angle to the range [-π, π].
        
        Args:
            angle (float): Angle in radians.
            
        Returns:
            float: Normalized angle in radians.
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def select_action(self, state, epsilon=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            epsilon (float, optional): Override epsilon value for exploration.
            
        Returns:
            int: Selected action index.
        """
        # Use instance epsilon if not provided
        if epsilon is None:
            epsilon = self.epsilon
        
        # With probability epsilon, select random action (exploration)
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Otherwise, select best action according to Q-network (exploitation)
        state_rep = self.get_state_representation(state)
        state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
        
        # No gradient calculation needed for action selection
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Return action with highest Q-value
        return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state (tuple): Current state (px, py, phi).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state (px, py, phi).
            done (bool): Whether the episode is done.
        """
        # Convert raw states to representation
        state_rep = self.get_state_representation(state)
        next_state_rep = self.get_state_representation(next_state)
        
        # Store in memory
        self.memory.append((state_rep, action, reward, next_state_rep, done))
    
    def replay(self, batch_size):
        """
        Train the network using experience replay.
        
        Args:
            batch_size (int): Number of experiences to sample.
            
        Returns:
            float: Loss value.
        """
        # Check if enough samples in memory
        if len(self.memory) < batch_size:
            return 0
        
        # Sample batch of experiences
        batch = random.sample(self.memory, batch_size)
        
        # Extract components from batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            # Double DQN: Select actions using policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Evaluate Q-values using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        # Compute expected Q values
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Track loss history
        self.loss_history.append(loss.item())
        
        # Increment step counter and update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network by copying weights from policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Update learning rate scheduler
        self.scheduler.step()
    
    def save(self, path="models/dqn_model.pth"):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'loss_history': self.loss_history
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path="models/dqn_model.pth"):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False
        
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        # Load loss history if available
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        
        print(f"Model loaded from {path}")
        return True

class EnhancedDQNAgent(DQNAgent):
    """
    Enhanced DQN agent for Problem 2 with environmental uncertainty.
    
    This agent extends the base DQNAgent to handle obstacle information
    and learn a generalizable policy across different environments.
    """
    
    def __init__(self, state_dim=11, action_dim=23, hidden_dim=128, 
                 learning_rate=0.0005, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.001, epsilon_decay=0.998, 
                 target_update_freq=10, memory_size=100000, 
                 device=None, goal_position=(1.2, 1.2)):
        """Initialize the Enhanced DQN agent with a larger state dimension."""
        super().__init__(state_dim=state_dim, action_dim=action_dim, 
                        hidden_dim=hidden_dim, learning_rate=learning_rate,
                        gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
                        epsilon_decay=epsilon_decay, target_update_freq=target_update_freq,
                        memory_size=memory_size, device=device, goal_position=goal_position)

    def get_enhanced_state(self, state, obstacles):
        """
        Create an enhanced state representation that includes obstacle information
        with an explicit feature for nearest obstacle distance.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            obstacles (list): List of obstacle dictionaries.
            
        Returns:
            numpy.ndarray: Enhanced state representation.
        """
        # Extract basic state
        px, py, phi = state
        
        # Calculate goal-relative features
        gx, gy = self.goal_position
        
        # Distance to goal
        dist_to_goal = np.sqrt((px - gx)**2 + (py - gy)**2)
        
        # Angle to goal relative to current orientation
        angle_to_goal = np.arctan2(gy - py, gx - px)
        angle_diff = self.normalize_angle(angle_to_goal - phi)
        
        # Sort obstacles by distance to robot
        sorted_obstacles = sorted(obstacles, key=lambda obs: 
                                np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2))
        
        # Calculate distance to nearest obstacle (accounting for obstacle radius)
        if sorted_obstacles:
            nearest_obs = sorted_obstacles[0]
            nearest_obs_dist = np.sqrt((px - nearest_obs['x'])**2 + (py - nearest_obs['y'])**2) - nearest_obs['r']
            nearest_obs_dist = max(0.01, nearest_obs_dist)  # Ensure it's not negative or zero
        else:
            nearest_obs_dist = 10.0  # Large value if no obstacles
        
        # Calculate distances and angles to the three obstacles
        obstacle_features = []
        
        # Take the three obstacles (or fewer if there are less than 3)
        for i in range(min(3, len(sorted_obstacles))):
            obs = sorted_obstacles[i]
            
            # Distance to obstacle center
            dist = np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2)
            
            # Angle to obstacle relative to current orientation
            angle = np.arctan2(obs['y'] - py, obs['x'] - px)
            rel_angle = self.normalize_angle(angle - phi)
            
            obstacle_features.extend([dist, rel_angle])
        
        # Pad with zeros if fewer than 3 obstacles
        while len(obstacle_features) < 6:  # 6 = 3 obstacles * 2 features (dist, angle)
            obstacle_features.extend([10.0, 0.0])  # Large distance, zero angle
        
        # Combine all features
        # Add nearest_obs_dist as a dedicated feature just after goal information
        return np.array([px, py, phi, dist_to_goal, angle_diff, nearest_obs_dist] + obstacle_features, 
                        dtype=np.float32)
        # return np.array([
        #     px, py, phi, dist_to_goal, angle_diff, nearest_obs_dist
        # ] + [
        #     np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2) - obs['r'] if 'r' in obs else 10.0
        #     for obs in sorted_obstacles[:3]  # Only take up to 3 obstacles
        # ], dtype=np.float32)

    def select_action(self, state, obstacles, epsilon=None):
        """
        Select an action using epsilon-greedy policy with obstacle awareness.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            obstacles (list): List of obstacle dictionaries.
            epsilon (float, optional): Override epsilon value for exploration.
            
        Returns:
            int: Selected action index.
        """
        # Use instance epsilon if not provided
        if epsilon is None:
            epsilon = self.epsilon
        
        # With probability epsilon, select random action (exploration)
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Otherwise, select best action according to Q-network (exploitation)
        state_rep = self.get_enhanced_state(state, obstacles)
        state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
        
        # No gradient calculation needed for action selection
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Return action with highest Q-value
        return q_values.max(1)[1].item()
    
    def remember(self, state, obstacles, action, reward, next_state, next_obstacles, done):
        """
        Store experience in replay memory with obstacle information.
        
        Args:
            state (tuple): Current state (px, py, phi).
            obstacles (list): Current obstacle configuration.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state (px, py, phi).
            next_obstacles (list): Next obstacle configuration (usually same as obstacles).
            done (bool): Whether the episode is done.
        """
        # Convert raw states to enhanced representation
        state_rep = self.get_enhanced_state(state, obstacles)
        next_state_rep = self.get_enhanced_state(next_state, next_obstacles)
        
        # Store in memory
        self.memory.append((state_rep, action, reward, next_state_rep, done))

class GoalConditionedDQNAgent(EnhancedDQNAgent):
    """
    Goal-conditioned DQN agent for Problem 3 with environmental uncertainty and variable goals.
    
    This agent extends the EnhancedDQNAgent to handle variable goal positions
    and learn a goal-conditioned policy across different environments.
    """
    
    def __init__(self, state_dim=14, action_dim=23, hidden_dim=128, 
                 learning_rate=0.0005, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.001, epsilon_decay=0.998, 
                 target_update_freq=10, memory_size=100000, 
                 device=None):
        """Initialize the Goal-Conditioned DQN agent with correct state dimension."""
        super().__init__(state_dim=state_dim, action_dim=action_dim, 
                        hidden_dim=hidden_dim, learning_rate=learning_rate,
                        gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
                        epsilon_decay=epsilon_decay, target_update_freq=target_update_freq,
                        memory_size=memory_size, device=device)
        
        # No fixed goal position as it will vary
        self.goal_position = None

    def get_goal_conditioned_state(self, state, obstacles, goal_position):
        """
        Create a goal-conditioned state representation that includes obstacle information
        and the current goal position.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            obstacles (list): List of obstacle dictionaries.
            goal_position (tuple): Goal position (x_g, y_g).
            
        Returns:
            numpy.ndarray: Goal-conditioned state representation.
        """
        # Extract basic state
        px, py, phi = state
        
        # Get goal position
        gx, gy = goal_position
        
        # Distance to goal
        dist_to_goal = np.sqrt((px - gx)**2 + (py - gy)**2)
        
        # Angle to goal relative to current orientation
        angle_to_goal = np.arctan2(gy - py, gx - px)
        angle_diff = self.normalize_angle(angle_to_goal - phi)
        
        # Include goal position explicitly in the state
        goal_features = [gx, gy]
        
        # Sort obstacles by distance to robot
        sorted_obstacles = sorted(obstacles, key=lambda obs: 
                                np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2))
        
        # Calculate distance to nearest obstacle (accounting for obstacle radius)
        if sorted_obstacles:
            nearest_obs = sorted_obstacles[0]
            nearest_obs_dist = np.sqrt((px - nearest_obs['x'])**2 + (py - nearest_obs['y'])**2) - nearest_obs['r']
            nearest_obs_dist = max(0.01, nearest_obs_dist)  # Ensure it's not negative or zero
        else:
            nearest_obs_dist = 10.0  # Large value if no obstacles
        
        # Calculate distances and angles to the three obstacles
        obstacle_features = []
        
        # Take the three obstacles (or fewer if there are less than 3)
        for i in range(min(3, len(sorted_obstacles))):
            obs = sorted_obstacles[i]
            
            # Distance to obstacle center
            dist = np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2)
            
            # Angle to obstacle relative to current orientation
            angle = np.arctan2(obs['y'] - py, obs['x'] - px)
            rel_angle = self.normalize_angle(angle - phi)
            
            obstacle_features.extend([dist, rel_angle])
        
        # Pad with zeros if fewer than 3 obstacles
        while len(obstacle_features) < 6:  # 6 = 3 obstacles * 2 features (dist, angle)
            obstacle_features.extend([10.0, 0.0])  # Large distance, zero angle
        
        # Combine all features: base state + goal position + goal-relative features + obstacle features
        return np.array([px, py, phi] + goal_features + [dist_to_goal, angle_diff, nearest_obs_dist] + obstacle_features, 
                        dtype=np.float32)

    def select_action(self, state, obstacles, goal_position, epsilon=None):
        """
        Select an action using epsilon-greedy policy with goal conditioning.
        
        Args:
            state (tuple): Robot state (px, py, phi).
            obstacles (list): List of obstacle dictionaries.
            goal_position (tuple): Goal position (x_g, y_g).
            epsilon (float, optional): Override epsilon value for exploration.
            
        Returns:
            int: Selected action index.
        """
        # Use instance epsilon if not provided
        if epsilon is None:
            epsilon = self.epsilon
        
        # With probability epsilon, select random action (exploration)
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Otherwise, select best action according to Q-network (exploitation)
        state_rep = self.get_goal_conditioned_state(state, obstacles, goal_position)
        state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
        
        # No gradient calculation needed for action selection
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Return action with highest Q-value
        return q_values.max(1)[1].item()
    
    def remember(self, state, obstacles, action, reward, next_state, next_obstacles, done, goal_position):
        """
        Store experience in replay memory with goal and obstacle information.
        
        Args:
            state (tuple): Current state (px, py, phi).
            obstacles (list): Current obstacle configuration.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state (px, py, phi).
            next_obstacles (list): Next obstacle configuration (usually same as obstacles).
            done (bool): Whether the episode is done.
            goal_position (tuple): Goal position (x_g, y_g).
        """
        # Convert raw states to goal-conditioned representation
        state_rep = self.get_goal_conditioned_state(state, obstacles, goal_position)
        next_state_rep = self.get_goal_conditioned_state(next_state, next_obstacles, goal_position)
        
        # Store in memory
        self.memory.append((state_rep, action, reward, next_state_rep, done))