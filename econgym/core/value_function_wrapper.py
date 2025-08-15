import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List, Union
from econgym.core.base_env import EconEnv
from dataclasses import dataclass
import gym
from gym.spaces import Box, Discrete


@dataclass
class ValueFunctionConfig:
    """Configuration for value function approximation."""
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    gamma: float = 0.99  # Discount factor
    
    # Optional: specify which parameters to learn
    learnable_params: List[str] = None


class ValueFunctionNetwork(nn.Module):
    """Neural network for value function approximation."""
    
    def __init__(self, state_dim: int, config: ValueFunctionConfig):
        super().__init__()
        
        # Build network architecture
        layers = []
        input_dim = state_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
                nn.Dropout(0.1)
            ])
            input_dim = config.hidden_dim
        
        # Output layer: predict value
        layers.append(nn.Linear(config.hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value function network."""
        return self.network(state)


class ValueFunctionWrapper(EconEnv):
    """
    A wrapper for EconEnv that adds value function approximation capabilities.
    Supports both traditional value function iteration and neural network-based approximation.
    """
    
    def __init__(self, env: EconEnv, config: Optional[ValueFunctionConfig] = None):
        """
        Initialize the value function wrapper.
        
        Args:
            env: The underlying economic environment
            config: Configuration for value function approximation
        """
        super().__init__()
        self.env = env
        self.config = config or ValueFunctionConfig()
        
        # Initialize value function network
        state_dim = np.prod(env.observation_space.shape)
        self.value_network = ValueFunctionNetwork(state_dim, self.config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Store transition data
        self.transitions = []
        
        # Propagate spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Propagate parameters
        if hasattr(env, 'parameters'):
            self._parameters = env.parameters
    
    def reset(self):
        """Reset the environment and clear transition data."""
        self.transitions = []
        return self.env.reset()
    
    def step(self, action):
        """Take a step in the environment and store the transition."""
        next_state, reward, done, info = self.env.step(action)
        
        # Store transition for value function learning
        self.transitions.append({
            'state': self.state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'done': done
        })
        
        return next_state, reward, done, info
    
    def train_value_function(self):
        """Train the value function network on collected transitions."""
        if len(self.transitions) < self.config.batch_size:
            return
        
        # Convert transitions to tensors
        states = torch.FloatTensor([t['state'] for t in self.transitions])
        next_states = torch.FloatTensor([t['next_state'] for t in self.transitions])
        rewards = torch.FloatTensor([t['reward'] for t in self.transitions])
        dones = torch.FloatTensor([t['done'] for t in self.transitions])
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Create batches
            indices = np.random.permutation(len(self.transitions))
            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                
                # Forward pass
                current_values = self.value_network(states[batch_indices])
                next_values = self.value_network(next_states[batch_indices])
                
                # Compute target values
                target_values = rewards[batch_indices] + \
                              (1 - dones[batch_indices]) * self.config.gamma * next_values.detach()
                
                # Compute loss
                loss = nn.MSELoss()(current_values, target_values)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get the value of a state using the learned value function."""
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            value = self.value_network(state_tensor)
        return value.item()
    
    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """
        Get the optimal action for a state using the learned value function.
        Uses one-step lookahead.
        """
        best_action = None
        best_value = -np.inf
        
        # Try all possible actions
        for action in self._get_possible_actions():
            next_state, reward, done, _ = self.env.step(action)
            value = reward + self.config.gamma * self.get_value(next_state)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _get_possible_actions(self) -> List[np.ndarray]:
        """Get all possible actions from the action space."""
        if isinstance(self.action_space, Discrete):
            return [np.array([i]) for i in range(self.action_space.n)]
        elif isinstance(self.action_space, Box):
            # For continuous action spaces, discretize
            num_actions = 10
            actions = []
            for i in range(num_actions):
                action = self.action_space.low + (self.action_space.high - self.action_space.low) * i / (num_actions - 1)
                actions.append(action)
            return actions
        else:
            raise NotImplementedError(f"Action space type {type(self.action_space)} not supported")
    
    def find_equilibrium(self, method: str = "value_iteration", **kwargs) -> Dict[str, Any]:
        """
        Find equilibrium using either value function iteration or neural network approximation.
        
        Args:
            method: Either "value_iteration" or "neural_network"
            **kwargs: Additional arguments for the chosen method
            
        Returns:
            Dictionary containing equilibrium results
        """
        if method == "value_iteration":
            if hasattr(self.env, "find_equilibrium"):
                return self.env.find_equilibrium(**kwargs)
            raise NotImplementedError("Environment does not support value iteration")
        
        elif method == "neural_network":
            # Train value function
            self.train_value_function()
            
            # Compute equilibrium using learned value function
            states = []
            values = []
            actions = []
            
            # Sample states from the state space
            num_samples = 1000
            for _ in range(num_samples):
                state = self.env.reset()
                value = self.get_value(state)
                action = self.get_policy(state)
                
                states.append(state)
                values.append(value)
                actions.append(action)
            
            return {
                'states': np.array(states),
                'values': np.array(values),
                'actions': np.array(actions),
                'method': 'neural_network'
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """Calibrate the environment parameters."""
        if hasattr(self.env, "calibrate"):
            return self.env.calibrate(targets, method, **kwargs)
        raise NotImplementedError("Environment does not support calibration.")
    
    def estimate(self, data: np.ndarray, moment_function: callable, 
                weight_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Estimate environment parameters."""
        if hasattr(self.env, "estimate"):
            return self.env.estimate(data, moment_function, weight_matrix, **kwargs)
        raise NotImplementedError("Environment does not support estimation.")
    
    def render(self, mode="human"):
        """Render the environment."""
        if hasattr(self.env, "render"):
            return self.env.render(mode)
        return None
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.env, "close"):
            self.env.close()


# Example usage:
if __name__ == "__main__":
    from envs.aiyagari_env import AiyagariEnv
    
    # Create environment
    env = AiyagariEnv()
    
    # Configure value function approximation
    config = ValueFunctionConfig(
        hidden_dim=128,
        num_layers=2,
        learning_rate=1e-3,
        gamma=0.95
    )
    
    # Wrap environment
    value_env = ValueFunctionWrapper(env, config)
    
    # Find equilibrium using neural network
    equilibrium = value_env.find_equilibrium(method="neural_network")
    
    # Print results
    print("Equilibrium States:", equilibrium['states'].shape)
    print("Equilibrium Values:", equilibrium['values'].shape)
    print("Equilibrium Actions:", equilibrium['actions'].shape)
    
    # Get value and policy for a specific state
    state = env.reset()
    value = value_env.get_value(state)
    action = value_env.get_policy(state)
    
    print(f"\nState: {state}")
    print(f"Value: {value}")
    print(f"Optimal Action: {action}") 