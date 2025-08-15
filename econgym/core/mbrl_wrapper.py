import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from econgym.core.base_env import EconEnv
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MBRLConfig:
    """Configuration for MBRL training."""
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "relu"
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100
    validation_split: float = 0.2
    
    # MBRL specific
    horizon: int = 5  # Planning horizon
    num_particles: int = 10  # Number of particles for uncertainty estimation
    temperature: float = 1.0  # Temperature for exploration
    
    # Optional: specify which parameters to learn
    learnable_params: List[str] = None


class DynamicsModel(nn.Module):
    """Neural network model for learning environment dynamics."""
    
    def __init__(self, state_dim: int, action_dim: int, config: MBRLConfig):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network architecture
        layers = []
        input_dim = state_dim + action_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
                nn.Dropout(0.1)
            ])
            input_dim = config.hidden_dim
        
        # Output layer: predict next state and reward
        layers.append(nn.Linear(config.hidden_dim, state_dim + 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the dynamics model."""
        x = torch.cat([state, action], dim=-1)
        output = self.network(x)
        next_state = output[:, :self.state_dim]
        reward = output[:, -1:]
        return next_state, reward


class MBRLWrapper(EconEnv):
    """
    A wrapper for EconEnv that adds Model-Based Reinforcement Learning capabilities.
    This wrapper learns a dynamics model of the environment and uses it for planning.
    """
    
    def __init__(self, env: EconEnv, config: Optional[MBRLConfig] = None):
        """
        Initialize the MBRL wrapper.
        
        Args:
            env: The underlying economic environment
            config: Configuration for MBRL training
        """
        super().__init__()
        self.env = env
        self.config = config or MBRLConfig()
        
        # Initialize dynamics model
        state_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)
        self.dynamics_model = DynamicsModel(state_dim, action_dim, self.config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.dynamics_model.parameters(),
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
        """
        Take a step in the environment and store the transition.
        """
        next_state, reward, done, info = self.env.step(action)
        
        # Store transition for model learning
        self.transitions.append({
            'state': self.state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'done': done
        })
        
        return next_state, reward, done, info
    
    def train_dynamics_model(self):
        """
        Train the dynamics model on collected transitions.
        """
        if len(self.transitions) < self.config.batch_size:
            return
        
        # Convert transitions to tensors
        states = torch.FloatTensor([t['state'] for t in self.transitions])
        actions = torch.FloatTensor([t['action'] for t in self.transitions])
        next_states = torch.FloatTensor([t['next_state'] for t in self.transitions])
        rewards = torch.FloatTensor([t['reward'] for t in self.transitions])
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Create batches
            indices = np.random.permutation(len(self.transitions))
            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                
                # Forward pass
                pred_next_states, pred_rewards = self.dynamics_model(
                    states[batch_indices],
                    actions[batch_indices]
                )
                
                # Compute loss
                state_loss = nn.MSELoss()(pred_next_states, next_states[batch_indices])
                reward_loss = nn.MSELoss()(pred_rewards, rewards[batch_indices])
                total_loss = state_loss + reward_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
    
    def plan_action(self, state: np.ndarray) -> np.ndarray:
        """
        Plan the next action using the learned dynamics model.
        Uses Model Predictive Control (MPC) with random shooting.
        """
        state = torch.FloatTensor(state)
        best_action = None
        best_value = -np.inf
        
        # Generate random action sequences
        for _ in range(self.config.num_particles):
            # Sample random actions
            actions = []
            for _ in range(self.config.horizon):
                action = self.action_space.sample()
                actions.append(action)
            
            # Simulate trajectory
            current_state = state
            total_reward = 0
            
            for action in actions:
                action_tensor = torch.FloatTensor(action)
                next_state, reward = self.dynamics_model(current_state, action_tensor)
                total_reward += reward.item()
                current_state = next_state
            
            # Update best action if better
            if total_reward > best_value:
                best_value = total_reward
                best_action = actions[0]  # Use first action from best sequence
        
        return best_action
    
    def find_equilibrium(self):
        """
        Find equilibrium using the learned dynamics model.
        """
        if hasattr(self.env, "find_equilibrium"):
            return self.env.find_equilibrium()
        raise NotImplementedError("No equilibrium solver available.")
    
    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """
        Calibrate the environment parameters.
        """
        if hasattr(self.env, "calibrate"):
            return self.env.calibrate(targets, method, **kwargs)
        raise NotImplementedError("Environment does not support calibration.")
    
    def estimate(self, data: np.ndarray, moment_function: callable, 
                weight_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate environment parameters.
        """
        if hasattr(self.env, "estimate"):
            return self.env.estimate(data, moment_function, weight_matrix, **kwargs)
        raise NotImplementedError("Environment does not support estimation.")
    
    def render(self, mode="human"):
        """
        Render the environment.
        """
        if hasattr(self.env, "render"):
            return self.env.render(mode)
        return None
    
    def close(self):
        """
        Clean up resources.
        """
        if hasattr(self.env, "close"):
            self.env.close()


# Example usage:
if __name__ == "__main__":
    from envs.Ryan_env import RyanEnv
    
    # Create environment
    env = RyanEnv(n_firms=3)
    
    # Wrap with MBRL
    mbrl_config = MBRLConfig(
        hidden_dim=128,
        num_layers=2,
        horizon=3,
        num_particles=20
    )
    mbrl_env = MBRLWrapper(env, mbrl_config)
    
    # Training loop
    for episode in range(100):
        state = mbrl_env.reset()
        episode_reward = 0
        
        for step in range(50):
            # Plan action using learned model
            action = mbrl_env.plan_action(state)
            
            # Take step in environment
            next_state, reward, done, _ = mbrl_env.step(action)
            episode_reward += reward
            
            # Train dynamics model
            mbrl_env.train_dynamics_model()
            
            if done:
                break
        
        print(f"Episode {episode}, Reward: {episode_reward}") 