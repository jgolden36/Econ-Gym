import numpy as np
from gymnasium import Wrapper
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class MBRLConfig:
    """Configuration for Model-Based Reinforcement Learning."""
    hidden_dim: int = 128
    num_layers: int = 2
    horizon: int = 3
    num_particles: int = 20

class DynamicsModel(nn.Module):
    """Neural network model for dynamics prediction."""
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class MBRLWrapper(Wrapper):
    """
    A wrapper that adds Model-Based Reinforcement Learning capabilities to an environment.
    """
    def __init__(self, env, config: MBRLConfig):
        super().__init__(env)
        self.config = config
        
        # Initialize dynamics model
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.dynamics_model = DynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.dynamics_model.parameters())
        
        # Store transitions for training
        self.transitions = []
        
    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple):
            self.state = result[0]
        else:
            self.state = result
        return result

    def step(self, action):
        """Execute one step in the environment."""
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # Store transition for training
        self.transitions.append((self.state, action, next_state, reward))
        self.state = next_state
        
        return next_state, reward, done, truncated, info
    
    def train_dynamics_model(self):
        """Train the dynamics model on stored transitions."""
        if len(self.transitions) < 2:
            return
            
        # Prepare batch
        states = torch.FloatTensor([t[0] for t in self.transitions])
        actions = torch.FloatTensor([t[1] for t in self.transitions])
        next_states = torch.FloatTensor([t[2] for t in self.transitions])
        
        # Forward pass
        predicted_next_states = self.dynamics_model(states, actions)
        
        # Compute loss
        loss = nn.MSELoss()(predicted_next_states, next_states)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear old transitions
        self.transitions = self.transitions[-1000:]  # Keep last 1000 transitions
        
    def plan_action(self, state):
        """Plan action using learned dynamics model."""
        # Convert state to tensor
        state = torch.FloatTensor(state)
        
        # Handle MultiDiscrete action space
        if hasattr(self.env.action_space, 'nvec'):
            low = np.zeros_like(self.env.action_space.nvec)
            high = self.env.action_space.nvec - 1
        else:
            low = self.env.action_space.low
            high = self.env.action_space.high
        
        # Sample random actions
        actions = torch.FloatTensor(np.random.uniform(
            low=low,
            high=high + 1,  # +1 because upper bound is exclusive in np.random.uniform
            size=(self.config.num_particles, self.env.action_space.shape[0])
        ))
        
        # Simulate trajectories
        best_reward = float('-inf')
        best_action = None
        
        for action in actions:
            total_reward = 0
            current_state = state.clone()
            
            for _ in range(self.config.horizon):
                # Predict next state
                next_state = self.dynamics_model(current_state.unsqueeze(0), action.unsqueeze(0))
                current_state = next_state.squeeze(0)
                
                # Compute reward (simplified)
                total_reward -= torch.norm(current_state - state)  # Penalize deviation from initial state
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
        
        # For MultiDiscrete, round and cast to int
        if hasattr(self.env.action_space, 'nvec'):
            return np.round(best_action.numpy()).astype(int)
        return best_action.numpy() 