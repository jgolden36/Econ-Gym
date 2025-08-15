import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
from econgym.core.base_solver import BaseSolver
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Neural network for DQN."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class PrisonersDQNSolver(BaseSolver):
    """
    DQN solver for the Repeated Prisoner's Dilemma environment.
    """
    def __init__(self, env, config: Dict[str, Any] = None):
        super().__init__(env)
        self.config = config or {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 64,
            'target_update': 10,
            'hidden_dim': 128
        }
        
        # Initialize networks
        self.state_dim = 3  # [round, last_action_1, last_action_2]
        self.action_dim = 2  # Cooperate (0) or Defect (1)
        
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim, self.config['hidden_dim'])
        self.target_net = DQNNetwork(self.state_dim, self.action_dim, self.config['hidden_dim'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate'])
        self.memory = deque(maxlen=self.config['memory_size'])
        self.epsilon = self.config['epsilon_start']
        self.episode_rewards = []  # Track episode rewards
        
    def _get_state_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        if isinstance(state, dict):
            state_array = np.array([
                state['round'],
                state['last_actions'][0],
                state['last_actions'][1]
            ], dtype=np.float32)
        else:
            state_array = np.array(state, dtype=np.float32)
        return torch.FloatTensor(state_array)
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            state_tensor = self._get_state_tensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def solve_equilibrium(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the DQN agent to find the equilibrium strategy.
        """
        self.episode_rewards = []  # Initialize episode rewards list
        best_reward = float('-inf')
        no_improvement_count = 0
        
        for episode in range(max_iter):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action for both players
                action1 = self.select_action(state)
                action2 = self.select_action(state)  # In practice, you might want different agents
                actions = (action1, action2)
                
                # Take step in environment
                next_state, reward, done, _, info = self.env.step(actions)
                
                # Store transition in memory
                self.memory.append((state, actions, reward, next_state, done))
                
                # Train the network
                if len(self.memory) >= self.config['batch_size']:
                    self._train_step()
                
                state = next_state
                episode_reward += np.mean(reward)  # Average reward across players
            
            # Update target network
            if episode % self.config['target_update'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Update epsilon
            self.epsilon = max(self.config['epsilon_end'], 
                             self.epsilon * self.config['epsilon_decay'])
            
            # Track progress
            self.episode_rewards.append(episode_reward)  # Store episode reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= 50:
                break
        
        # Store the learned policy and value function
        self.equilibrium_policy = self.policy_net
        self.equilibrium_value = self.target_net
        self.equilibrium_state = np.array([self.target_net, self.policy_net], dtype=object)
        
        return self.equilibrium_policy, self.equilibrium_value
    
    def _train_step(self):
        """Perform one step of training."""
        # Sample batch from memory
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.stack([self._get_state_tensor(s) for s in states])
        action_batch = torch.LongTensor([a[0] for a in actions])  # Only use first player's action
        reward_batch = torch.FloatTensor([r[0] for r in rewards])  # Only use first player's reward
        next_state_batch = torch.stack([self._get_state_tensor(s) for s in next_states])
        done_batch = torch.FloatTensor(dones)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (1 - done_batch) * self.config['gamma'] * next_state_values
        
        # Compute loss and update
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _update_parameters(self, params: np.ndarray) -> None:
        """Update environment parameters."""
        self.env.parameters['rounds'] = int(params[0])
        self.env.parameters['payoff_matrix'] = {
            (0, 0): (params[1], params[1]),
            (0, 1): (params[2], params[3]),
            (1, 0): (params[3], params[2]),
            (1, 1): (params[4], params[4])
        }
    
    def _get_parameter_bounds(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for optimization."""
        x0 = [
            self.env.rounds,
            3.0,  # mutual cooperation payoff
            0.0,  # sucker's payoff
            5.0,  # temptation payoff
            1.0   # mutual defection payoff
        ]
        bounds = [
            (5, 20),    # rounds
            (2, 4),     # mutual cooperation
            (0, 1),     # sucker's payoff
            (4, 6),     # temptation
            (0, 2)      # mutual defection
        ]
        return np.array(x0), bounds
    
    def _get_parameters_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'rounds': int(params[0]),
            'mutual_cooperation': params[1],
            'sucker_payoff': params[2],
            'temptation': params[3],
            'mutual_defection': params[4]
        }
    
    def _compute_moments(self, states: List[np.ndarray], rewards: List[np.ndarray]) -> Dict[str, float]:
        """Compute moments from simulated data."""
        cooperation_rate = np.mean([np.mean(s['last_actions'] == 0) for s in states])
        avg_reward = np.mean([np.mean(r) for r in rewards])
        
        return {
            'cooperation_rate': cooperation_rate,
            'avg_reward': avg_reward
        }
    
    def _interpolate_value(self, state: np.ndarray) -> np.ndarray:
        """Get value function for given state using the trained model."""
        if self.equilibrium_value is None:
            self.solve_equilibrium()
        state_tensor = self._get_state_tensor(state)
        with torch.no_grad():
            return self.equilibrium_value(state_tensor).numpy()
    
    def _interpolate_policy(self, state: np.ndarray) -> np.ndarray:
        """Get policy for given state using the trained model."""
        if self.equilibrium_policy is None:
            self.solve_equilibrium()
        state_tensor = self._get_state_tensor(state)
        with torch.no_grad():
            return self.equilibrium_policy(state_tensor).argmax().numpy() 