import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple, List

# Optional heavy dependencies: import lazily/optionally
try:  # SB3 is optional
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
except Exception:  # pragma: no cover - optional
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore

try:  # Torch is optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore


class BBLGameEnv(EconEnv, NumpyRNGMixin):
    """
    A simplified dynamic game environment inspired by 
    Bajari, Benkard, & Levin (2007) on Estimating Dynamic Models of Imperfect Competition.
    
    This environment models a market with multiple competing firms. Each firm has its own state
    (e.g., cost, capacity, or reputation) and chooses a pricing decision from a discrete set.
    A market shock affects all firms, and the firms' payoffs depend on both their own actions and
    those of their competitors.
    """
    def __init__(self, n_firms=3, state_dim=1, beta=0.95):
        """
        Parameters:
            n_firms: Number of competing firms.
            state_dim: Dimensionality of each firm's state (here, assumed scalar).
            beta: Discount factor in the dynamic game.
        """
        super().__init__()
        # Initialize parameters
        self.parameters = {
            'n_firms': n_firms,
            'state_dim': state_dim,
            'beta': beta,
            'price_levels': 3,  # Number of possible price levels
            'state_min': 0.0,   # Minimum state value
            'state_max': 100.0, # Maximum state value
            'revenue_multiplier': 10.0,  # Multiplier for revenue calculation
            'cost_multiplier': 0.1,     # Multiplier for cost calculation
            'state_adjustment': {        # State adjustment parameters
                'low_price': -1.0,
                'high_price': 1.0,
                'medium_price_std': 0.5
            }
        }
        
        # For simplicity, assume each firm's state is continuous between 0 and 100.
        # The overall observation is an array of shape (n_firms, state_dim).
        self.observation_space = spaces.Box(
            low=self.parameters['state_min'],
            high=self.parameters['state_max'],
            shape=(self.parameters['n_firms'], self.parameters['state_dim']),
            dtype=np.float32
        )
        
        # Each firm can choose from a discrete set of price levels.
        self.action_space = spaces.MultiDiscrete([self.parameters['price_levels']] * self.parameters['n_firms'])
        
        # Initialize state
        self.state = None
        
        # Initialize solver
        self.solver = None
        # Create VecEnv only if SB3 is available
        self.vec_env = DummyVecEnv([lambda: self]) if DummyVecEnv is not None else None

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        self.reseed(seed)
        # Initialize each firm's state randomly
        self.state = self.rng.uniform(
            self.parameters['state_min'],
            self.parameters['state_max'],
            size=(self.parameters['n_firms'], self.parameters['state_dim'])
        )
        return self.state, {}

    def step(self, action):
        """
        Execute one time step in the environment dynamics.
        
        Args:
            action: Array of price levels chosen by each firm
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Calculate market demand based on average price
        avg_price = np.mean(action) / (self.parameters['price_levels'] - 1)
        market_demand = self.parameters['revenue_multiplier'] * (1 - avg_price)
        
        # Calculate each firm's revenue and cost
        revenues = []
        costs = []
        for firm in range(self.parameters['n_firms']):
            # Revenue depends on market share and price
            price = action[firm] / (self.parameters['price_levels'] - 1)
            market_share = 1 / self.parameters['n_firms']  # Equal market share for simplicity
            revenue = market_demand * market_share * price
            revenues.append(revenue)
            
            # Cost depends on state and price
            cost = self.parameters['cost_multiplier'] * self.state[firm, 0] * price
            costs.append(cost)
        
        # Calculate profits
        rewards = np.array(revenues) - np.array(costs)
        
        # Update states based on actions
        next_state = np.copy(self.state)
        for firm in range(self.parameters['n_firms']):
            if action[firm] == 0:  # Low price
                adjustment = self.parameters['state_adjustment']['low_price']
            elif action[firm] == self.parameters['price_levels'] - 1:  # High price
                adjustment = self.parameters['state_adjustment']['high_price']
            else:  # Medium price
                adjustment = self.rng.normal(0, self.parameters['state_adjustment']['medium_price_std'])
            
            next_state[firm, 0] = np.clip(
                self.state[firm, 0] + adjustment,
                self.parameters['state_min'],
                self.parameters['state_max']
            )
        
        self.state = next_state
        done = False
        truncated = False
        return next_state, rewards, done, truncated, {}

    def find_equilibrium(self):
        """
        Find the Markov Perfect Equilibrium using PPO from stable-baselines3.
        Returns the optimal policy and value function.
        """
        if self.solver is None:
            if PPO is None or self.vec_env is None:
                raise NotImplementedError(
                    "stable-baselines3 not available. Install with `pip install econgym[sb3]` to use find_equilibrium()."
                )
            self.solver = PPO("MlpPolicy", self.vec_env, verbose=0)
        
        # Train the model
        self.solver.learn(total_timesteps=100000)
        
        # Get the policy and value function
        policy = self.solver.policy
        value_fn = self.solver.policy.value_net
        
        return policy, value_fn

    def get_policy(self, state):
        """
        Get the optimal action for a given state using the trained policy.
        """
        if self.solver is None:
            if PPO is None:
                raise NotImplementedError(
                    "stable-baselines3 not available. Install with `pip install econgym[sb3]` to use get_policy()."
                )
            self.find_equilibrium()
        return self.solver.predict(state)[0]

    def get_value_function(self, state):
        """
        Get the value function for a given state using the trained model.
        """
        if self.solver is None:
            if PPO is None or torch is None:
                raise NotImplementedError(
                    "Torch/SB3 not available. Install with `pip install econgym[sb3]` to use get_value_function()."
                )
            self.find_equilibrium()
        # torch is checked above
        return self.solver.policy.value_net(torch.FloatTensor(state))

    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """
        Calibrate the model parameters to match target moments.
        """
        from scipy.optimize import minimize
        
        def objective(params):
            # Update parameters
            self.parameters['revenue_multiplier'] = params[0]
            self.parameters['cost_multiplier'] = params[1]
            self.parameters['state_adjustment']['high_price'] = params[2]
            
            # Reset solver when parameters change
            self.solver = None
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute moments
            avg_price = np.mean([np.mean(state) for state in states])
            avg_profit = np.mean([np.mean(reward) for reward in rewards])
            price_dispersion = np.mean([np.std(state) for state in states])
            
            # Compute squared differences from targets
            error = 0
            if 'avg_price' in targets:
                error += (avg_price - targets['avg_price'])**2
            if 'avg_profit' in targets:
                error += (avg_profit - targets['avg_profit'])**2
            if 'price_dispersion' in targets:
                error += (price_dispersion - targets['price_dispersion'])**2
                
            return error
        
        # Get initial parameters and bounds
        x0 = [
            self.parameters['revenue_multiplier'],
            self.parameters['cost_multiplier'],
            self.parameters['state_adjustment']['high_price']
        ]
        bounds = [
            (5, 20),    # revenue_multiplier
            (0.05, 0.2), # cost_multiplier
            (0.5, 2.0)   # state_adjustment
        ]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['revenue_multiplier'] = result.x[0]
        self.parameters['cost_multiplier'] = result.x[1]
        self.parameters['state_adjustment']['high_price'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'revenue_multiplier': result.x[0],
                'cost_multiplier': result.x[1],
                'state_adjustment_high_price': result.x[2]
            },
            'objective_value': result.fun
        }

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current states: {self.state}")
            print(f"Parameters: revenue_multiplier={self.parameters['revenue_multiplier']:.2f}, "
                  f"cost_multiplier={self.parameters['cost_multiplier']:.2f}, "
                  f"state_adjustment_high_price={self.parameters['state_adjustment']['high_price']:.2f}")
        return None 