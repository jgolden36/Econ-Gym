import numpy as np
from typing import Dict, Any, Tuple, List
from econgym.core.base_solver import BaseSolver
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

class RyanSolver(BaseSolver):
    """
    Solver for the Ryan dynamic entry-exit model.
    Uses Gym's built-in solvers for policy/value approximation.
    Handles calibration, estimation, and equilibrium computation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.model = None
        self.vec_env = DummyVecEnv([lambda: env])
        
    def solve_equilibrium(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the Markov Perfect Equilibrium using PPO.
        """
        # Initialize PPO model if not already done
        if self.model is None:
            self.model = PPO("MlpPolicy", self.vec_env, verbose=0)
        
        # Train the model
        self.model.learn(total_timesteps=100000)
        
        # Get the policy and value function
        policy = self.model.policy
        value_fn = self.model.policy.value_net
        
        # Store equilibrium state
        self.equilibrium_policy = policy
        self.equilibrium_value = value_fn
        self.equilibrium_state = np.array([value_fn, policy], dtype=object)
        
        return policy, value_fn

    def _update_parameters(self, params: np.ndarray) -> None:
        """Update environment parameters."""
        self.env.parameters['beta'] = params[0]
        self.env.parameters['entry_cost'] = params[1]
        self.env.parameters['exit_value'] = params[2]
        self.env.parameters['demand_elasticity'] = params[3]
        
        # Reset model when parameters change
        self.model = None

    def _get_parameter_bounds(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for optimization."""
        x0 = [
            self.env.parameters['beta'],
            self.env.parameters['entry_cost'],
            self.env.parameters['exit_value'],
            self.env.parameters['demand_elasticity']
        ]
        bounds = [
            (0.9, 0.99),    # beta
            (100, 1000),    # entry_cost
            (0, 100),       # exit_value
            (1.0, 3.0)      # demand_elasticity
        ]
        return np.array(x0), bounds

    def _get_parameters_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'beta': params[0],
            'entry_cost': params[1],
            'exit_value': params[2],
            'demand_elasticity': params[3]
        }

    def _compute_moments(self, states: List[np.ndarray], rewards: List[np.ndarray]) -> Dict[str, float]:
        """Compute moments from simulated data."""
        # Convert states and rewards to arrays for easier computation
        states_array = np.array(states)
        rewards_array = np.array(rewards)
        
        # Compute moments
        entry_rate = np.mean([np.mean(state == 1) for state in states_array])
        exit_rate = np.mean([np.mean(state == 0) for state in states_array])
        avg_profit = np.mean([np.mean(reward) for reward in rewards_array])
        
        return {
            'entry_rate': entry_rate,
            'exit_rate': exit_rate,
            'avg_profit': avg_profit
        }

    def _interpolate_value(self, state: np.ndarray) -> np.ndarray:
        """Get value function for given state using the trained model."""
        if self.model is None:
            self.solve_equilibrium()
        return self.model.policy.value_net(torch.FloatTensor(state))

    def _interpolate_policy(self, state: np.ndarray) -> np.ndarray:
        """Get policy for given state using the trained model."""
        if self.model is None:
            self.solve_equilibrium()
        return self.model.predict(state)[0] 