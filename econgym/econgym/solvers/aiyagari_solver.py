import numpy as np
from typing import Dict, Any, Tuple, List
from econgym.core.base_solver import BaseSolver

class AiyagariSolver(BaseSolver):
    """
    Solver for the Aiyagari incomplete markets model.
    Handles equilibrium solving, calibration, and estimation.
    """
    def solve_equilibrium(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the stationary equilibrium using value function iteration.
        """
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        
        # Initialize value function and policy
        V = np.zeros(n_states)
        policy = np.zeros(n_states, dtype=int)
        
        for iteration in range(max_iter):
            V_new = np.zeros_like(V)
            policy_new = np.zeros_like(policy)
            
            for state in range(n_states):
                # Get current state
                current_state = np.array([state])
                
                # Compute value for each action
                action_values = np.zeros(n_actions)
                for action in range(n_actions):
                    # Get next state and reward
                    next_state, reward, _, _, _ = self.env.step(action)
                    action_values[action] = reward + self.env.parameters['beta'] * V[next_state[0]]
                
                # Update value and policy
                best_action = np.argmax(action_values)
                V_new[state] = action_values[best_action]
                policy_new[state] = best_action
            
            # Check convergence
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                policy = policy_new
                break
                
            V = V_new
            policy = policy_new
        
        self.equilibrium_value = V
        self.equilibrium_policy = policy
        self.equilibrium_state = np.array([V, policy], dtype=object)
        return policy, V

    def _update_parameters(self, params: np.ndarray) -> None:
        """Update environment parameters."""
        self.env.parameters['beta'] = params[0]
        self.env.parameters['risk_aversion'] = params[1]
        self.env.parameters['income_persistence'] = params[2]

    def _get_parameter_bounds(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for optimization."""
        x0 = [
            self.env.parameters['beta'],
            self.env.parameters['risk_aversion'],
            self.env.parameters['income_persistence']
        ]
        bounds = [
            (0.9, 0.99),  # beta
            (1.0, 5.0),   # risk_aversion
            (0.5, 0.95)   # income_persistence
        ]
        return np.array(x0), bounds

    def _get_parameters_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'beta': params[0],
            'risk_aversion': params[1],
            'income_persistence': params[2]
        }

    def _compute_moments(self, states: List[np.ndarray], rewards: List[np.ndarray]) -> Dict[str, float]:
        """Compute moments from simulated data."""
        # Convert states and rewards to arrays for easier computation
        states_array = np.array(states)
        rewards_array = np.array(rewards)
        
        # Compute moments
        avg_consumption = np.mean([np.mean(reward) for reward in rewards_array])
        consumption_volatility = np.std([np.mean(reward) for reward in rewards_array])
        avg_assets = np.mean([np.mean(state) for state in states_array])
        
        return {
            'avg_consumption': avg_consumption,
            'consumption_volatility': consumption_volatility,
            'avg_assets': avg_assets
        }

    def _interpolate_value(self, state: np.ndarray) -> np.ndarray:
        """Interpolate value function for given state."""
        # For discrete states, just return the value at the state
        return self.equilibrium_value[int(state)]

    def _interpolate_policy(self, state: np.ndarray) -> np.ndarray:
        """Interpolate policy for given state."""
        # For discrete states, just return the policy at the state
        return self.equilibrium_policy[int(state)] 