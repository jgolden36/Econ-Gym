import numpy as np
from typing import Dict, Any, Tuple, List
from econgym.core.base_solver import BaseSolver

class BBLSolver(BaseSolver):
    """
    Solver for the BBL dynamic game model.
    Handles equilibrium solving, calibration, and estimation.
    """
    def solve_equilibrium(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the Markov Perfect Equilibrium using value function iteration.
        """
        n_firms = self.env.n_firms
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        
        # Initialize value functions and policies for each firm
        V = np.zeros((n_firms, n_states))
        policy = np.zeros((n_firms, n_states), dtype=int)
        
        for iteration in range(max_iter):
            V_new = np.zeros_like(V)
            policy_new = np.zeros_like(policy)
            
            for firm in range(n_firms):
                for state in range(n_states):
                    # Get current state for all firms
                    current_state = np.zeros(n_firms)
                    current_state[firm] = state
                    
                    # Compute value for each action
                    action_values = np.zeros(n_actions)
                    for action in range(n_actions):
                        # Get next state and reward
                        next_state, reward, _, _, _ = self.env.step(action)
                        action_values[action] = reward + self.env.parameters['beta'] * V[firm, next_state[firm]]
                    
                    # Update value and policy
                    best_action = np.argmax(action_values)
                    V_new[firm, state] = action_values[best_action]
                    policy_new[firm, state] = best_action
            
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
        self.env.parameters['cost_scale'] = params[1]
        self.env.parameters['demand_elasticity'] = params[2]

    def _get_parameter_bounds(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for optimization."""
        x0 = [
            self.env.parameters['beta'],
            self.env.parameters['cost_scale'],
            self.env.parameters['demand_elasticity']
        ]
        bounds = [
            (0.9, 0.99),  # beta
            (0.1, 1.0),   # cost_scale
            (1.0, 3.0)    # demand_elasticity
        ]
        return np.array(x0), bounds

    def _get_parameters_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'beta': params[0],
            'cost_scale': params[1],
            'demand_elasticity': params[2]
        }

    def _compute_moments(self, states: List[np.ndarray], rewards: List[np.ndarray]) -> Dict[str, float]:
        """Compute moments from simulated data."""
        # Convert states and rewards to arrays for easier computation
        states_array = np.array(states)
        rewards_array = np.array(rewards)
        
        # Compute moments
        avg_price = np.mean([np.mean(state) for state in states_array])
        avg_profit = np.mean([np.mean(reward) for reward in rewards_array])
        price_dispersion = np.mean([np.std(state) for state in states_array])
        
        return {
            'avg_price': avg_price,
            'avg_profit': avg_profit,
            'price_dispersion': price_dispersion
        }

    def _interpolate_value(self, state: np.ndarray) -> np.ndarray:
        """Interpolate value function for given state."""
        # For discrete states, just return the value at the state
        return self.equilibrium_value[:, int(state)]

    def _interpolate_policy(self, state: np.ndarray) -> np.ndarray:
        """Interpolate policy for given state."""
        # For discrete states, just return the policy at the state
        return self.equilibrium_policy[:, int(state)] 