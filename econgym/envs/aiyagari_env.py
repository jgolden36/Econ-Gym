import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import bisect
from econgym.core.base_env import EconEnv
from typing import Dict, Any, Optional, Tuple


class AiyagariEnv(EconEnv):
    """
    A toy Aiyagari environment with a partial equilibrium interest rate r.
    """
    def __init__(self, r=0.02, a_max=20.0, shock_vals=[0.5, 1.5], shock_probs=[0.5, 0.5]):
        super().__init__()
        # Initialize parameters
        self.parameters = {
            'r': r,
            'a_max': a_max,
            'shock_vals': np.array(shock_vals),
            'shock_probs': np.array(shock_probs),
            'gamma': 2.0,  # CRRA risk aversion parameter
            'saving_fraction': 0.8,  # Default saving fraction for equilibrium
            'alpha': 0.36,  # Capital share parameter
            'beta': 0.95  # Discount factor
        }
        
        # Define action space: agent chooses next period's asset level.
        self.action_space = spaces.Box(low=0.0, high=self.parameters['a_max'], shape=(1,), dtype=np.float32)
        
        # Define observation space: (current assets, current shock)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=self.parameters['a_max'], 
            shape=(2,), 
            dtype=np.float32
        )
        
        self._state = None  # will hold the current state: [assets, shock]
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed
            options: Optional configuration
            
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        # Initialize state: for instance, start with 10 assets and a random shock.
        shock = np.random.choice(self.parameters['shock_vals'], p=self.parameters['shock_probs'])
        self._state = np.array([10.0, shock], dtype=np.float32)
        return self._state, {}  # Gymnasium requires returning a tuple of (observation, info)

    def step(self, action):
        """
        Executes one time step in the environment.
        The action is interpreted as the next period's asset level.
        """
        current_assets, current_shock = self._state
        
        # For illustration, assume income equals the shock value.
        income = current_shock
        
        # Compute available resources: current assets grow with interest plus income.
        available_resources = current_assets * (1 + self.parameters['r']) + income
        
        # Assume action represents the desired asset holdings for next period.
        next_assets = np.clip(action, 0.0, self.parameters['a_max'])[0]
        
        # Calculate consumption as the difference.
        consumption = available_resources - next_assets
        # Ensure consumption is positive to avoid math errors.
        consumption = max(consumption, 1e-8)
        
        # Compute utility using CRRA
        reward = (consumption**(1 - self.parameters['gamma'])) / (1 - self.parameters['gamma'])
        
        # Draw next period's shock.
        next_shock = np.random.choice(self.parameters['shock_vals'], p=self.parameters['shock_probs'])
        self._state = np.array([next_assets, next_shock], dtype=np.float32)
        
        done = False  # this toy model runs indefinitely
        truncated = False  # no truncation in this environment
        info = {}  # no additional info to return
        return self._state, reward, done, truncated, info

    def find_equilibrium(self, tol=1e-4, max_iter=1000):
        """
        Compute the stationary equilibrium for the Aiyagari model.
        This involves finding the interest rate that clears the capital market.
        """
        def excess_demand(r):
            self.parameters['r'] = r
            # Solve the household's problem
            V, policy = self._solve_household_problem()
            # Simulate to get the stationary distribution
            assets = self._simulate_stationary_distribution(policy)
            # Compute aggregate capital supply
            K_supply = np.mean(assets)
            # Compute aggregate capital demand (simplified)
            K_demand = (self.parameters['r'] / self.parameters['alpha'])**(1/(self.parameters['alpha']-1))
            return K_supply - K_demand

        # Find the interest rate that clears the market
        r_eq = bisect(excess_demand, 0.01, 0.1, rtol=tol)
        self.parameters['r'] = r_eq
        
        # Solve for the equilibrium value function and policy
        V, policy = self._solve_household_problem()
        
        self.equilibrium_value = V
        self.equilibrium_policy = policy
        self.equilibrium_state = np.array([V, policy], dtype=object)
        return policy, V

    def _solve_household_problem(self):
        """
        Solve the household's dynamic programming problem.
        """
        # Discretize the asset space
        a_grid = np.linspace(0, self.parameters['a_max'], 100)
        
        # Initialize value function and policy
        V = np.zeros((len(self.parameters['shock_vals']), len(a_grid)))
        policy = np.zeros((len(self.parameters['shock_vals']), len(a_grid)))
        
        # Value function iteration
        for iteration in range(1000):
            V_new = np.copy(V)
            for i, shock in enumerate(self.parameters['shock_vals']):
                for j, a in enumerate(a_grid):
                    # Compute income
                    income = shock
                    
                    # Compute available resources
                    resources = a * (1 + self.parameters['r']) + income
                    
                    # Find optimal next period assets
                    best_value = -np.inf
                    best_action = 0
                    
                    for next_a in a_grid:
                        if next_a <= resources:
                            consumption = resources - next_a
                            if consumption > 0:
                                utility = (consumption**(1 - self.parameters['gamma'])) / (1 - self.parameters['gamma'])
                                
                                # Expected continuation value
                                cont_val = 0
                                for next_shock_idx in range(len(self.parameters['shock_vals'])):
                                    cont_val += self.parameters['shock_probs'][next_shock_idx] * V[next_shock_idx, np.argmin(np.abs(a_grid - next_a))]
                                
                                value = utility + self.parameters['beta'] * cont_val
                                
                                if value > best_value:
                                    best_value = value
                                    best_action = next_a
                    
                    V_new[i, j] = best_value
                    policy[i, j] = best_action
            
            # Check for convergence
            if np.max(np.abs(V_new - V)) < 1e-4:
                break
                
            V = V_new
        
        return V, policy

    def _simulate_stationary_distribution(self, policy):
        """
        Simulate the model to find the stationary distribution of assets.
        """
        a_grid = np.linspace(0, self.parameters['a_max'], 100)
        assets = []
        
        # Start with initial assets
        current_a = 10.0  # Initial assets
        current_shock = np.random.choice(self.parameters['shock_vals'], p=self.parameters['shock_probs'])
        
        # Simulate for many periods
        for _ in range(1000):
            # Find the closest grid point
            a_idx = np.argmin(np.abs(a_grid - current_a))
            shock_idx = np.where(self.parameters['shock_vals'] == current_shock)[0][0]
            
            # Get next period's assets
            next_a = policy[shock_idx, a_idx]
            
            # Draw next period's shock
            next_shock = np.random.choice(self.parameters['shock_vals'], p=self.parameters['shock_probs'])
            
            # Update state
            current_a = next_a
            current_shock = next_shock
            
            # Store assets
            assets.append(current_a)
        
        return np.array(assets)

    def get_value_function(self, state):
        """
        Get the value function for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [assets, shock]
            
        Returns:
            The value function for the current state
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for the current state
        a_grid = np.linspace(0, self.parameters['a_max'], 100)
        a_idx = np.argmin(np.abs(a_grid - state[0]))
        shock_idx = np.where(self.parameters['shock_vals'] == state[1])[0][0]
        
        return self.equilibrium_value[shock_idx, a_idx]

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [assets, shock]
            
        Returns:
            The policy for the current state
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for the current state
        a_grid = np.linspace(0, self.parameters['a_max'], 100)
        a_idx = np.argmin(np.abs(a_grid - state[0]))
        shock_idx = np.where(self.parameters['shock_vals'] == state[1])[0][0]
        
        return self.equilibrium_policy[shock_idx, a_idx]

    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """
        Calibrate the model parameters to match target moments.
        
        Args:
            targets: Dictionary of target moments to match
            method: Optimization method to use
            **kwargs: Additional arguments for the optimizer
            
        Returns:
            Dictionary containing calibration results
        """
        from scipy.optimize import minimize
        
        def objective(params):
            # Update parameters
            self.parameters['r'] = params[0]
            self.parameters['saving_fraction'] = params[1]
            
            # Simulate the model
            states, _ = self.simulate(n_periods=1000)
            
            # Compute moments
            mean_assets = np.mean(states[:, 0])
            std_assets = np.std(states[:, 0])
            
            # Compute squared differences from targets
            error = 0
            if 'mean_assets' in targets:
                error += (mean_assets - targets['mean_assets'])**2
            if 'std_assets' in targets:
                error += (std_assets - targets['std_assets'])**2
                
            return error
        
        # Initial guess and bounds
        x0 = [self.parameters['r'], self.parameters['saving_fraction']]
        bounds = [(0.001, 0.05), (0.1, 0.9)]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['r'] = result.x[0]
        self.parameters['saving_fraction'] = result.x[1]
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'r': result.x[0],
                'saving_fraction': result.x[1]
            },
            'objective_value': result.fun
        }

    def estimate(self, data: np.ndarray, moment_function: callable, 
                weight_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate model parameters using GMM.
        
        Args:
            data: Empirical data
            moment_function: Function that computes moments
            weight_matrix: Weighting matrix for GMM
            **kwargs: Additional arguments for estimation
            
        Returns:
            Dictionary containing estimation results
        """
        from scipy.optimize import minimize
        
        def gmm_objective(params):
            # Update parameters
            self.parameters['r'] = params[0]
            self.parameters['saving_fraction'] = params[1]
            
            # Simulate the model
            states, _ = self.simulate(n_periods=1000)
            
            # Compute simulated moments
            sim_moments = moment_function(self, states)
            
            # Compute GMM objective
            diff = sim_moments - data
            return diff @ weight_matrix @ diff
        
        # Initial guess and bounds
        x0 = [self.parameters['r'], self.parameters['saving_fraction']]
        bounds = [(0.001, 0.05), (0.1, 0.9)]
        
        # Optimize
        result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['r'] = result.x[0]
        self.parameters['saving_fraction'] = result.x[1]
        
        return {
            'success': result.success,
            'message': result.message,
            'estimated_parameters': {
                'r': result.x[0],
                'saving_fraction': result.x[1]
            },
            'objective_value': result.fun
        }

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        """
        if mode == "human":
            print(f"Current State: Assets={self._state[0]:.2f}, Shock={self._state[1]:.2f}")
            print(f"Parameters: r={self.parameters['r']:.4f}, saving_fraction={self.parameters['saving_fraction']:.2f}")
        return None


# Example usage:
if __name__ == "__main__":
    env = AiyagariEnv()
    state = env.reset()
    print("Initial state:", state)
    
    # Take a sample action.
    next_state, reward, done, info = env.step(np.array([12.0], dtype=np.float32))
    print("Next state:", next_state, "Reward:", reward)
    
    # Find the equilibrium interest rate.
    eq_policy, eq_value = env.find_equilibrium()
    print("Equilibrium policy:", eq_policy)
    print("Equilibrium value function:", eq_value)
    
    # Example calibration
    targets = {'mean_assets': 10.0, 'std_assets': 2.0}
    calibration_results = env.calibrate(targets)
    print("Calibration results:", calibration_results)
    
    # Example estimation
    data = np.array([10.0, 2.0])  # Example empirical moments
    moment_function = lambda env, states: np.array([np.mean(states[:, 0]), np.std(states[:, 0])])
    weight_matrix = np.eye(2)
    estimation_results = env.estimate(data, moment_function, weight_matrix)
    print("Estimation results:", estimation_results)