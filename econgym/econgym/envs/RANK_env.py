import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
from scipy.optimize import minimize, bisect

class RANKEnv(EconEnv, NumpyRNGMixin):
    """
    Representative Agent New Keynesian (RANK) environment.
    
    This environment models a simple RANK economy with:
    - A representative household that chooses consumption and labor
    - A representative firm that sets prices
    - A central bank that sets interest rates
    - Shocks to productivity and monetary policy
    
    The state space consists of:
    - Capital stock
    - Productivity shock
    - Interest rate shock
    
    The action space consists of:
    - Consumption level
    """
    def __init__(self, 
                 beta: float = 0.99,
                 alpha: float = 0.33,
                 delta: float = 0.025,
                 phi_pi: float = 1.5,
                 phi_y: float = 0.5,
                 rho_a: float = 0.9,
                 rho_r: float = 0.8):
        """
        Initialize the RANK environment.
        
        Args:
            beta: Discount factor
            alpha: Capital share in production
            delta: Depreciation rate
            phi_pi: Taylor rule coefficient on inflation
            phi_y: Taylor rule coefficient on output gap
            rho_a: Persistence of productivity shock
            rho_r: Persistence of monetary policy shock
        """
        super().__init__()
        # Initialize parameters
        self.parameters = {
            'beta': beta,        # Discount factor
            'alpha': alpha,      # Capital share
            'delta': delta,      # Depreciation rate
            'phi_pi': phi_pi,    # Taylor rule coefficient on inflation
            'phi_y': phi_y,      # Taylor rule coefficient on output gap
            'rho_a': rho_a,      # Persistence of productivity shock
            'rho_r': rho_r,      # Persistence of monetary policy shock
            'sigma_a': 0.01,     # Standard deviation of productivity shock
            'sigma_r': 0.005,    # Standard deviation of monetary policy shock
            'K_min': 0.1,        # Minimum capital
            'K_max': 100.0,      # Maximum capital
            'initial_K': 50.0    # Initial capital
        }
        
        # Define observation space: (capital, productivity, interest rate)
        self.observation_space = spaces.Box(
            low=np.array([self.parameters['K_min'], -0.1, -0.1]),
            high=np.array([self.parameters['K_max'], 0.1, 0.1]),
            dtype=np.float32
        )
        
        # Define action space: consumption
        self.action_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        
        # Initialize shocks
        self.A = 1.0  # Productivity level
        self.r = 0.0  # Interest rate deviation

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        self.reseed(seed)
            
        self.state = np.array([
            self.parameters['initial_K'],
            np.log(self.A),
            self.r
        ], dtype=np.float32)
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one period of the economy.
        
        Args:
            action: Consumption level
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        K, log_A, r = self.state
        
        # Extract consumption with numerical stability
        C = np.clip(float(action[0]), self.action_space.low[0], self.action_space.high[0])
        
        # Compute production
        Y = self.A * (K ** self.parameters['alpha'])
        
        # Update capital
        K_next = Y - C + (1 - self.parameters['delta']) * K
        K_next = np.clip(K_next, self.parameters['K_min'], self.parameters['K_max'])
        
        # Update productivity shock with numerical stability
        log_A_next = self.parameters['rho_a'] * log_A + self.rng.normal(0, self.parameters['sigma_a'])
        log_A_next = np.clip(log_A_next, -0.1, 0.1)  # Clip to observation space bounds
        self.A = np.exp(log_A_next)
        
        # Update interest rate shock with numerical stability
        r_next = self.parameters['rho_r'] * r + self.rng.normal(0, self.parameters['sigma_r'])
        r_next = np.clip(r_next, -0.1, 0.1)  # Clip to observation space bounds
        self.r = r_next
        
        # Compute utility with numerical stability
        reward = np.log(np.maximum(C, 1e-10))  # Avoid log(0)
        
        # Update state
        self.state = np.array([K_next, log_A_next, r_next], dtype=np.float32)
        
        # Check for termination conditions
        done = False
        truncated = False
        
        # Add additional information
        info = {
            'production': Y,
            'consumption': C,
            'capital': K_next,
            'productivity': self.A,
            'interest_rate': self.r
        }
        
        return self.state, reward, done, truncated, info

    def find_equilibrium(self, tol=1e-4, max_iter=1000):
        """
        Compute the steady state equilibrium of the RANK model.
        
        Returns:
            policy: Steady state consumption policy
            V: Steady state value function
        """
        # Discretize state space
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        # Initialize value function and policy
        V = np.zeros((len(A_grid), len(r_grid), len(K_grid)))
        policy = np.zeros((len(A_grid), len(r_grid), len(K_grid)))
        
        # Value function iteration
        for iteration in range(max_iter):
            V_new = np.copy(V)
            delta = 0
            
            for i, log_A in enumerate(A_grid):
                for j, r in enumerate(r_grid):
                    for k, K in enumerate(K_grid):
                        A = np.exp(log_A)
                        Y = A * (K ** self.parameters['alpha'])
                        
                        # Find optimal consumption
                        best_value = -np.inf
                        best_action = 0
                        
                        for C in np.linspace(0.1, Y, 50):
                            K_next = Y - C + (1 - self.parameters['delta']) * K
                            K_next = np.clip(K_next, self.parameters['K_min'], self.parameters['K_max'])
                            
                            # Find next state indices
                            K_idx = np.argmin(np.abs(K_grid - K_next))
                            A_idx = np.argmin(np.abs(A_grid - self.parameters['rho_a'] * log_A))
                            r_idx = np.argmin(np.abs(r_grid - self.parameters['rho_r'] * r))
                            
                            value = np.log(C) + self.parameters['beta'] * V[A_idx, r_idx, K_idx]
                            
                            if value > best_value:
                                best_value = value
                                best_action = C
                        
                        V_new[i, j, k] = best_value
                        policy[i, j, k] = best_action
                        delta = max(delta, abs(V_new[i, j, k] - V[i, j, k]))
            
            V = V_new
            if delta < tol:
                break
        
        self.equilibrium_value = V
        self.equilibrium_policy = policy
        self.equilibrium_state = np.array([V, policy], dtype=object)
        return policy, V

    def get_value_function(self, state):
        """
        Get the value function for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [K, log_A, r]
            
        Returns:
            The value function for the current state
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for the current state
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        K_idx = np.argmin(np.abs(K_grid - state[0]))
        A_idx = np.argmin(np.abs(A_grid - state[1]))
        r_idx = np.argmin(np.abs(r_grid - state[2]))
        
        return self.equilibrium_value[A_idx, r_idx, K_idx]

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [K, log_A, r]
            
        Returns:
            The policy for the current state
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for the current state
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        K_idx = np.argmin(np.abs(K_grid - state[0]))
        A_idx = np.argmin(np.abs(A_grid - state[1]))
        r_idx = np.argmin(np.abs(r_grid - state[2]))
        
        return self.equilibrium_policy[A_idx, r_idx, K_idx]

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
            self.parameters['alpha'] = params[0]
            self.parameters['delta'] = params[1]
            self.parameters['phi_pi'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute moments
            mean_output = np.mean([self.parameters['A'] * (K ** self.parameters['alpha']) 
                                 for K in states[:, 0]])
            mean_consumption = np.mean(rewards)
            mean_capital = np.mean(states[:, 0])
            
            # Compute squared differences from targets
            error = 0
            if 'mean_output' in targets:
                error += (mean_output - targets['mean_output'])**2
            if 'mean_consumption' in targets:
                error += (mean_consumption - targets['mean_consumption'])**2
            if 'mean_capital' in targets:
                error += (mean_capital - targets['mean_capital'])**2
                
            return error
        
        # Initial guess and bounds
        x0 = [
            self.parameters['alpha'],
            self.parameters['delta'],
            self.parameters['phi_pi']
        ]
        bounds = [
            (0.2, 0.4),    # alpha
            (0.01, 0.05),  # delta
            (1.0, 2.0)     # phi_pi
        ]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['alpha'] = result.x[0]
        self.parameters['delta'] = result.x[1]
        self.parameters['phi_pi'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'alpha': result.x[0],
                'delta': result.x[1],
                'phi_pi': result.x[2]
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
            self.parameters['alpha'] = params[0]
            self.parameters['delta'] = params[1]
            self.parameters['phi_pi'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute simulated moments
            sim_moments = moment_function(self, states, rewards)
            
            # Compute GMM objective
            diff = sim_moments - data
            return diff @ weight_matrix @ diff
        
        # Initial guess and bounds
        x0 = [
            self.parameters['alpha'],
            self.parameters['delta'],
            self.parameters['phi_pi']
        ]
        bounds = [
            (0.2, 0.4),    # alpha
            (0.01, 0.05),  # delta
            (1.0, 2.0)     # phi_pi
        ]
        
        # Optimize
        result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['alpha'] = result.x[0]
        self.parameters['delta'] = result.x[1]
        self.parameters['phi_pi'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'estimated_parameters': {
                'alpha': result.x[0],
                'delta': result.x[1],
                'phi_pi': result.x[2]
            },
            'objective_value': result.fun
        }

    def plot_stationary_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot the stationary distribution of the economy.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not hasattr(self, 'equilibrium_state'):
            self.find_equilibrium()
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot capital policy function
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        axes[0, 0].plot(K_grid, self.equilibrium_policy[0, 0, :])
        axes[0, 0].set_title('Capital Policy Function')
        axes[0, 0].set_xlabel('Capital')
        axes[0, 0].set_ylabel('Consumption')
        
        # Plot value function
        axes[0, 1].plot(K_grid, self.equilibrium_value[0, 0, :])
        axes[0, 1].set_title('Value Function')
        axes[0, 1].set_xlabel('Capital')
        axes[0, 1].set_ylabel('Value')
        
        # Plot productivity response
        A_grid = np.linspace(-0.1, 0.1, 20)
        axes[1, 0].plot(A_grid, self.equilibrium_policy[:, 0, 50])
        axes[1, 0].set_title('Productivity Response')
        axes[1, 0].set_xlabel('Log Productivity')
        axes[1, 0].set_ylabel('Consumption')
        
        # Plot interest rate response
        r_grid = np.linspace(-0.1, 0.1, 20)
        axes[1, 1].plot(r_grid, self.equilibrium_policy[0, :, 50])
        axes[1, 1].set_title('Interest Rate Response')
        axes[1, 1].set_xlabel('Interest Rate')
        axes[1, 1].set_ylabel('Consumption')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def find_stationary_distribution(self, policy: Optional[np.ndarray] = None, 
                                   n_periods: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the stationary distribution of the economy.
        
        Args:
            policy: Optional policy function to use
            n_periods: Number of periods to simulate
            
        Returns:
            Tuple of (asset_dist, income_dist)
        """
        if policy is None:
            if not hasattr(self, 'equilibrium_policy'):
                self.find_equilibrium()
            policy = self.equilibrium_policy
            
        # Initialize arrays to store distributions
        asset_dist = np.zeros(n_periods)
        income_dist = np.zeros(n_periods)
        
        # Initialize state
        state, _ = self.reset()
        asset_dist[0] = state[0]  # Capital
        income_dist[0] = state[1]  # Productivity
        
        # Simulate forward
        for t in range(1, n_periods):
            # Get current state
            state = np.array([asset_dist[t-1], income_dist[t-1], 0])
            
            # Get action from policy
            K_idx = np.argmin(np.abs(np.linspace(self.parameters['K_min'], 
                                                self.parameters['K_max'], 100) - state[0]))
            A_idx = np.argmin(np.abs(np.linspace(-0.1, 0.1, 20) - state[1]))
            r_idx = 0  # Use steady state interest rate
            
            action = policy[A_idx, r_idx, K_idx]
            
            # Step environment
            next_state, _, _, _, _ = self.step(np.array([action]))
            
            # Update distributions
            asset_dist[t] = next_state[0]
            income_dist[t] = next_state[1]
        
        return asset_dist, income_dist

    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            print("\nCurrent State:")
            print(f"  Capital: {self.state[0]:.2f}")
            print(f"  Log Productivity: {self.state[1]:.2f}")
            print(f"  Interest Rate: {self.state[2]:.2f}")
            
            print("\nParameters:")
            print(f"  Alpha: {self.parameters['alpha']:.2f}")
            print(f"  Delta: {self.parameters['delta']:.2f}")
            print(f"  Phi_pi: {self.parameters['phi_pi']:.2f}")
            print(f"  Phi_y: {self.parameters['phi_y']:.2f}")
            print(f"  Rho_a: {self.parameters['rho_a']:.2f}")
            print(f"  Rho_r: {self.parameters['rho_r']:.2f}")

# Example usage:
if __name__ == "__main__":
    env = RANKEnv()
    state = env.reset()
    print("Initial state:", state)
    
    # Take a sample action (sample a consumption level from the action space).
    action = env.action_space.sample()
    next_state, reward, done, truncated, _ = env.step(action)
    print("Next state:", next_state, "Reward:", reward)
    
    # Compute an equilibrium (optimal policy) via value function iteration.
    policy, V = env.find_equilibrium()
    print("\nOptimal policy for a few discretized states:")
    # For each productivity shock value, show the optimal consumption for a few capital levels.
    K_grid = np.linspace(env.parameters['K_min'], env.parameters['K_max'], 100)
    for i, A in enumerate(env.parameters['A_vals']):
        print(f"Productivity Shock A = {A:.2f}:")
        for K in np.linspace(env.parameters['K_min'], env.parameters['K_max'], 5):
            K_idx = np.argmin(np.abs(K_grid - K))
            print(f"  Capital {K:.2f}: Optimal consumption = {policy[i, K_idx]:.2f}")