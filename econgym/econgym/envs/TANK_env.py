import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
from scipy.optimize import minimize, bisect

class TANKEnv(EconEnv, NumpyRNGMixin):
    """
    Two-Agent New Keynesian (TANK) environment.
    
    This environment models a TANK economy with:
    - Two types of households: savers and borrowers
    - A representative firm that sets prices
    - A central bank that sets interest rates
    - Shocks to productivity and monetary policy
    
    The state space consists of:
    - Capital stock for each agent type
    - Productivity shock
    - Interest rate shock
    
    The action space consists of:
    - Consumption levels for each agent type
    """
    def __init__(self, 
                 beta: float = 0.99,
                 alpha: float = 0.33,
                 delta: float = 0.025,
                 phi_pi: float = 1.5,
                 phi_y: float = 0.5,
                 rho_a: float = 0.9,
                 rho_r: float = 0.8,
                 num_agent_types: int = 2):
        """
        Initialize the TANK environment.
        
        Args:
            beta: Discount factor
            alpha: Capital share in production
            delta: Depreciation rate
            phi_pi: Taylor rule coefficient on inflation
            phi_y: Taylor rule coefficient on output gap
            rho_a: Persistence of productivity shock
            rho_r: Persistence of monetary policy shock
            num_agent_types: Number of agent types (default: 2 for TANK)
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
            'initial_K': 50.0,   # Initial capital
            'lambda_s': 0.7,     # Share of savers
            'lambda_b': 0.3      # Share of borrowers
        }
        
        # Define observation space: (capital, productivity, interest rate) for each agent type
        self.observation_space = spaces.Box(
            low=np.tile(np.array([self.parameters['K_min'], -0.1, -0.1]), (num_agent_types, 1)),
            high=np.tile(np.array([self.parameters['K_max'], 0.1, 0.1]), (num_agent_types, 1)),
            dtype=np.float32
        )
        
        # Define action space: consumption for each agent type
        self.action_space = spaces.Box(
            low=np.zeros(num_agent_types),
            high=np.full(num_agent_types, 100.0),
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
            
        # Initialize capital for each agent type
        K = np.full(self.observation_space.shape[0], self.parameters['initial_K'])
        
        # Initialize productivity and interest rate shocks
        log_A = np.zeros(self.observation_space.shape[0])
        r = np.zeros(self.observation_space.shape[0])
        
        self.state = np.column_stack([K, log_A, r])
        return self.state, {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one period of the economy.
        
        Args:
            actions: Consumption levels for each agent type
            
        Returns:
            Tuple of (next_state, rewards, done, truncated, info)
        """
        # Extract consumption for each agent type
        C = np.clip(actions, self.action_space.low, self.action_space.high)
        
        # Compute production for each agent type
        Y = self.A * (self.state[:, 0] ** self.parameters['alpha'])
        
        # Update capital for each agent type
        K_next = Y - C + (1 - self.parameters['delta']) * self.state[:, 0]
        K_next = np.clip(K_next, self.parameters['K_min'], self.parameters['K_max'])
        
        # Update productivity shock with numerical stability
        log_A_next = (self.parameters['rho_a'] * self.state[:, 1] +
                      self.rng.normal(0, self.parameters['sigma_a'], self.observation_space.shape[0]))
        log_A_next = np.clip(log_A_next, -0.1, 0.1)  # Clip to observation space bounds
        self.A = np.exp(np.mean(log_A_next))
        
        # Update interest rate shock with numerical stability
        r_next = (self.parameters['rho_r'] * self.state[:, 2] +
                  self.rng.normal(0, self.parameters['sigma_r'], self.observation_space.shape[0]))
        r_next = np.clip(r_next, -0.1, 0.1)  # Clip to observation space bounds
        self.r = np.mean(r_next)
        
        # Compute utility for each agent type with numerical stability
        rewards = np.log(np.maximum(C, 1e-10))  # Avoid log(0)
        
        # Update state
        self.state = np.column_stack([K_next, log_A_next, r_next])
        
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
        
        return self.state, rewards, done, truncated, info

    def find_equilibrium(self, tol=1e-4, max_iter=1000):
        """
        Compute the steady state equilibrium of the TANK model.
        
        Returns:
            policy: Steady state consumption policy for each agent type
            V: Steady state value function for each agent type
        """
        # Use coarser grids for faster computation
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 50)  # Reduced from 100
        A_grid = np.linspace(-0.1, 0.1, 10)  # Reduced from 20
        r_grid = np.linspace(-0.1, 0.1, 10)  # Reduced from 20
        
        # Initialize value function and policy for each agent type
        V = np.zeros((self.observation_space.shape[0], len(A_grid), len(r_grid), len(K_grid)))
        policy = np.zeros((self.observation_space.shape[0], len(A_grid), len(r_grid), len(K_grid)))
        
        # Pre-compute production for all states
        A_values = np.exp(A_grid)
        Y_grid = np.zeros((len(A_grid), len(K_grid)))
        for i, A in enumerate(A_values):
            Y_grid[i] = A * (K_grid ** self.parameters['alpha'])
        
        # Pre-compute transition indices
        A_next_indices = np.array([np.argmin(np.abs(A_grid - self.parameters['rho_a'] * a)) for a in A_grid])
        r_next_indices = np.array([np.argmin(np.abs(r_grid - self.parameters['rho_r'] * r)) for r in r_grid])
        
        # Value function iteration with vectorization
        for iteration in range(max_iter):
            V_new = np.copy(V)
            delta = 0
            
            for agent_type in range(self.observation_space.shape[0]):
                # Vectorize over A and r dimensions
                for i, log_A in enumerate(A_grid):
                    for j, r in enumerate(r_grid):
                        # Get production for this state
                        Y = Y_grid[i]
                        
                        # Create consumption grid for this state
                        C_grid = np.linspace(0.1, np.max(Y), 20)  # Reduced from 50
                        
                        # Compute next period capital for all consumption levels
                        K_next = np.maximum(
                            Y[:, np.newaxis] - C_grid + (1 - self.parameters['delta']) * K_grid[:, np.newaxis],
                            self.parameters['K_min']
                        )
                        K_next = np.minimum(K_next, self.parameters['K_max'])
                        
                        # Find next state indices for all K_next values
                        K_next_indices = np.argmin(np.abs(K_grid[:, np.newaxis] - K_next), axis=0)
                        
                        # Get continuation values
                        A_next_idx = A_next_indices[i]
                        r_next_idx = r_next_indices[j]
                        continuation_values = V[agent_type, A_next_idx, r_next_idx, K_next_indices]
                        
                        # Compute current period utility
                        current_utility = np.log(C_grid)
                        
                        # Compute total value for all consumption levels
                        total_values = current_utility + self.parameters['beta'] * continuation_values
                        
                        # Find optimal consumption and value
                        best_idx = np.argmax(total_values, axis=0)
                        best_values = total_values[best_idx, np.arange(len(K_grid))]
                        best_actions = C_grid[best_idx]
                        
                        # Update value function and policy
                        V_new[agent_type, i, j] = best_values
                        policy[agent_type, i, j] = best_actions
                        
                        # Update maximum difference
                        delta = max(delta, np.max(np.abs(V_new[agent_type, i, j] - V[agent_type, i, j])))
            
            # Check convergence
            if delta < tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update value function with relaxation
            V = 0.5 * V + 0.5 * V_new
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}, Max difference: {delta:.6f}")
        
        self.equilibrium_value = V
        self.equilibrium_policy = policy
        self.equilibrium_state = np.array([V, policy], dtype=object)
        return policy, V

    def get_value_function(self, state):
        """
        Get the value function for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [K, log_A, r] for each agent type
            
        Returns:
            The value function for the current state for each agent type
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for each agent type's current state
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        values = np.zeros(self.observation_space.shape[0])
        for agent_type in range(self.observation_space.shape[0]):
            K_idx = np.argmin(np.abs(K_grid - state[agent_type, 0]))
            A_idx = np.argmin(np.abs(A_grid - state[agent_type, 1]))
            r_idx = np.argmin(np.abs(r_grid - state[agent_type, 2]))
            
            values[agent_type] = self.equilibrium_value[agent_type, A_idx, r_idx, K_idx]
        
        return values

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [K, log_A, r] for each agent type
            
        Returns:
            The policy for the current state for each agent type
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for each agent type's current state
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        actions = np.zeros(self.observation_space.shape[0])
        for agent_type in range(self.observation_space.shape[0]):
            K_idx = np.argmin(np.abs(K_grid - state[agent_type, 0]))
            A_idx = np.argmin(np.abs(A_grid - state[agent_type, 1]))
            r_idx = np.argmin(np.abs(r_grid - state[agent_type, 2]))
            
            actions[agent_type] = self.equilibrium_policy[agent_type, A_idx, r_idx, K_idx]
        
        return actions

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
                                 for K in states[:, :, 0].flatten()])
            mean_consumption = np.mean(rewards)
            mean_capital = np.mean(states[:, :, 0])
            
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

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print("Current State:")
            for agent_type in range(self.observation_space.shape[0]):
                print(f"Agent {agent_type}: K={self.state[agent_type, 0]:.2f}, log(A)={self.state[agent_type, 1]:.2f}, r={self.state[agent_type, 2]:.2f}")
            print(f"Parameters: alpha={self.parameters['alpha']:.2f}, "
                  f"delta={self.parameters['delta']:.2f}, "
                  f"phi_pi={self.parameters['phi_pi']:.2f}")
        return None

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
        
        # Plot capital distribution
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        for agent_type in range(self.observation_space.shape[0]):
            axes[0, 0].plot(K_grid, self.equilibrium_policy[agent_type, 0, 0, :], 
                          label=f'Agent {agent_type}')
        axes[0, 0].set_title('Capital Policy Function')
        axes[0, 0].set_xlabel('Capital')
        axes[0, 0].set_ylabel('Consumption')
        axes[0, 0].legend()
        
        # Plot value function
        for agent_type in range(self.observation_space.shape[0]):
            axes[0, 1].plot(K_grid, self.equilibrium_value[agent_type, 0, 0, :],
                          label=f'Agent {agent_type}')
        axes[0, 1].set_title('Value Function')
        axes[0, 1].set_xlabel('Capital')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        
        # Plot productivity response
        A_grid = np.linspace(-0.1, 0.1, 20)
        for agent_type in range(self.observation_space.shape[0]):
            axes[1, 0].plot(A_grid, self.equilibrium_policy[agent_type, :, 0, 50],
                          label=f'Agent {agent_type}')
        axes[1, 0].set_title('Productivity Response')
        axes[1, 0].set_xlabel('Log Productivity')
        axes[1, 0].set_ylabel('Consumption')
        axes[1, 0].legend()
        
        # Plot interest rate response
        r_grid = np.linspace(-0.1, 0.1, 20)
        for agent_type in range(self.observation_space.shape[0]):
            axes[1, 1].plot(r_grid, self.equilibrium_policy[agent_type, 0, :, 50],
                          label=f'Agent {agent_type}')
        axes[1, 1].set_title('Interest Rate Response')
        axes[1, 1].set_xlabel('Interest Rate')
        axes[1, 1].set_ylabel('Consumption')
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _compute_gini_coefficient(self, states: np.ndarray) -> float:
        """
        Compute the Gini coefficient for a given distribution of states.
        
        Args:
            states: Array of states
            
        Returns:
            Gini coefficient
        """
        # Extract capital holdings
        capital = states[:, 0]
        
        # Sort capital holdings
        sorted_capital = np.sort(capital)
        
        # Compute cumulative shares
        n = len(capital)
        index = np.arange(1, n + 1)
        
        # Compute Gini coefficient
        return ((2 * np.sum(index * sorted_capital)) / (n * np.sum(sorted_capital))) - (n + 1) / n

    def find_stationary_distribution(self, policy: Optional[np.ndarray] = None, 
                                   n_periods: int = 1000, 
                                   n_agents: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the stationary distribution of the economy.
        
        Args:
            policy: Optional policy function to use
            n_periods: Number of periods to simulate
            n_agents: Number of agents to simulate
            
        Returns:
            Tuple of (asset_dist, income_dist, joint_dist)
        """
        if policy is None:
            if not hasattr(self, 'equilibrium_policy'):
                self.find_equilibrium()
            policy = self.equilibrium_policy
            
        # Initialize arrays to store distributions
        asset_dist = np.zeros((n_periods, n_agents))
        income_dist = np.zeros((n_periods, n_agents))
        
        # Initialize agents
        for i in range(n_agents):
            state, _ = self.reset()
            asset_dist[0, i] = state[0, 0]  # Capital
            income_dist[0, i] = state[0, 1]  # Productivity
            
        # Simulate forward
        for t in range(1, n_periods):
            for i in range(n_agents):
                # Get current state
                state = np.array([[asset_dist[t-1, i], income_dist[t-1, i], 0]])
                
                # Get action from policy
                K_idx = np.argmin(np.abs(np.linspace(self.parameters['K_min'], 
                                                    self.parameters['K_max'], 100) - state[0, 0]))
                A_idx = np.argmin(np.abs(np.linspace(-0.1, 0.1, 20) - state[0, 1]))
                r_idx = 0  # Use steady state interest rate
                
                action = policy[0, A_idx, r_idx, K_idx]
                
                # Step environment
                next_state, _, _, _, _ = self.step(np.array([action]))
                
                # Update distributions
                asset_dist[t, i] = next_state[0, 0]
                income_dist[t, i] = next_state[0, 1]
                
        # Compute joint distribution
        joint_dist = np.histogram2d(asset_dist[-1], income_dist[-1], 
                                  bins=[50, 20], 
                                  range=[[self.parameters['K_min'], self.parameters['K_max']], 
                                        [-0.1, 0.1]])[0]
        
        return asset_dist[-1], income_dist[-1], joint_dist

    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            print("\nCurrent State:")
            for agent_type in range(self.observation_space.shape[0]):
                print(f"Agent {agent_type}:")
                print(f"  Capital: {self.state[agent_type, 0]:.2f}")
                print(f"  Log Productivity: {self.state[agent_type, 1]:.2f}")
                print(f"  Interest Rate: {self.state[agent_type, 2]:.2f}")
            
            print("\nParameters:")
            print(f"  Alpha: {self.parameters['alpha']:.2f}")
            print(f"  Delta: {self.parameters['delta']:.2f}")
            print(f"  Phi_pi: {self.parameters['phi_pi']:.2f}")
            print(f"  Phi_y: {self.parameters['phi_y']:.2f}")
            print(f"  Rho_a: {self.parameters['rho_a']:.2f}")
            print(f"  Rho_r: {self.parameters['rho_r']:.2f}")