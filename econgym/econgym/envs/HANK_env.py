import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HANKEnv(EconEnv, NumpyRNGMixin):
    """
    Heterogeneous Agent New Keynesian (HANK) environment.
    
    This environment models a HANK economy with:
    - Heterogeneous households that choose consumption and labor
    - A representative firm that sets prices
    - A central bank that sets interest rates
    - Shocks to productivity and monetary policy
    
    The state space consists of:
    - Capital stock for each agent
    - Productivity shock
    - Interest rate shock
    
    The action space consists of:
    - Consumption levels for each agent
    """
    def __init__(self, 
                 num_agents: int = 10,
                 beta: float = 0.99,
                 alpha: float = 0.33,
                 delta: float = 0.025,
                 phi_pi: float = 1.5,
                 phi_y: float = 0.5,
                 rho_a: float = 0.9,
                 rho_r: float = 0.8):
        """
        Initialize the HANK environment.
        
        Args:
            num_agents: Number of heterogeneous agents
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
            'num_agents': num_agents,
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
        
        # Define observation space: (capital, productivity, interest rate) for each agent
        self.observation_space = spaces.Box(
            low=np.tile(np.array([self.parameters['K_min'], -0.1, -0.1]), (self.parameters['num_agents'], 1)),
            high=np.tile(np.array([self.parameters['K_max'], 0.1, 0.1]), (self.parameters['num_agents'], 1)),
            dtype=np.float32
        )
        
        # Define action space: consumption for each agent
        self.action_space = spaces.Box(
            low=np.zeros(self.parameters['num_agents']),
            high=np.full(self.parameters['num_agents'], 100.0),
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
            
        # Initialize capital for each agent
        K = np.full(self.parameters['num_agents'], self.parameters['initial_K'])
        
        # Initialize productivity and interest rate shocks
        log_A = np.zeros(self.parameters['num_agents'])
        r = np.zeros(self.parameters['num_agents'])
        
        self.state = np.column_stack([K, log_A, r])
        return self.state, {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Execute one period of the economy.
        
        Args:
            actions: Consumption levels for each agent
            
        Returns:
            Tuple of (next_state, rewards, done, truncated, info)
        """
        # Extract consumption for each agent
        C = np.clip(actions, self.action_space.low, self.action_space.high)
        
        # Aggregate production with Cobb-Douglas Y = A * K^alpha * L^(1-alpha)
        K_i = self.state[:, 0]
        z_i = np.exp(self.state[:, 1])  # idiosyncratic productivity units, proxy for effective labor
        K_agg = float(np.sum(K_i))
        L_agg = float(np.sum(z_i))
        Y_agg = float(self.A * (K_agg ** self.parameters['alpha']) * (L_agg ** (1.0 - self.parameters['alpha'])))

        # Factor prices
        if K_agg <= 0 or L_agg <= 0:
            r_k = 0.0
            w = 0.0
        else:
            r_k = float(self.parameters['alpha'] * self.A * (K_agg ** (self.parameters['alpha'] - 1.0)) * (L_agg ** (1.0 - self.parameters['alpha'])))
            w = float((1.0 - self.parameters['alpha']) * self.A * (K_agg ** self.parameters['alpha']) * (L_agg ** (-self.parameters['alpha'])))

        # Update capital for each agent: K' = (1-delta)K + r_k*K + w*z - C
        K_next = (1.0 - self.parameters['delta']) * K_i + r_k * K_i + w * z_i - C
        K_next = np.clip(K_next, self.parameters['K_min'], self.parameters['K_max'])
        
        # Update productivity shock with numerical stability
        log_A_next = (self.parameters['rho_a'] * self.state[:, 1] +
                      self.rng.normal(0, self.parameters['sigma_a'], self.parameters['num_agents']))
        log_A_next = np.clip(log_A_next, -0.1, 0.1)  # Clip to observation space bounds
        self.A = np.exp(np.mean(log_A_next))
        
        # Update interest rate shock with numerical stability
        r_next = (self.parameters['rho_r'] * self.state[:, 2] +
                  self.rng.normal(0, self.parameters['sigma_r'], self.parameters['num_agents']))
        r_next = np.clip(r_next, -0.1, 0.1)  # Clip to observation space bounds
        self.r = np.mean(r_next)
        
        # Compute utility for each agent with numerical stability
        rewards = np.log(np.maximum(C, 1e-10))  # Avoid log(0)
        
        # Update state
        self.state = np.column_stack([K_next, log_A_next, r_next])
        
        # Check for termination conditions
        done = False
        truncated = False
        
        # Add additional information
        info = {
            'Y_agg': Y_agg,
            'K_agg': K_agg,
            'L_agg': L_agg,
            'r_k': r_k,
            'w': w,
            'consumption': C,
            'capital': K_next,
            'productivity': self.A,
            'interest_rate': self.r
        }
        
        return self.state, rewards, done, truncated, info

    def simulate(self, n_periods: int = 1000, policy: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the HANK model for a given number of periods.

        If a policy is not provided and an equilibrium policy is available,
        actions are chosen using `get_policy`.

        Returns arrays with shapes:
        - states: (T, num_agents, 3)
        - rewards: (T, num_agents)
        """
        states: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        state, _ = self.reset()

        for _ in range(n_periods):
            if policy is None:
                if hasattr(self, 'equilibrium_policy'):
                    action = self.get_policy(state)
                else:
                    action = self.action_space.sample()
            else:
                action = policy(state)

            state, reward, done, truncated, _ = self.step(action)
            states.append(state.copy())
            rewards.append(reward.copy())
            if done or truncated:
                break

        return np.array(states), np.array(rewards)

    def find_equilibrium(self, tol=1e-4, max_iter=1000):
        """
        Compute the steady state equilibrium of the HANK model.
        
        Returns:
            policy: Steady state consumption policy for each agent
            V: Steady state value function for each agent
        """
        # Discretize state space
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        # Initialize value function and policy for each agent
        V = np.zeros((self.parameters['num_agents'], len(A_grid), len(r_grid), len(K_grid)))
        policy = np.zeros((self.parameters['num_agents'], len(A_grid), len(r_grid), len(K_grid)))
        
        # Value function iteration
        for iteration in range(max_iter):
            V_new = np.copy(V)
            delta = 0
            
            for agent in range(self.parameters['num_agents']):
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
                                
                                value = np.log(C) + self.parameters['beta'] * V[agent, A_idx, r_idx, K_idx]
                                
                                if value > best_value:
                                    best_value = value
                                    best_action = C
                            
                            V_new[agent, i, j, k] = best_value
                            policy[agent, i, j, k] = best_action
                            delta = max(delta, abs(V_new[agent, i, j, k] - V[agent, i, j, k]))
            
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
            state: The current state of the environment [K, log_A, r] for each agent
            
        Returns:
            The value function for the current state for each agent
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for each agent's current state
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        values = np.zeros(self.parameters['num_agents'])
        for agent in range(self.parameters['num_agents']):
            K_idx = np.argmin(np.abs(K_grid - state[agent, 0]))
            A_idx = np.argmin(np.abs(A_grid - state[agent, 1]))
            r_idx = np.argmin(np.abs(r_grid - state[agent, 2]))
            
            values[agent] = self.equilibrium_value[agent, A_idx, r_idx, K_idx]
        
        return values

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [K, log_A, r] for each agent
            
        Returns:
            The policy for the current state for each agent
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for each agent's current state
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        A_grid = np.linspace(-0.1, 0.1, 20)
        r_grid = np.linspace(-0.1, 0.1, 20)
        
        actions = np.zeros(self.parameters['num_agents'])
        for agent in range(self.parameters['num_agents']):
            K_idx = np.argmin(np.abs(K_grid - state[agent, 0]))
            A_idx = np.argmin(np.abs(A_grid - state[agent, 1]))
            r_idx = np.argmin(np.abs(r_grid - state[agent, 2]))
            
            actions[agent] = self.equilibrium_policy[agent, A_idx, r_idx, K_idx]
        
        return actions

    def calibrate(self, targets: Dict[str, Any], method: str = "L-BFGS-B", **kwargs) -> Dict[str, Any]:
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
            # Output Y = A * K^alpha where A = exp(log_A)
            K_series = states[..., 0]
            log_A_series = states[..., 1]
            Y_series = np.exp(log_A_series) * (K_series ** self.parameters['alpha'])
            mean_output = float(np.mean(Y_series))

            # Rewards are log(C); recover average consumption from exp(reward)
            mean_consumption = float(np.mean(np.exp(rewards)))

            mean_capital = float(np.mean(K_series))
            
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
        # Ensure bounds-compatible optimizer by default
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
                weight_matrix: np.ndarray, method: str = "L-BFGS-B", **kwargs) -> Dict[str, Any]:
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
        result = minimize(gmm_objective, x0, method=method, bounds=bounds, **kwargs)
        
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
        
        # Plot capital distribution
        K_grid = np.linspace(self.parameters['K_min'], self.parameters['K_max'], 100)
        for agent in range(self.parameters['num_agents']):
            axes[0, 0].plot(K_grid, self.equilibrium_policy[agent, 0, 0, :], 
                          label=f'Agent {agent}')
        axes[0, 0].set_title('Capital Policy Function')
        axes[0, 0].set_xlabel('Capital')
        axes[0, 0].set_ylabel('Consumption')
        axes[0, 0].legend()
        
        # Plot value function
        for agent in range(self.parameters['num_agents']):
            axes[0, 1].plot(K_grid, self.equilibrium_value[agent, 0, 0, :],
                          label=f'Agent {agent}')
        axes[0, 1].set_title('Value Function')
        axes[0, 1].set_xlabel('Capital')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        
        # Plot productivity response
        A_grid = np.linspace(-0.1, 0.1, 20)
        for agent in range(self.parameters['num_agents']):
            axes[1, 0].plot(A_grid, self.equilibrium_policy[agent, :, 0, 50],
                          label=f'Agent {agent}')
        axes[1, 0].set_title('Productivity Response')
        axes[1, 0].set_xlabel('Log Productivity')
        axes[1, 0].set_ylabel('Consumption')
        axes[1, 0].legend()
        
        # Plot interest rate response
        r_grid = np.linspace(-0.1, 0.1, 20)
        for agent in range(self.parameters['num_agents']):
            axes[1, 1].plot(r_grid, self.equilibrium_policy[agent, 0, :, 50],
                          label=f'Agent {agent}')
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
                                   n_periods: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the stationary distribution of the economy.
        
        Args:
            policy: Optional policy function to use
            n_periods: Number of periods to simulate
            n_agents: Number of agents to simulate
            
        Returns:
            Tuple of (asset_dist, income_dist, joint_dist) for the final period across agents
        """
        if policy is None and not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()

        # Initialize with current environment size
        num_agents = self.parameters['num_agents']
        asset_dist = np.zeros((n_periods, num_agents))
        income_dist = np.zeros((n_periods, num_agents))

        # Reset environment once with all agents
        state, _ = self.reset()
        asset_dist[0, :] = state[:, 0]
        income_dist[0, :] = state[:, 1]

        # Simulate forward using the environment dynamics
        for t in range(1, n_periods):
            actions = self.get_policy(self.state) if policy is None else policy
            next_state, _, _, _, _ = self.step(actions)
            asset_dist[t, :] = next_state[:, 0]
            income_dist[t, :] = next_state[:, 1]

        # Compute joint distribution at final period
        joint_dist = np.histogram2d(
            asset_dist[-1], income_dist[-1],
            bins=[50, 20],
            range=[[self.parameters['K_min'], self.parameters['K_max']], [-0.1, 0.1]]
        )[0]

        return asset_dist[-1], income_dist[-1], joint_dist

    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            print("\nCurrent State:")
            for agent in range(self.parameters['num_agents']):
                print(f"Agent {agent}:")
                print(f"  Capital: {self.state[agent, 0]:.2f}")
                print(f"  Log Productivity: {self.state[agent, 1]:.2f}")
                print(f"  Interest Rate: {self.state[agent, 2]:.2f}")
            
            print("\nParameters:")
            print(f"  Alpha: {self.parameters['alpha']:.2f}")
            print(f"  Delta: {self.parameters['delta']:.2f}")
            print(f"  Phi_pi: {self.parameters['phi_pi']:.2f}")
            print(f"  Phi_y: {self.parameters['phi_y']:.2f}")
            print(f"  Rho_a: {self.parameters['rho_a']:.2f}")
            print(f"  Rho_r: {self.parameters['rho_r']:.2f}")

# Example usage:
if __name__ == "__main__":
    env = HANKEnv(num_agents=5)
    state, _ = env.reset()
    print("Initial state:")
    print(state)

    # Sample consumption actions for each agent.
    actions = env.action_space.sample()
    next_state, rewards, done, truncated, _ = env.step(actions)
    print("\nNext state:")
    print(next_state)
    print("Rewards:")
    print(rewards)

    # Compute the equilibrium (optimal policy).
    # Warning: with default grid sizes this can be slow/heavy.
    # Consider reducing grid sizes in find_equilibrium for quick tests.
    # policy, V = env.find_equilibrium()
    # Example: simulate for a short horizon using random policy
    sim_states, sim_rewards = env.simulate(n_periods=10)
    print("\nSimulated states shape:", sim_states.shape)
    print("Simulated rewards shape:", sim_rewards.shape)