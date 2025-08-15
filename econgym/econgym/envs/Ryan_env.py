import numpy as np
import gymnasium as gym
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


class RyanEnv(EconEnv, NumpyRNGMixin):
    """
    A simplified environment inspired by Ryan (2012), 
    "The Cost of Environment Regulation in a Concentrated Industry."

    This environment models a concentrated industry with n_firms competing in production.
    Each firm faces a stochastic production cost (e.g., a cost shock) and an environmental 
    regulation cost (e.g., an emissions tax per unit). Firms choose their production levels 
    from a discrete set.

    Market demand is given by the inverse demand function:
         P = A - B*(total production)
    and each firm's profit is:
         profit = production * P - production * cost - tau * production

    State:
         For each firm, the state is its current production cost shock,
         which can take one of two values (e.g., 1.0 for low cost and 2.0 for high cost).
    Action:
         For each firm, the action is its production level, chosen from {0, 1, 2}.
    """
    def __init__(self, n_firms=3, A=10.0, B=0.5, tau=1.0, beta=0.95):
        super().__init__()
        # Initialize parameters
        self.parameters = {
            'n_firms': n_firms,
            'A': A,          # Demand intercept
            'B': B,          # Demand slope
            'tau': tau,      # Environmental regulation cost per unit
            'beta': beta,    # Discount factor for equilibrium computation
            'cost_states': np.array([1.0, 2.0]),  # Possible cost states
            'cost_trans': np.array([[0.8, 0.2],   # Transition probabilities
                                   [0.3, 0.7]])
        }
        
        # The observation for each firm is its current cost shock.
        # Joint state: an array of shape (n_firms,)
        self.observation_space = spaces.Box(
            low=np.min(self.parameters['cost_states']),
            high=np.max(self.parameters['cost_states']),
            shape=(self.parameters['n_firms'],),
            dtype=np.float32
        )

        # Each firm chooses its production level from the discrete set {0, 1, 2}.
        self.action_space = spaces.MultiDiscrete([3] * self.parameters['n_firms'])

        # Initialize state: draw a cost shock for each firm.
        self.rng = np.random.default_rng()
        self.state = self.rng.choice(self.parameters['cost_states'], size=self.parameters['n_firms'])

    def reset(self, seed=None, options=None):
        """Reset the environment state by drawing cost shocks for each firm."""
        self.reseed(seed)
        self.state = self.rng.choice(self.parameters['cost_states'], size=self.parameters['n_firms'])
        return self.state, {}

    def step(self, actions):
        """
        Execute one period:
          - Firms choose production levels (actions).
          - Total production determines the market price: P = A - B*(total production).
          - Each firm's profit is computed as:
                profit = production * P - production * cost - tau * production.
          - Each firm's cost shock transitions stochastically according to cost_trans.
        """
        production = np.array(actions)  # Production chosen by each firm.
        total_production = np.sum(production)
        price = self.parameters['A'] - self.parameters['B'] * total_production

        # Compute profits for each firm.
        profits = production * price - production * self.state - self.parameters['tau'] * production

        # Transition each firm's cost shock.
        new_state = np.zeros_like(self.state)
        for i in range(self.parameters['n_firms']):
            current_state_idx = np.where(self.parameters['cost_states'] == self.state[i])[0][0]
            new_state[i] = self.rng.choice(
                self.parameters['cost_states'], 
                p=self.parameters['cost_trans'][current_state_idx]
            )
        self.state = new_state

        done = False
        truncated = False
        return self.state, profits, done, truncated, {}

    def find_equilibrium(self, tol=1e-4, max_iter=1000, plot=True):
        """
        Compute a Markov Perfect Equilibrium (MPE) for the dynamic game using value
        function iteration. Each firm's value function satisfies:

          V(c) = max_{a in {0,1,2}} { a * [A - B*(a + sum(a_{-i}))] - a * c - tau*a
                                    + beta * E[V(c')] }

        where:
          - c is the current cost shock,
          - a is the firm's production level,
          - a_{-i} are the production levels of other firms,
          - c' is the next period's cost shock.
        
        The equilibrium is found by iterating until each firm's policy is a best response
        to the policies of other firms.
        
        Returns:
            policy: An array of shape (n_firms, num_cost_states) mapping each cost shock to an optimal production level.
            V: An array of shape (n_firms, num_cost_states) representing the value function.
        """
        num_states = len(self.parameters['cost_states'])
        V = np.zeros((self.parameters['n_firms'], num_states))
        policy = np.zeros((self.parameters['n_firms'], num_states), dtype=int)
        
        for iteration in range(max_iter):
            V_new = np.copy(V)
            policy_new = np.copy(policy)
            delta = 0
            
            # For each firm
            for i in range(self.parameters['n_firms']):
                # For each cost state
                for c_idx, cost in enumerate(self.parameters['cost_states']):
                    best_value = -np.inf
                    best_action = 0
                    
                    # For each possible action
                    for a in range(3):  # production choices: 0, 1, 2
                        # Compute total production considering other firms' actions
                        total_prod = a
                        for j in range(self.parameters['n_firms']):
                            if j != i:
                                # Get other firms' actions based on their current policies
                                # Consider all possible cost states for other firms
                                other_action = 0
                                for other_c_idx in range(num_states):
                                    other_action += policy[j, other_c_idx] * self.parameters['cost_trans'][c_idx, other_c_idx]
                                total_prod += other_action
                        
                        # Compute price and profit
                        price = self.parameters['A'] - self.parameters['B'] * total_prod
                        profit = a * price - a * cost - self.parameters['tau'] * a
                        
                        # Expected continuation value
                        cont_val = 0
                        for next_c_idx in range(num_states):
                            cont_val += self.parameters['cost_trans'][c_idx, next_c_idx] * V[i, next_c_idx]
                        
                        value = profit + self.parameters['beta'] * cont_val
                        
                        if value > best_value:
                            best_value = value
                            best_action = a
                    
                    V_new[i, c_idx] = best_value
                    policy_new[i, c_idx] = best_action
                    delta = max(delta, abs(V_new[i, c_idx] - V[i, c_idx]))
            
            # Check for convergence
            if delta < tol:
                break
                
            V = V_new
            policy = policy_new
        
        self.equilibrium_policy = policy
        self.equilibrium_value = V
        self.equilibrium_state = np.array([V, policy], dtype=object)
        
        if plot:
            self.plot_equilibrium(policy, V)
            self.plot_transition_probabilities()
        
        return policy, V

    def get_value_function(self, state):
        """
        Get the value function for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment
            
        Returns:
            The value function for the current state
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for each firm's current cost state
        values = np.zeros(self.parameters['n_firms'])
        for i in range(self.parameters['n_firms']):
            cost_idx = np.where(self.parameters['cost_states'] == state[i])[0][0]
            values[i] = self.equilibrium_value[i, cost_idx]
        
        return values

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment
            
        Returns:
            The policy for the current state
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for each firm's current cost state
        actions = np.zeros(self.parameters['n_firms'], dtype=int)
        for i in range(self.parameters['n_firms']):
            cost_idx = np.where(self.parameters['cost_states'] == state[i])[0][0]
            actions[i] = self.equilibrium_policy[i, cost_idx]
        
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
        from scipy.optimize import minimize
        
        def objective(params):
            # Update parameters
            self.parameters['A'] = params[0]
            self.parameters['B'] = params[1]
            self.parameters['tau'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, profits = self.simulate(n_periods=1000)
            
            # Compute moments
            avg_production = np.mean([np.sum(a) for a in states])
            avg_profit = np.mean(profits)
            market_concentration = np.mean([np.max(a) / np.sum(a) for a in states])
            
            # Compute squared differences from targets
            error = 0
            if 'avg_production' in targets:
                error += (avg_production - targets['avg_production'])**2
            if 'avg_profit' in targets:
                error += (avg_profit - targets['avg_profit'])**2
            if 'market_concentration' in targets:
                error += (market_concentration - targets['market_concentration'])**2
                
            return error
        
        # Initial guess and bounds
        x0 = [
            self.parameters['A'],
            self.parameters['B'],
            self.parameters['tau']
        ]
        bounds = [
            (5, 15),    # A: demand intercept
            (0.1, 1.0), # B: demand slope
            (0.1, 2.0)  # tau: environmental tax
        ]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['A'] = result.x[0]
        self.parameters['B'] = result.x[1]
        self.parameters['tau'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'A': result.x[0],
                'B': result.x[1],
                'tau': result.x[2]
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
            self.parameters['A'] = params[0]
            self.parameters['B'] = params[1]
            self.parameters['tau'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, profits = self.simulate(n_periods=1000)
            
            # Compute simulated moments
            sim_moments = moment_function(self, states, profits)
            
            # Compute GMM objective
            diff = sim_moments - data
            return diff @ weight_matrix @ diff
        
        # Initial guess and bounds
        x0 = [
            self.parameters['A'],
            self.parameters['B'],
            self.parameters['tau']
        ]
        bounds = [
            (5, 15),    # A: demand intercept
            (0.1, 1.0), # B: demand slope
            (0.1, 2.0)  # tau: environmental tax
        ]
        
        # Optimize
        result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['A'] = result.x[0]
        self.parameters['B'] = result.x[1]
        self.parameters['tau'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'estimated_parameters': {
                'A': result.x[0],
                'B': result.x[1],
                'tau': result.x[2]
            },
            'objective_value': result.fun
        }

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current cost shocks: {self.state}")
            print(f"Parameters: A={self.parameters['A']:.2f}, B={self.parameters['B']:.2f}, "
                  f"tau={self.parameters['tau']:.2f}")
        return None

    def plot_simulation_results(self, states, rewards, n_periods=1000):
        """
        Plot simulation results including cost shocks, profits, and market concentration.
        
        Args:
            states: List of states from simulation
            rewards: List of rewards from simulation
            n_periods: Number of periods to plot
        """
        # Convert to numpy arrays
        states = np.array(states)
        rewards = np.array(rewards)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot cost shocks
        for i in range(self.parameters['n_firms']):
            ax1.plot(states[:n_periods, i], label=f'Firm {i+1}')
        ax1.set_title('Cost Shocks Over Time')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Cost Shock')
        ax1.legend()
        ax1.grid(True)
        
        # Plot profits
        for i in range(self.parameters['n_firms']):
            ax2.plot(rewards[:n_periods, i], label=f'Firm {i+1}')
        ax2.set_title('Profits Over Time')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Profit')
        ax2.legend()
        ax2.grid(True)
        
        # Plot market concentration
        total_production = np.sum(states[:n_periods], axis=1)
        concentration = np.max(states[:n_periods], axis=1) / total_production
        ax3.plot(concentration)
        ax3.set_title('Market Concentration Over Time')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Concentration (Max Share)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_equilibrium(self, policy, V):
        """
        Plot the equilibrium policy and value function.
        
        Args:
            policy: Equilibrium policy array
            V: Value function array
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot policy
        for i in range(self.parameters['n_firms']):
            ax1.plot(self.parameters['cost_states'], policy[i], 
                    marker='o', label=f'Firm {i+1}')
        ax1.set_title('Equilibrium Policy')
        ax1.set_xlabel('Cost State')
        ax1.set_ylabel('Production Level')
        ax1.legend()
        ax1.grid(True)
        
        # Plot value function
        for i in range(self.parameters['n_firms']):
            ax2.plot(self.parameters['cost_states'], V[i], 
                    marker='o', label=f'Firm {i+1}')
        ax2.set_title('Value Function')
        ax2.set_xlabel('Cost State')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_transition_probabilities(self):
        """Plot the transition probability matrix as a heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.parameters['cost_trans'], 
                   annot=True, 
                   fmt='.2f',
                   xticklabels=self.parameters['cost_states'],
                   yticklabels=self.parameters['cost_states'])
        plt.title('Cost State Transition Probabilities')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.show()

    def simulate(self, n_periods=1000, n_agents=1000, plot=True):
        """
        Simulate the model for a given number of periods.
        
        Args:
            n_periods: Number of periods to simulate
            n_agents: Number of agents to simulate
            plot: Whether to plot the results
            
        Returns:
            Tuple of (states, rewards) where states is a list of states and rewards is a list of rewards
        """
        states = []
        rewards = []
        
        # Reset environment
        state, _ = self.reset()
        
        for _ in range(n_periods):
            # Get action from equilibrium policy
            action = self.get_policy(state)
            
            # Take step
            next_state, reward, _, _, _ = self.step(action)
            
            # Store results
            states.append(state)
            rewards.append(reward)
            
            # Update state
            state = next_state
        
        if plot:
            self.plot_simulation_results(states, rewards, n_periods)
        
        return states, rewards


# Example usage:
if __name__ == "__main__":
    env = RyanEnv(n_firms=3)
    state = env.reset()
    print("Initial state (cost shocks):", state)
    
    # Sample actions: each firm randomly selects a production level from {0, 1, 2}.
    actions = env.action_space.sample()
    print("Sample actions (production levels):", actions)
    
    next_state, profits, done, _ = env.step(actions)
    print("Next state (cost shocks):", next_state)
    print("Profits:", profits)
    
    # Approximate an equilibrium policy.
    policy, V = env.find_equilibrium()
    print("\nEquilibrium policy for a representative firm (for each cost shock):")
    for i, cost in enumerate(env.parameters['cost_states']):
        print(f"  Cost shock = {cost:.1f}: Optimal production = {policy[i]}, Value = {V[i]:.2f}")
    
    # Example calibration
    targets = {
        'avg_production': 4.0,
        'avg_profit': 2.0,
        'market_concentration': 0.4
    }
    calibration_results = env.calibrate(targets)
    print("\nCalibration results:", calibration_results)
    
    # Example estimation
    data = np.array([4.0, 2.0, 0.4])  # Example empirical moments
    moment_function = lambda env, states, profits: np.array([
        np.mean([np.sum(a) for a in states]),  # average production
        np.mean(profits),                      # average profit
        np.mean([np.max(a) / np.sum(a) for a in states])  # market concentration
    ])
    weight_matrix = np.eye(3)
    estimation_results = env.estimate(data, moment_function, weight_matrix)
    print("\nEstimation results:", estimation_results)