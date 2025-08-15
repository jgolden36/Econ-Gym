import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from typing import Dict, Any, Optional, Tuple, List


class BBLGameEnv(EconEnv):
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
        
        # Each firm chooses among 3 discrete pricing levels: 
        # 0 = low price, 1 = medium price, 2 = high price.
        self.action_space = spaces.MultiDiscrete([self.parameters['price_levels']] * self.parameters['n_firms'])
        
        # Initialize firm-specific states.
        self.state = np.zeros((self.parameters['n_firms'], self.parameters['state_dim']), dtype=np.float32)
        
        # A common market shock (e.g., demand shock) in [0, 1].
        self.market_shock = 0.0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        Firm states are randomly drawn in [0, 100] and a market shock is drawn uniformly.
        """
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(
            low=self.parameters['state_min'],
            high=self.parameters['state_max'],
            size=(self.parameters['n_firms'], self.parameters['state_dim'])
        ).astype(np.float32)
        self.market_shock = np.random.rand()
        return self._get_obs(), {}
    
    def step(self, actions):
        """
        Execute one period of the dynamic game.
        
        Parameters:
            actions: An array of length n_firms where each element is a discrete action 
                     (pricing decision) for the corresponding firm.
        
        Returns:
            next_obs: Updated observation (firm states).
            rewards: Immediate payoffs for each firm (as a numpy array).
            done: Boolean flag (always False here as the game is ongoing).
            truncated: Boolean flag for episode truncation.
            info: Dictionary for additional information.
        """
        # Compute payoffs for each firm given the actions and market shock.
        rewards = self._compute_payoffs(actions)
        
        # Update each firm's state based on its chosen action.
        new_state = self.state.copy()
        for i in range(self.parameters['n_firms']):
            # A simplified dynamic: a high price (action 2) may improve the firm's state
            # (e.g., reputation or perceived quality), while a low price (action 0) may reduce it.
            if actions[i] == 0:
                new_state[i] = np.maximum(
                    new_state[i] + self.parameters['state_adjustment']['low_price'],
                    self.parameters['state_min']
                )
            elif actions[i] == 2:
                new_state[i] = np.minimum(
                    new_state[i] + self.parameters['state_adjustment']['high_price'],
                    self.parameters['state_max']
                )
            else:
                # Medium price results in a small random adjustment.
                adjustment = np.random.normal(
                    0,
                    self.parameters['state_adjustment']['medium_price_std'],
                    size=new_state[i].shape
                )
                new_state[i] = np.clip(
                    new_state[i] + adjustment,
                    self.parameters['state_min'],
                    self.parameters['state_max']
                )
        
        self.state = new_state
        # Update market shock for the next period.
        self.market_shock = np.random.rand()
        
        done = False  # The dynamic game is assumed to run indefinitely.
        truncated = False
        return self._get_obs(), rewards, done, truncated, {}

    def _get_obs(self):
        """
        Return the current observation, which is the matrix of firm states.
        """
        return self.state

    def _compute_payoffs(self, actions):
        """
        Compute each firm's immediate payoff based on its action, the actions of rivals,
        and the current market shock.
        
        For illustration, we assume:
            revenue_i = (price level of firm i) * revenue_multiplier * (market_shock)
            cost_i = cost_multiplier * (firm i's current state)
            
        The payoff is then: revenue_i - cost_i.
        Note that in a richer model, rival actions would affect demand and strategic interactions.
        """
        payoffs = np.zeros(self.parameters['n_firms'])
        for i in range(self.parameters['n_firms']):
            price_level = actions[i]
            revenue = price_level * self.parameters['revenue_multiplier'] * self.market_shock
            cost = self.parameters['cost_multiplier'] * self.state[i, 0]
            payoffs[i] = revenue - cost
        return payoffs

    def find_equilibrium(self, tol=1e-6, max_iter=1000):
        """
        Compute a Markov Perfect Equilibrium (MPE) policy using value function iteration.
        
        For each firm, we discretize its state (0 to 100) and compute a value function V(s)
        that satisfies:
        
            V(s) = max_{action in {0,1,2}} { immediate_payoff(s, action, s_{-i}, a_{-i}) + beta * V(s_next) }
        
        where s_{-i} and a_{-i} are the states and actions of other firms.
        
        The equilibrium is found by iterating until each firm's policy is a best response
        to the policies of other firms.
        
        Returns:
            policy: A (n_firms x 101) array mapping each discretized state (0 to 100) to an action.
            V: A (n_firms x 101) array representing the value function for each firm.
        """
        # Discretize state space: assume integer states 0, 1, ..., 100.
        num_states = 101
        V = np.zeros((self.parameters['n_firms'], num_states))
        policy = np.zeros((self.parameters['n_firms'], num_states), dtype=int)
        
        # For simplicity, use an average market shock (e.g., 0.5) in the immediate payoff.
        avg_shock = 0.5
        
        # Iterate until convergence of policies
        for iteration in range(max_iter):
            V_new = np.copy(V)
            policy_new = np.copy(policy)
            delta = 0
            
            # For each firm
            for i in range(self.parameters['n_firms']):
                # For each state
                for s in range(num_states):
                    # Evaluate each action
                    value_actions = np.zeros(self.parameters['price_levels'])
                    
                    # For each possible action
                    for a in range(self.parameters['price_levels']):
                        # Compute next state for firm i
                        if a == 0:
                            next_s = max(s - 1, 0)
                        elif a == 2:
                            next_s = min(s + 1, 100)
                        else:
                            next_s = s
                        
                        # Compute immediate payoff considering other firms' actions
                        # For each possible state of other firms
                        total_payoff = 0
                        for other_states in self._get_other_firms_states(s, i):
                            # Get actions of other firms based on their current policies
                            other_actions = np.array([
                                policy[j, other_states[j]] for j in range(self.parameters['n_firms']) if j != i
                            ])
                            
                            # Compute payoff for firm i given its action and other firms' actions
                            actions = np.insert(other_actions, i, a)
                            payoff = self._compute_payoff_for_firm(i, actions, s, other_states, avg_shock)
                            total_payoff += payoff
                        
                        # Average payoff over possible states of other firms
                        value_actions[a] = total_payoff / len(self._get_other_firms_states(s, i))
                        
                        # Add continuation value
                        value_actions[a] += self.parameters['beta'] * V[i, next_s]
                    
                    # Update value function and policy
                    best_a = np.argmax(value_actions)
                    V_new[i, s] = value_actions[best_a]
                    policy_new[i, s] = best_a
                    delta = max(delta, abs(V_new[i, s] - V[i, s]))
            
            # Check for convergence
            if delta < tol:
                break
                
            V = V_new
            policy = policy_new
        
        self.V = V
        self.policy = policy
        self.equilibrium_state = np.array([V, policy], dtype=object)
        return policy, V

    def _get_other_firms_states(self, current_state, firm_idx):
        """
        Generate possible states for other firms given the current state of firm_idx.
        For simplicity, we consider a small set of possible states around the current state.
        """
        possible_states = []
        for j in range(self.parameters['n_firms']):
            if j != firm_idx:
                # Consider states within ±2 of the current state
                states = np.arange(max(0, current_state - 2), min(101, current_state + 3))
                possible_states.append(states)
        
        # Generate all combinations of possible states
        from itertools import product
        return list(product(*possible_states))

    def _compute_payoff_for_firm(self, firm_idx, actions, firm_state, other_states, market_shock):
        """
        Compute the payoff for a specific firm given its state, action, and the states and actions of other firms.
        """
        # Construct the full state vector
        state = np.zeros((self.parameters['n_firms'], self.parameters['state_dim']))
        state[firm_idx] = firm_state
        for j, s in enumerate(other_states):
            if j < firm_idx:
                state[j] = s
            else:
                state[j+1] = s
        
        # Compute revenue
        price_level = actions[firm_idx]
        revenue = price_level * self.parameters['revenue_multiplier'] * market_shock
        
        # Compute cost
        cost = self.parameters['cost_multiplier'] * state[firm_idx, 0]
        
        # Add competitive effects
        # If other firms choose higher prices, this firm's revenue increases
        # If other firms choose lower prices, this firm's revenue decreases
        competitive_effect = 0
        for j, a in enumerate(actions):
            if j != firm_idx:
                competitive_effect += (price_level - a) * 0.1  # Small effect of price differences
        
        return revenue - cost + competitive_effect

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
            self.parameters['revenue_multiplier'] = params[0]
            self.parameters['cost_multiplier'] = params[1]
            self.parameters['state_adjustment']['high_price'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute moments
            avg_price = np.mean([np.mean(a) for a in states])
            avg_profit = np.mean(rewards)
            market_concentration = np.mean([np.max(a) / np.sum(a) for a in states])
            
            # Compute squared differences from targets
            error = 0
            if 'avg_price' in targets:
                error += (avg_price - targets['avg_price'])**2
            if 'avg_profit' in targets:
                error += (avg_profit - targets['avg_profit'])**2
            if 'market_concentration' in targets:
                error += (market_concentration - targets['market_concentration'])**2
                
            return error
        
        # Initial guess and bounds
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
                'state_adjustment': result.x[2]
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
            self.parameters['revenue_multiplier'] = params[0]
            self.parameters['cost_multiplier'] = params[1]
            self.parameters['state_adjustment']['high_price'] = params[2]
            
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
        result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['revenue_multiplier'] = result.x[0]
        self.parameters['cost_multiplier'] = result.x[1]
        self.parameters['state_adjustment']['high_price'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'estimated_parameters': {
                'revenue_multiplier': result.x[0],
                'cost_multiplier': result.x[1],
                'state_adjustment': result.x[2]
            },
            'objective_value': result.fun
        }

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print("Firm States:")
            for i in range(self.parameters['n_firms']):
                print(f"  Firm {i}: State {self.state[i]}")
            print(f"Market Shock: {self.market_shock}")
            print(f"Parameters: revenue_multiplier={self.parameters['revenue_multiplier']:.2f}, "
                  f"cost_multiplier={self.parameters['cost_multiplier']:.2f}, "
                  f"state_adjustment={self.parameters['state_adjustment']['high_price']:.2f}")
        return None

    def get_value_function(self, state):
        """
        Get the value function for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment (firm states)
            
        Returns:
            The value function for the current state
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for each firm's current state
        values = np.zeros(self.parameters['n_firms'])
        for i in range(self.parameters['n_firms']):
            # Find the closest state in the discretized grid
            state_idx = int(state[i, 0] * 100 / self.parameters['state_max'])
            state_idx = np.clip(state_idx, 0, 100)
            values[i] = self.equilibrium_value[i, state_idx]
        
        return values

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment (firm states)
            
        Returns:
            The policy for the current state
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for each firm's current state
        actions = np.zeros(self.parameters['n_firms'], dtype=int)
        for i in range(self.parameters['n_firms']):
            # Find the closest state in the discretized grid
            state_idx = int(state[i, 0] * 100 / self.parameters['state_max'])
            state_idx = np.clip(state_idx, 0, 100)
            actions[i] = self.equilibrium_policy[i, state_idx]
        
        return actions


# Example usage:
if __name__ == "__main__":
    env = BBLGameEnv(n_firms=3)
    obs = env.reset()
    print("Initial observation (firm states):")
    print(obs)
    
    # Sample actions: each firm randomly chooses a pricing level.
    actions = env.action_space.sample()
    print("Sample actions:", actions)
    
    next_obs, rewards, done, truncated, _ = env.step(actions)
    print("Next observation (firm states):")
    print(next_obs)
    print("Rewards:", rewards)
    
    # Compute an equilibrium policy
    policy, V = env.find_equilibrium()
    print("\nEquilibrium policy (for a representative discretized state):")
    for i in range(env.parameters['n_firms']):
        print(f"Firm {i}:")
        for s in range(0, 101, 20):
            print(f"  State {s}: Action {policy[i, s]}, Value {V[i, s]:.2f}")
    
    # Example calibration
    targets = {
        'avg_price': 1.5,
        'avg_profit': 2.0,
        'market_concentration': 0.4
    }
    calibration_results = env.calibrate(targets)
    print("\nCalibration results:", calibration_results)
    
    # Example estimation
    data = np.array([1.5, 2.0, 0.4])  # Example empirical moments
    moment_function = lambda env, states, rewards: np.array([
        np.mean([np.mean(a) for a in states]),  # average price
        np.mean(rewards),                      # average profit
        np.mean([np.max(a) / np.sum(a) for a in states])  # market concentration
    ])
    weight_matrix = np.eye(3)
    estimation_results = env.estimate(data, moment_function, weight_matrix)
    print("\nEstimation results:", estimation_results)