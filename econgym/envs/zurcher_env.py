import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class ZurcherEnv(EconEnv):
    """
    Harold Zurcher Bus Replacement Model Environment.

    State: 
        - m: current mileage (discrete)
    Action: 
        - 0: Continue using the engine
        - 1: Replace the engine
    """
    def __init__(self, max_mileage=100, replace_cost=500, beta=0.9):
        super().__init__()
        # Initialize parameters
        self.parameters = {
            'max_mileage': max_mileage,
            'replace_cost': replace_cost,
            'beta': beta,
            'maintenance_cost_base': 2.0,
            'maintenance_cost_slope': 0.1
        }
        
        # Define the state space: mileage from 0 to max_mileage (inclusive)
        self.observation_space = spaces.Discrete(self.parameters['max_mileage'] + 1)
        # Define the action space: 0 = continue, 1 = replace
        self.action_space = spaces.Discrete(2)
        self.state = None  # Current mileage

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state (new engine with zero mileage)."""
        if seed is not None:
            np.random.seed(seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        """
        Execute one time step in the environment dynamics.
        
        Action:
            - 0: Continue operating (increase mileage)
            - 1: Replace the engine (reset mileage to 0 and incur cost)
        """
        mileage = self.state

        if action == 1:  # Replace the engine
            next_state = 0
            reward = -self.parameters['replace_cost']
        else:            # Continue operating
            next_state = min(mileage + 1, self.parameters['max_mileage'])
            reward = -self.maintenance_cost(mileage)
        self.state = next_state
        # When mileage reaches the maximum, a replacement is forced.
        done = (mileage >= self.parameters['max_mileage'])
        truncated = False
        return next_state, reward, done, truncated, {}

    def maintenance_cost(self, mileage):
        """Maintenance cost function that increases linearly with mileage."""
        return (self.parameters['maintenance_cost_base'] + 
                self.parameters['maintenance_cost_slope'] * mileage)

    def find_equilibrium(self, tol=1e-6, max_iter=1000):
        """
        Solve for the optimal replacement policy using value function iteration.
        
        For each mileage state m (0 <= m <= max_mileage), the value function satisfies:
            V(m) = max {
                        -maintenance_cost(m) + beta * V(m+1)       if m < max_mileage,
                        -replace_cost + beta * V(0)                  (replace action),
                        [At m == max_mileage, forced replacement is applied]
                     }
        The method returns both the optimal policy and the value function.
        """
        # Initialize the value function for each mileage state.
        V = np.zeros(self.parameters['max_mileage'] + 1)
        # To store the optimal action (0: continue, 1: replace) for each state.
        policy = np.zeros(self.parameters['max_mileage'] + 1, dtype=int)
        
        print("Starting value function iteration...")
        for iteration in range(max_iter):
            V_new = np.zeros_like(V)
            for m in range(self.parameters['max_mileage'] + 1):
                if m == self.parameters['max_mileage']:
                    # At maximum mileage, replacement is forced.
                    value_replace = -self.parameters['replace_cost'] + self.parameters['beta'] * V[0]
                    V_new[m] = value_replace
                    policy[m] = 1
                else:
                    # If the engine is continued:
                    maintenance = self.maintenance_cost(m)
                    value_continue = -maintenance + self.parameters['beta'] * V[m+1]
                    
                    # If the engine is replaced:
                    value_replace = -self.parameters['replace_cost'] + self.parameters['beta'] * V[0]
                    
                    # Choose the action with higher value
                    if value_continue >= value_replace:
                        V_new[m] = value_continue
                        policy[m] = 0
                    else:
                        V_new[m] = value_replace
                        policy[m] = 1
            
            # Check convergence of the value function
            diff = np.max(np.abs(V_new - V))
            if diff < tol:
                print(f"Value function iteration converged after {iteration + 1} iterations")
                print(f"Final difference: {diff:.2e}")
                V = V_new
                break
            V = V_new
            
            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Max difference: {diff:.2e}")
        
        self.V = V
        self.policy = policy
        self.equilibrium_state = np.array([V, policy], dtype=object)
        return policy, V

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
            self.parameters['replace_cost'] = params[0]
            self.parameters['maintenance_cost_base'] = params[1]
            self.parameters['maintenance_cost_slope'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Compute moments
            replacement_frequency = np.mean(policy)
            avg_mileage = np.mean([m for m, a in enumerate(policy) if a == 0])
            
            # Compute squared differences from targets
            error = 0
            if 'replacement_frequency' in targets:
                error += (replacement_frequency - targets['replacement_frequency'])**2
            if 'avg_mileage' in targets:
                error += (avg_mileage - targets['avg_mileage'])**2
                
            return error
        
        # Initial guess and bounds
        x0 = [
            self.parameters['replace_cost'],
            self.parameters['maintenance_cost_base'],
            self.parameters['maintenance_cost_slope']
        ]
        bounds = [
            (100, 1000),  # replace_cost
            (1, 5),       # maintenance_cost_base
            (0.05, 0.2)   # maintenance_cost_slope
        ]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['replace_cost'] = result.x[0]
        self.parameters['maintenance_cost_base'] = result.x[1]
        self.parameters['maintenance_cost_slope'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'replace_cost': result.x[0],
                'maintenance_cost_base': result.x[1],
                'maintenance_cost_slope': result.x[2]
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
            self.parameters['replace_cost'] = params[0]
            self.parameters['maintenance_cost_base'] = params[1]
            self.parameters['maintenance_cost_slope'] = params[2]
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Compute simulated moments
            sim_moments = moment_function(self, policy)
            
            # Compute GMM objective
            diff = sim_moments - data
            return diff @ weight_matrix @ diff
        
        # Initial guess and bounds
        x0 = [
            self.parameters['replace_cost'],
            self.parameters['maintenance_cost_base'],
            self.parameters['maintenance_cost_slope']
        ]
        bounds = [
            (100, 1000),  # replace_cost
            (1, 5),       # maintenance_cost_base
            (0.05, 0.2)   # maintenance_cost_slope
        ]
        
        # Optimize
        result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['replace_cost'] = result.x[0]
        self.parameters['maintenance_cost_base'] = result.x[1]
        self.parameters['maintenance_cost_slope'] = result.x[2]
        
        return {
            'success': result.success,
            'message': result.message,
            'estimated_parameters': {
                'replace_cost': result.x[0],
                'maintenance_cost_base': result.x[1],
                'maintenance_cost_slope': result.x[2]
            },
            'objective_value': result.fun
        }

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current mileage: {self.state}")
            print(f"Parameters: replace_cost={self.parameters['replace_cost']:.2f}, "
                  f"maintenance_cost_base={self.parameters['maintenance_cost_base']:.2f}, "
                  f"maintenance_cost_slope={self.parameters['maintenance_cost_slope']:.2f}")
        return None

    def get_value_function(self, state):
        """
        Get the value function for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [mileage, shock]
            
        Returns:
            The value function for the current state
        """
        if not hasattr(self, 'equilibrium_value'):
            self.find_equilibrium()
        
        # Get the value for the current state
        mileage_idx = int(state[0] * 100 / self.parameters['max_mileage'])
        mileage_idx = np.clip(mileage_idx, 0, 100)
        shock_idx = np.where(self.parameters['shock_vals'] == state[1])[0][0]
        
        return self.equilibrium_value[shock_idx, mileage_idx]

    def get_policy(self, state):
        """
        Get the policy for a given state.
        This method is used by the value function wrapper.
        
        Args:
            state: The current state of the environment [mileage, shock]
            
        Returns:
            The policy for the current state
        """
        if not hasattr(self, 'equilibrium_policy'):
            self.find_equilibrium()
        
        # Get the policy for the current state
        mileage_idx = int(state[0] * 100 / self.parameters['max_mileage'])
        mileage_idx = np.clip(mileage_idx, 0, 100)
        shock_idx = np.where(self.parameters['shock_vals'] == state[1])[0][0]
        
        return self.equilibrium_policy[shock_idx, mileage_idx]


def load_bus_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess bus replacement data.
    
    Args:
        file_path: Path to the CSV file containing bus data
        
    Returns:
        DataFrame with processed bus data
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    required_cols = ['state', 'decision', 'usage']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")
    
    # Convert state to mileage (if needed)
    if df['state'].max() > 100:  # Assuming max_mileage is 100
        df['state'] = (df['state'] / df['state'].max() * 100).round()
    
    # Ensure decision is binary (0 or 1)
    df['decision'] = df['decision'].astype(int)
    
    return df


def compute_empirical_moments(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical moments from the bus data.
    
    Args:
        df: DataFrame containing bus data
        
    Returns:
        Tuple of (moments, weight_matrix)
    """
    # Compute replacement frequency
    replacement_freq = df['decision'].mean()
    
    # Compute average mileage at replacement
    avg_mileage = df[df['decision'] == 1]['state'].mean()
    
    # Compute variance of mileage at replacement
    mileage_var = df[df['decision'] == 1]['state'].var()
    
    # Compute average usage between replacements
    usage_between_replacements = df.groupby(
        (df['decision'] == 1).cumsum()
    )['usage'].mean().mean()
    
    # Stack moments
    moments = np.array([
        replacement_freq,
        avg_mileage,
        mileage_var,
        usage_between_replacements
    ])
    
    # Create weight matrix (using inverse of variance)
    weight_matrix = np.diag([
        1 / (replacement_freq * (1 - replacement_freq)),  # Variance of replacement frequency
        1 / mileage_var,  # Inverse of mileage variance
        1 / (2 * mileage_var**2),  # Variance of variance
        1 / df['usage'].var()  # Variance of usage
    ])
    
    return moments, weight_matrix


def simulate_moments(env: ZurcherEnv, policy: np.ndarray, n_simulations: int = 1000) -> np.ndarray:
    """
    Simulate moments from the model for comparison with empirical moments.
    
    Args:
        env: ZurcherEnv instance
        policy: Optimal policy array
        n_simulations: Number of simulations to run
        
    Returns:
        Array of simulated moments
    """
    # Initialize arrays to store simulation results
    replacement_decisions = []
    mileages_at_replacement = []
    usages_between_replacements = []
    
    for _ in range(n_simulations):
        state, _ = env.reset()
        usage = 0
        mileages = []
        
        while True:
            # Get action from policy
            action = policy[state]
            
            # Record mileage if replacement occurs
            if action == 1:
                mileages_at_replacement.append(state)
                usages_between_replacements.append(usage)
                usage = 0
            else:
                usage += 1
                mileages.append(state)
            
            # Take step
            state, _, done, _, _ = env.step(action)
            
            if done:
                break
        
        replacement_decisions.append(action)
    
    # Compute moments
    replacement_freq = np.mean(replacement_decisions)
    avg_mileage = np.mean(mileages_at_replacement)
    mileage_var = np.var(mileages_at_replacement)
    avg_usage = np.mean(usages_between_replacements)
    
    return np.array([
        replacement_freq,
        avg_mileage,
        mileage_var,
        avg_usage
    ])


# Example usage:
if __name__ == "__main__":
    # Create environment
    env = ZurcherEnv(max_mileage=100, replace_cost=500, beta=0.9)
    
    try:
        # Load and process data
        df = load_bus_data("bus_data.csv")
        print("\nLoaded bus data:")
        print(df.head())
        
        # Compute empirical moments
        empirical_moments, weight_matrix = compute_empirical_moments(df)
        print("\nEmpirical moments:")
        print(f"Replacement frequency: {empirical_moments[0]:.3f}")
        print(f"Average mileage at replacement: {empirical_moments[1]:.1f}")
        print(f"Variance of mileage at replacement: {empirical_moments[2]:.1f}")
        print(f"Average usage between replacements: {empirical_moments[3]:.1f}")
        
        # Define moment function for estimation
        def moment_function(env, policy):
            return simulate_moments(env, policy)
        
        # Estimate parameters
        print("\nEstimating parameters...")
        estimation_results = env.estimate(
            empirical_moments,
            moment_function,
            weight_matrix,
            method="BFGS",
            options={'maxiter': 1000}
        )
        
        print("\nEstimation results:")
        print(f"Success: {estimation_results['success']}")
        print(f"Message: {estimation_results['message']}")
        print("\nEstimated parameters:")
        for param, value in estimation_results['estimated_parameters'].items():
            print(f"{param}: {value:.2f}")
        
        # Simulate with estimated parameters
        policy, V = env.find_equilibrium()
        simulated_moments = simulate_moments(env, policy)
        
        print("\nModel fit:")
        print("Moment\t\tEmpirical\tSimulated")
        print("-" * 40)
        moments_names = ["Replacement freq", "Avg mileage", "Mileage var", "Avg usage"]
        for name, emp, sim in zip(moments_names, empirical_moments, simulated_moments):
            print(f"{name:<15} {emp:>8.3f} {sim:>12.3f}")
        
    except FileNotFoundError:
        print("\nNo data file found. Running basic demo...")
        # Basic demo without data
        state, _ = env.reset()
        print("Initial state (mileage):", state)
        
        # Simulate a few steps with random actions
        for _ in range(5):
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            env.render()
            print("Reward:", reward, "Done:", done)
        
        # Solve for optimal policy
        opt_policy, V = env.find_equilibrium()
        print("\nOptimal policy (0: continue, 1: replace) and value function:")
        for m in range(len(opt_policy)):
            print(f"Mileage {m:3d}: Action {opt_policy[m]} , Value {V[m]:.2f}")