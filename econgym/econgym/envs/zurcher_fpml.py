import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.optimize import minimize
from econgym.envs.zurcher_env import ZurcherEnv

class ZurcherFPML:
    """
    Fixed Point Maximum Likelihood estimation for the Zurcher bus replacement model.
    Implements the nested fixed point algorithm from Rust (1987).
    """
    def __init__(self, 
                 max_mileage: int = 400000,  # Increased to handle actual mileage values
                 replace_cost: float = 500.0,
                 beta: float = 0.99,
                 poisson_lambda: float = 1.0):
        self.env = ZurcherEnv(
            max_mileage=max_mileage,
            replace_cost=replace_cost,
            beta=beta,
            poisson_lambda=poisson_lambda
        )
        
        # Pre-compute Poisson probabilities for jumps
        self.max_jump = 20
        self.poisson_probs = np.array([
            np.exp(-poisson_lambda) * 
            (poisson_lambda ** j) / np.math.factorial(j)
            for j in range(self.max_jump)
        ])
        self.poisson_probs = self.poisson_probs / np.sum(self.poisson_probs)
        
        # Initialize value function and policy
        self.value_function = None
        self.policy = None
        
        # State space discretization
        self.state_grid = np.linspace(0, max_mileage, 1000)  # Fine grid for interpolation
        
    def get_state_index(self, state):
        """Convert continuous state to nearest grid point."""
        return np.abs(self.state_grid - state).argmin()
        
    def fixed_point_mapping(self, V: np.ndarray) -> np.ndarray:
        """
        Fixed point mapping for the value function.
        Implements the Bellman operator T(V) = max_a {r(a) + β * E[V(s')|s,a]}
        """
        V_new = np.zeros_like(V)
        
        # Process states in batches to show progress
        batch_size = 100
        n_batches = len(self.state_grid) // batch_size + (1 if len(self.state_grid) % batch_size else 0)
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(self.state_grid))
            
            if batch % 10 == 0:
                print(f"Processing states {start_idx} to {end_idx} of {len(self.state_grid)}")
            
            for i in range(start_idx, end_idx):
                m = self.state_grid[i]
                
                # Value of replacing
                replace_value = -self.env.parameters['replace_cost'] + self.env.parameters['beta'] * V[0]
                
                # Value of not replacing
                keep_value = -self.env.maintenance_cost(m)
                # Add expected future value
                for jump, prob in enumerate(self.poisson_probs):
                    next_m = min(m + jump * 1000, self.env.parameters['max_mileage'])  # Scale jumps
                    next_idx = self.get_state_index(next_m)
                    keep_value += self.env.parameters['beta'] * prob * V[next_idx]
                
                # Choose best action
                V_new[i] = max(replace_value, keep_value)
        
        return V_new
    
    def find_fixed_point(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the fixed point of the value function using value function iteration.
        Returns both the value function and the optimal policy.
        """
        print("Starting fixed point iteration...")
        V = np.zeros(len(self.state_grid))
        policy = np.zeros(len(self.state_grid), dtype=int)
        
        for iteration in range(max_iter):
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{max_iter}")
            
            V_new = self.fixed_point_mapping(V)
            
            # Update policy
            for i, m in enumerate(self.state_grid):
                replace_value = -self.env.parameters['replace_cost'] + self.env.parameters['beta'] * V[0]
                keep_value = -self.env.maintenance_cost(m)
                for jump, prob in enumerate(self.poisson_probs):
                    next_m = min(m + jump * 1000, self.env.parameters['max_mileage'])
                    next_idx = self.get_state_index(next_m)
                    keep_value += self.env.parameters['beta'] * prob * V[next_idx]
                policy[i] = 1 if replace_value > keep_value else 0
            
            # Check convergence
            diff = np.max(np.abs(V_new - V))
            if iteration % 100 == 0:
                print(f"Current difference: {diff:.6f}")
            
            if diff < tol:
                print(f"Converged after {iteration + 1} iterations with difference {diff:.6f}")
                V = V_new
                break
                
            V = V_new
        
        if iteration == max_iter - 1:
            print(f"Warning: Did not converge after {max_iter} iterations. Final difference: {diff:.6f}")
        
        self.value_function = V
        self.policy = policy
        return V, policy
    
    def choice_probabilities(self, state: int, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute choice probabilities using the logit formula.
        P(a|s) = exp(v(s,a)/ε) / Σ_a' exp(v(s,a')/ε)
        """
        state_idx = self.get_state_index(state)
        
        # Value of replacing
        replace_value = -self.env.parameters['replace_cost'] + self.env.parameters['beta'] * self.value_function[0]
        
        # Value of not replacing
        keep_value = -self.env.maintenance_cost(state)
        for jump, prob in enumerate(self.poisson_probs):
            next_m = min(state + jump * 1000, self.env.parameters['max_mileage'])
            next_idx = self.get_state_index(next_m)
            keep_value += self.env.parameters['beta'] * prob * self.value_function[next_idx]
        
        # Compute logit probabilities with numerical stability
        values = np.array([keep_value, replace_value])
        values_centered = values - np.max(values)
        exp_values = np.exp(values_centered / epsilon)
        probs = exp_values / np.sum(exp_values)
        
        # Ensure probabilities are valid
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)  # Renormalize
        
        return probs
    
    def log_likelihood(self, data: List[Tuple[int, int]], epsilon: float = 1e-6) -> float:
        """
        Compute the log-likelihood of the observed decisions.
        Uses log-sum-exp trick for numerical stability.
        
        Args:
            data: List of (state, action) tuples
            epsilon: Scale parameter for the logit formula
            
        Returns:
            Log-likelihood value
        """
        if self.value_function is None:
            self.find_fixed_point()
        
        ll = 0.0
        valid_observations = 0
        
        for state, action in data:
            try:
                probs = self.choice_probabilities(state, epsilon)
                # Convert action to index (0 for keep, 1 for replace)
                action_idx = int(action)
                if action_idx >= len(probs):
                    print(f"Warning: Action {action_idx} out of bounds for state {state}")
                    continue
                
                # Use log-sum-exp trick for numerical stability
                log_prob = np.log(probs[action_idx])
                if not np.isnan(log_prob) and not np.isinf(log_prob):
                    ll += log_prob
                    valid_observations += 1
                else:
                    print(f"Warning: Invalid log probability for state {state}, action {action}")
            except Exception as e:
                print(f"Warning: Error processing state {state}, action {action}: {str(e)}")
                continue
        
        if valid_observations == 0:
            print("Warning: No valid observations for log-likelihood computation")
            return -np.inf
            
        return ll / valid_observations  # Return average log-likelihood
    
    def estimate(self, data: List[Tuple[int, int]], 
                initial_params: Dict[str, float] = None,
                epsilon: float = 1e-6) -> Dict[str, Any]:
        """
        Estimate model parameters using Fixed Point Maximum Likelihood.
        
        Args:
            data: List of (state, action) tuples
            initial_params: Initial parameter values
            epsilon: Scale parameter for the logit formula
            
        Returns:
            Dictionary containing estimation results
        """
        print("Starting FPML estimation...")
        if initial_params is None:
            initial_params = {
                'replace_cost': 500.0,
                'maintenance_cost_base': 20.0,
                'maintenance_cost_slope': 1.0
            }
        
        def objective(params):
            print(f"\nTrying parameters: replace_cost={params[0]:.2f}, base={params[1]:.2f}, slope={params[2]:.2f}")
            # Update parameters
            self.env.parameters['replace_cost'] = params[0]
            self.env.parameters['maintenance_cost_base'] = params[1]
            self.env.parameters['maintenance_cost_slope'] = params[2]
            
            # Find fixed point with tighter convergence criteria
            print("Finding fixed point...")
            self.find_fixed_point(tol=1e-6, max_iter=1000)  # Reduced from 2000
            
            # Compute log-likelihood
            print("Computing log-likelihood...")
            ll = self.log_likelihood(data, epsilon)
            
            if np.isnan(ll) or np.isinf(ll):
                print("Warning: Invalid log-likelihood value")
                return 1e10  # Return large value for invalid likelihoods
            
            print(f"Log-likelihood: {ll:.2f}")
            return -ll  # Minimize negative log-likelihood
        
        # Initial guess and bounds
        x0 = [
            initial_params['replace_cost'],
            initial_params['maintenance_cost_base'],
            initial_params['maintenance_cost_slope']
        ]
        
        # Adjust bounds based on beta
        if self.env.parameters['beta'] > 0.9:  # High discount factor
            bounds = [
                (300, 1000),    # replace_cost
                (10, 50),       # maintenance_cost_base
                (0.5, 5.0)      # maintenance_cost_slope
            ]
        else:  # Low discount factor
            bounds = [
                (100, 500),     # replace_cost
                (5, 30),        # maintenance_cost_base
                (0.1, 2.0)      # maintenance_cost_slope
            ]
        
        print("\nStarting optimization...")
        # Try single starting point first
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}  # Reduced from 200 iterations
            )
            
            if result.success:
                print("\nOptimization successful!")
                print(f"Final parameters: replace_cost={result.x[0]:.2f}, base={result.x[1]:.2f}, slope={result.x[2]:.2f}")
                print(f"Final log-likelihood: {-result.fun:.2f}")
                
                # Update parameters with optimal values
                self.env.parameters['replace_cost'] = result.x[0]
                self.env.parameters['maintenance_cost_base'] = result.x[1]
                self.env.parameters['maintenance_cost_slope'] = result.x[2]
                
                # Find final fixed point
                print("\nComputing final value function and policy...")
                self.find_fixed_point(tol=1e-6, max_iter=1000)
                
                return {
                    'success': True,
                    'message': result.message,
                    'estimated_parameters': {
                        'replace_cost': result.x[0],
                        'maintenance_cost_base': result.x[1],
                        'maintenance_cost_slope': result.x[2]
                    },
                    'log_likelihood': -result.fun,
                    'value_function': self.value_function,
                    'policy': self.policy
                }
            else:
                print(f"\nOptimization failed: {result.message}")
                return {
                    'success': False,
                    'message': result.message,
                    'estimated_parameters': initial_params,
                    'log_likelihood': -np.inf,
                    'value_function': None,
                    'policy': None
                }
                
        except Exception as e:
            print(f"\nError during optimization: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'estimated_parameters': initial_params,
                'log_likelihood': -np.inf,
                'value_function': None,
                'policy': None
            } 