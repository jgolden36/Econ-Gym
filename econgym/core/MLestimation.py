import gym
import numpy as np
from scipy.optimize import minimize
import functools

# Optionally import Numba and Jax.
try:
    from numba import njit
except ImportError:
    njit = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

def _objective(params, likelihood_function, data):
    """
    Computes the negative log likelihood.
    """
    return -likelihood_function(params, data)

class MLestimation:
    """
    A maximum likelihood estimation (MLE) solver for generic Gym environments.
    
    It either uses an existing MLE method from the environment or applies a generic
    optimization routine to maximize the likelihood. Optionally, it can accelerate the
    likelihood computation using Numba or Jax.
    """
    
    def __init__(self, env, likelihood_function=None, data=None, acceleration=None):
        """
        :param env: A gym.Env instance representing the environment.
        :param likelihood_function: Optional. A function f(params, data) -> float returning the log likelihood.
                                    If not provided, the constructor will attempt to use env.log_likelihood.
        :param data: Optional. Pre-collected data (e.g., trajectories) for estimation.
        :param acceleration: Optional. One of {None, 'numba', 'jax'} to accelerate likelihood computation.
        """
        self.env = env
        self.likelihood_function = likelihood_function or getattr(env, "log_likelihood", None)
        if self.likelihood_function is None:
            raise ValueError("No likelihood function provided and the environment does not have a 'log_likelihood' method.")
        self.data = data
        self.acceleration = acceleration
        if self.acceleration == 'numba' and njit is None:
            raise ImportError("Numba is not installed but acceleration='numba' was specified.")
        if self.acceleration == 'jax' and jax is None:
            raise ImportError("Jax is not installed but acceleration='jax' was specified.")

    def collect_data(self, num_episodes=100, max_steps=100):
        """
        Collect data by running episodes with random actions.
        The data is stored as a list of episodes, where each episode is a list of
        tuples: (observation, action, reward, next_observation, done).
        
        :param num_episodes: Number of episodes to simulate.
        :param max_steps: Maximum steps per episode.
        :return: The collected data.
        """
        data = []
        for _ in range(num_episodes):
            episode_data = []
            obs = self.env.reset()
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                next_obs, reward, done, info = self.env.step(action)
                episode_data.append((obs, action, reward, next_obs, done))
                obs = next_obs
                if done:
                    break
            data.append(episode_data)
        self.data = data
        return data

    def solve(self, initial_params, **kwargs):
        """
        Solve for the maximum likelihood estimate of the parameters.
        
        :param initial_params: An initial guess for the parameters as a NumPy array.
        :param kwargs: Additional keyword arguments for scipy.optimize.minimize.
        :return: A tuple (estimated_params, optimization_result).
        """
        # Use the environment's own method if available.
        if hasattr(self.env, "maximum_likelihood_estimate"):
            return self.env.maximum_likelihood_estimate(initial_params, data=self.data)
        
        if self.data is None:
            self.collect_data()
        
        # Create a partial objective function.
        objective = functools.partial(_objective, likelihood_function=self.likelihood_function, data=self.data)
        
        # Optionally compile the objective function.
        if self.acceleration == 'numba':
            objective = njit(objective)
        elif self.acceleration == 'jax':
            objective = jax.jit(objective)
            # Wrap to ensure the result is a plain Python float.
            objective = lambda params: float(objective(params))
        
        # Minimize the negative log likelihood.
        res = minimize(lambda params: objective(params), initial_params, **kwargs)
        return res.x, res

# ===== Example Usage =====
if __name__ == "__main__":
    import numpy as np
    from examples.ml_estimation_example import SimpleEconEnv
    
    def run_estimation_example():
        """
        Run a demonstration of maximum likelihood estimation on a simple economic environment.
        """
        try:
            # Create environment with known parameter value
            env = SimpleEconEnv()
            true_theta = 0.5  # Set a known true parameter value
            env.parameters = {'theta': true_theta}
            
            print("Starting Maximum Likelihood Estimation Example")
            print(f"True parameter value: {true_theta}")
            
            # Create ML estimation solver
            # Try to use acceleration if available
            acceleration = None
            if njit is not None:
                print("Using Numba acceleration")
                acceleration = 'numba'
            elif jax is not None:
                print("Using JAX acceleration")
                acceleration = 'jax'
            else:
                print("Running without acceleration")
            
            mle_solver = MLestimation(env, acceleration=acceleration)
            
            # Collect data
            print("\nCollecting data...")
            data = mle_solver.collect_data(num_episodes=100, max_steps=50)
            print(f"Collected {len(data)} episodes")
            
            # Initial parameter guess
            initial_params = np.array([0.0])
            
            # Run estimation with different methods
            methods = ['L-BFGS-B', 'Nelder-Mead']
            best_result = None
            best_params = None
            best_objective = float('inf')
            
            print("\nTrying different optimization methods...")
            for method in methods:
                print(f"\nTrying method: {method}")
                try:
                    estimated_params, opt_result = mle_solver.solve(
                        initial_params,
                        method=method,
                        options={'disp': True}
                    )
                    
                    if opt_result.fun < best_objective:
                        best_objective = opt_result.fun
                        best_params = estimated_params
                        best_result = opt_result
                        
                    print(f"Method {method} results:")
                    print(f"Estimated parameter: {estimated_params[0]:.4f}")
                    print(f"Objective value: {opt_result.fun:.4f}")
                    print(f"Success: {opt_result.success}")
                    
                except Exception as e:
                    print(f"Method {method} failed: {str(e)}")
            
            if best_result is not None:
                print("\nBest estimation results:")
                print(f"True parameter value: {true_theta}")
                print(f"Estimated parameter value: {best_params[0]:.4f}")
                print(f"Final objective value: {best_objective:.4f}")
                print(f"Optimization success: {best_result.success}")
                
                # Test the estimated parameters
                env.parameters = {'theta': best_params[0]}
                test_states, test_rewards = env.simulate(n_periods=1000)
                print(f"\nTest simulation results:")
                print(f"Mean reward: {np.mean(test_rewards):.4f}")
                print(f"Reward std: {np.std(test_rewards):.4f}")
            else:
                print("\nAll optimization methods failed")
                
        except Exception as e:
            print(f"Error in estimation example: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the example
    run_estimation_example()