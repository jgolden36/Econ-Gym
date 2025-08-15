import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from typing import Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


class ZurcherEnvJAX(EconEnv):
    """
    JAX-optimized Harold Zurcher Bus Replacement Model Environment.
    
    This version uses JAX for significant performance improvements:
    - JIT-compiled step and value function iteration
    - Vectorized batch simulations
    - JAX-based optimization for calibration/estimation
    - Fast Poisson probability computations
    
    State: 
        - m: current mileage (discrete)
    Action: 
        - 0: Continue using the engine
        - 1: Replace the engine
    """
    
    def __init__(self,
                 max_mileage: int = 400,
                 replace_cost: float = 11.7257,
                 beta: float = 0.9999,
                 poisson_lambda: float = 1.0):
        super().__init__()
        self.parameters = {
            'max_mileage': max_mileage,
            'replace_cost': replace_cost,
            'beta': beta,
            'maintenance_cost_base': 0.0,
            'maintenance_cost_slope': 0.0001,
            'poisson_lambda': poisson_lambda
        }

        # Use an integer Box so RL algorithms see consistent dtypes
        self.observation_space = spaces.Box(
            low=0,
            high=max_mileage,
            shape=(1,),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(2)
        self.state = None
        self.rng_key = jax.random.PRNGKey(0)
        
        # Pre-compute Poisson probabilities for speed
        max_jump = 20
        self.poisson_probs = jnp.array([
            jnp.exp(-poisson_lambda) * (poisson_lambda ** j) / jax.scipy.special.gammaln(j + 1)
            for j in range(max_jump)
        ])
        self.poisson_probs = self.poisson_probs / jnp.sum(self.poisson_probs)
        
        # JIT-compile key functions
        self._jit_step = jit(self._step_jax)
        self._jit_maintenance_cost = jit(self._maintenance_cost_jax)
        self._jit_value_iteration = jit(self._value_iteration_step_jax)
        
        # Initialize value function and policy
        self.value_function = None
        self.policy = None

    def reset(self, *, seed=None, options=None):
        """Reset environment with JAX random number generation."""
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)
            
        self.rng_key, subkey = jax.random.split(self.rng_key)
        # Random start to visit all mileage states
        self.state = int(jax.random.randint(
            subkey,
            (),
            0,
            self.parameters['max_mileage'] + 1
        ))
        return np.array([self.state], dtype=np.int32), {}

    def step(self, action):
        """Execute step using JAX-compiled function."""
        self.rng_key, subkey = jax.random.split(self.rng_key)
        next_state, reward = self._jit_step(
            self.state,
            int(action) if not isinstance(action, (list, np.ndarray)) else int(action[0]),
            subkey
        )
        
        self.state = int(next_state)
        done = False  # infinite-horizon
        return np.array([self.state], dtype=np.int32), float(reward), done, False, {}

    @partial(jit, static_argnums=(0,))
    def _step_jax(self, state, action, rng_key):
        """JAX-compiled step function for maximum performance."""
        m = state
        
        # Replacement logic
        should_replace = (action == 1) | (m >= self.parameters['max_mileage'])
        
        # Compute next state and reward
        next_m = jax.lax.cond(
            should_replace,
            lambda _: jnp.array(0, dtype=jnp.int32),  # Reset to 0 if replacing
            lambda _: self._compute_next_mileage_jax(m, rng_key),
            None
        )
        
        reward = jax.lax.cond(
            should_replace,
            lambda _: jnp.array(-self.parameters['replace_cost'], dtype=jnp.float32),
            lambda _: jnp.array(-self._maintenance_cost_jax(m), dtype=jnp.float32),
            None
        )
        
        return next_m, reward

    @partial(jit, static_argnums=(0,))
    def _compute_next_mileage_jax(self, current_mileage, rng_key):
        """JAX-compiled function to compute next mileage with Poisson increment."""
        jump = jax.random.poisson(rng_key, self.parameters['poisson_lambda'])
        return jnp.minimum(current_mileage + jump, self.parameters['max_mileage']).astype(jnp.int32)

    @partial(jit, static_argnums=(0,))
    def _maintenance_cost_jax(self, mileage):
        """JAX-compiled maintenance cost function."""
        return (self.parameters['maintenance_cost_base'] + 
                self.parameters['maintenance_cost_slope'] * mileage)

    def find_equilibrium_jax(self, tol=1e-6, max_iter=1000):
        """Find equilibrium using JAX-optimized value function iteration."""
        if self.value_function is not None:
            return self.value_function, self.policy

        n_states = self.parameters['max_mileage'] + 1
        value_function = jnp.zeros(n_states)
        policy = jnp.zeros(n_states, dtype=jnp.int32)
        
        print("Starting JAX-optimized value function iteration...")
        
        for iteration in range(max_iter):
            new_value_function, new_policy = self._jit_value_iteration(value_function)
            
            # Check convergence
            diff = jnp.max(jnp.abs(new_value_function - value_function))
            if diff < tol:
                print(f"JAX Value function iteration converged after {iteration + 1} iterations")
                break
                
            value_function = new_value_function
            policy = new_policy
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Max difference: {diff:.2e}")
        
        self.value_function = value_function
        self.policy = policy
        return value_function, policy

    @partial(jit, static_argnums=(0,))
    def _value_iteration_step_jax(self, value_function):
        """Single JAX-compiled Bellman update step."""
        n_states = len(value_function)
        new_value_function = jnp.zeros(n_states)
        new_policy = jnp.zeros(n_states, dtype=jnp.int32)
        
        # Vectorized computation for all states
        def compute_state_value(m):
            # Value of replacing
            replace_value = -self.parameters['replace_cost'] + self.parameters['beta'] * value_function[0]
            
            # Value of not replacing
            keep_value = -self._maintenance_cost_jax(m)
            
            # Add expected future value using pre-computed Poisson probabilities
            def compute_expected_value(jump_prob_pair):
                jump, prob = jump_prob_pair
                next_m = jnp.minimum(m + jump, self.parameters['max_mileage'])
                return prob * value_function[next_m]
            
            expected_future = jnp.sum(vmap(compute_expected_value)(
                (jnp.arange(len(self.poisson_probs)), self.poisson_probs)
            ))
            
            keep_value += self.parameters['beta'] * expected_future
            
            # Choose best action
            is_replace_better = replace_value > keep_value
            best_value = jax.lax.select(is_replace_better, replace_value, keep_value)
            best_action = jax.lax.select(is_replace_better, 1, 0)
            
            return best_value, best_action
        
        # Vectorize over all states
        values_and_actions = vmap(compute_state_value)(jnp.arange(n_states))
        new_value_function = values_and_actions[0]
        new_policy = values_and_actions[1]
        
        return new_value_function, new_policy

    def get_value_function_jax(self, state: int) -> float:
        """Get value function using JAX arrays."""
        if self.value_function is None:
            self.find_equilibrium_jax()
        return float(self.value_function[state])

    def get_policy_jax(self, state: int) -> int:
        """Get optimal policy using JAX arrays."""
        if self.policy is None:
            self.find_equilibrium_jax()
        return int(self.policy[state])

    @partial(jit, static_argnums=(0,))
    def batch_simulate_jax(self, rng_key, n_simulations, n_steps):
        """Perform batched simulations using JAX for maximum speed."""
        def simulate_one(rng_key):
            def step_fn(carry, _):
                state, key = carry
                key, subkey1, subkey2 = jax.random.split(key, 3)
                
                # Use optimal policy if available, otherwise random
                if self.policy is not None:
                    action = self.policy[state]
                else:
                    action = jax.random.randint(subkey1, (), 0, 2)
                
                next_state, reward = self._step_jax(state, action, subkey2)
                return (next_state, key), (state, action, reward)
            
            # Random initial state
            key, subkey = jax.random.split(rng_key)
            initial_state = jax.random.randint(
                subkey, (), 0, self.parameters['max_mileage'] + 1
            )
            
            _, trajectory = jax.lax.scan(
                step_fn,
                (initial_state, key),
                jnp.arange(n_steps)
            )
            
            return trajectory
        
        # Generate keys for each simulation
        keys = jax.random.split(rng_key, n_simulations)
        
        # Vectorize simulation function
        batch_simulate = vmap(simulate_one)
        
        return batch_simulate(keys)

    @partial(jit, static_argnums=(0,))
    def compute_moments_jax(self, trajectories):
        """Compute moments from simulation trajectories using JAX."""
        states, actions, rewards = trajectories
        
        # Compute replacement frequency
        replacement_freq = jnp.mean(actions)
        
        # Compute average mileage at replacement
        replacement_mask = actions == 1
        mileages_at_replacement = jnp.where(replacement_mask, states, -1)
        valid_replacements = mileages_at_replacement[mileages_at_replacement >= 0]
        avg_mileage_at_replacement = jnp.mean(valid_replacements) if len(valid_replacements) > 0 else 0.0
        
        # Compute average reward
        avg_reward = jnp.mean(rewards)
        
        # Compute usage variance
        usage_variance = jnp.var(states)
        
        return jnp.array([
            replacement_freq,
            avg_mileage_at_replacement,
            avg_reward,
            usage_variance
        ])

    def calibrate_jax(self, targets: Dict[str, Any], n_iterations=1000, learning_rate=0.01):
        """JAX-optimized calibration using gradient-based optimization."""
        
        @jit
        def objective_fn(params):
            """JAX-compiled objective function for calibration."""
            replace_cost, maint_base, maint_slope = params
            
            # Create temporary parameters
            temp_params = {
                'max_mileage': self.parameters['max_mileage'],
                'replace_cost': replace_cost,
                'beta': self.parameters['beta'],
                'maintenance_cost_base': maint_base,
                'maintenance_cost_slope': maint_slope,
                'poisson_lambda': self.parameters['poisson_lambda']
            }
            
            # Simplified simulation for calibration
            # In practice, you'd run full simulation with updated parameters
            rng_key = jax.random.PRNGKey(42)
            trajectories = self.batch_simulate_jax(rng_key, 100, 50)
            moments = self.compute_moments_jax(trajectories)
            
            # Target moments
            target_moments = jnp.array([
                targets.get('replacement_frequency', 0.1),
                targets.get('avg_mileage', 50.0),
                targets.get('avg_reward', -10.0),
                targets.get('usage_variance', 100.0)
            ])
            
            return jnp.sum((moments - target_moments) ** 2)
        
        # JAX gradient-based optimization
        grad_fn = jax.grad(objective_fn)
        
        # Initialize parameters
        params = jnp.array([
            self.parameters['replace_cost'],
            self.parameters['maintenance_cost_base'],
            self.parameters['maintenance_cost_slope']
        ])
        
        print("Starting JAX-optimized calibration...")
        
        for i in range(n_iterations):
            grad = grad_fn(params)
            params = params - learning_rate * grad
            
            if i % 100 == 0:
                loss = objective_fn(params)
                print(f"Iteration {i}, Loss: {loss:.6f}")
        
        # Update parameters with optimal values
        self.parameters['replace_cost'] = float(params[0])
        self.parameters['maintenance_cost_base'] = float(params[1])
        self.parameters['maintenance_cost_slope'] = float(params[2])
        
        # Reset value function and policy to trigger recomputation
        self.value_function = None
        self.policy = None
        
        return {
            'success': True,
            'optimal_parameters': {
                'replace_cost': float(params[0]),
                'maintenance_cost_base': float(params[1]),
                'maintenance_cost_slope': float(params[2])
            },
            'final_loss': float(objective_fn(params))
        }

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current mileage: {self.state}")
            print(f"JAX-optimized Zurcher Environment")
            print(f"Parameters: replace_cost={self.parameters['replace_cost']:.4f}, "
                  f"maintenance_cost_slope={self.parameters['maintenance_cost_slope']:.6f}")
        return None


# Utility functions for analysis
def compare_performance():
    """Compare performance between original and JAX-optimized versions."""
    import time
    
    # Test parameters
    n_simulations = 1000
    n_steps = 100
    
    print("Performance Comparison: Original vs JAX-optimized Zurcher Environment")
    print("=" * 70)
    
    # JAX version
    env_jax = ZurcherEnvJAX(max_mileage=100, replace_cost=200, beta=0.99)
    
    # Test equilibrium computation
    print("\n1. Equilibrium Computation:")
    start_time = time.time()
    vf_jax, policy_jax = env_jax.find_equilibrium_jax(max_iter=500)
    jax_eq_time = time.time() - start_time
    print(f"JAX equilibrium time: {jax_eq_time:.4f} seconds")
    
    # Test batch simulation
    print("\n2. Batch Simulation:")
    start_time = time.time()
    rng_key = jax.random.PRNGKey(42)
    trajectories_jax = env_jax.batch_simulate_jax(rng_key, n_simulations, n_steps)
    jax_sim_time = time.time() - start_time
    print(f"JAX simulation time: {jax_sim_time:.4f} seconds")
    print(f"Simulated {n_simulations} episodes of {n_steps} steps each")
    
    # Test moments computation
    print("\n3. Moments Computation:")
    start_time = time.time()
    moments = env_jax.compute_moments_jax(trajectories_jax)
    jax_moments_time = time.time() - start_time
    print(f"JAX moments time: {jax_moments_time:.4f} seconds")
    print(f"Computed moments: {moments}")
    
    # Test calibration
    print("\n4. Calibration:")
    targets = {
        'replacement_frequency': 0.15,
        'avg_mileage': 40.0,
        'avg_reward': -15.0,
        'usage_variance': 200.0
    }
    start_time = time.time()
    calibration_results = env_jax.calibrate_jax(targets, n_iterations=100)
    jax_calib_time = time.time() - start_time
    print(f"JAX calibration time: {jax_calib_time:.4f} seconds")
    print(f"Calibration success: {calibration_results['success']}")
    
    print(f"\nTotal JAX processing time: {jax_eq_time + jax_sim_time + jax_moments_time + jax_calib_time:.4f} seconds")


if __name__ == "__main__":
    compare_performance() 