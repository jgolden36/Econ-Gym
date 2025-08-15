import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from typing import Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


class BBLGameEnvJAX(EconEnv):
    """
    A JAX-optimized version of the BBL game environment for improved performance.
    
    This environment models a market with multiple competing firms using JAX for 
    fast JIT-compiled computations. Key optimizations:
    - JIT-compiled step function
    - Vectorized computations for multiple firms
    - JAX-based value function iteration
    - Batched simulations
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
        self.observation_space = spaces.Box(
            low=self.parameters['state_min'],
            high=self.parameters['state_max'],
            shape=(self.parameters['n_firms'], self.parameters['state_dim']),
            dtype=np.float32
        )
        
        # Each firm can choose from a discrete set of price levels.
        self.action_space = spaces.MultiDiscrete([self.parameters['price_levels']] * self.parameters['n_firms'])
        
        # Initialize state
        self.state = None
        self.rng_key = jax.random.PRNGKey(0)
        
        # JIT-compile key functions
        self._jit_step = jit(self._step_jax)
        self._jit_compute_rewards = jit(self._compute_rewards_jax)
        self._jit_update_states = jit(self._update_states_jax)
        
        # Value function iteration components
        self.value_function = None
        self.policy = None

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)
            
        self.rng_key, subkey = jax.random.split(self.rng_key)
        # Initialize each firm's state randomly using JAX
        self.state = jax.random.uniform(
            subkey,
            shape=(self.parameters['n_firms'], self.parameters['state_dim']),
            minval=self.parameters['state_min'],
            maxval=self.parameters['state_max']
        )
        return np.array(self.state), {}

    def step(self, action):
        """Execute one time step using JAX-compiled function."""
        self.rng_key, subkey = jax.random.split(self.rng_key)
        next_state, rewards = self._jit_step(
            jnp.array(self.state), 
            jnp.array(action), 
            subkey
        )
        
        self.state = next_state
        done = False
        truncated = False
        return np.array(next_state), np.array(rewards), done, truncated, {}

    @partial(jit, static_argnums=(0,))
    def _step_jax(self, state, action, rng_key):
        """JAX-compiled step function for maximum performance."""
        # Compute rewards
        rewards = self._compute_rewards_jax(state, action)
        
        # Update states
        next_state = self._update_states_jax(state, action, rng_key)
        
        return next_state, rewards

    @partial(jit, static_argnums=(0,))
    def _compute_rewards_jax(self, state, action):
        """JAX-compiled reward computation."""
        # Calculate market demand based on average price
        avg_price = jnp.mean(action) / (self.parameters['price_levels'] - 1)
        market_demand = self.parameters['revenue_multiplier'] * (1 - avg_price)
        
        # Vectorized computation for all firms
        prices = action / (self.parameters['price_levels'] - 1)
        market_shares = jnp.ones(self.parameters['n_firms']) / self.parameters['n_firms']
        revenues = market_demand * market_shares * prices
        
        # Vectorized cost computation
        costs = self.parameters['cost_multiplier'] * state[:, 0] * prices
        
        return revenues - costs

    @partial(jit, static_argnums=(0,))
    def _update_states_jax(self, state, action, rng_key):
        """JAX-compiled state update function."""
        def update_single_firm(carry, x):
            i, state_val, action_val, rng_key = x
            
            adjustment = jax.lax.cond(
                action_val == 0,
                lambda _: self.parameters['state_adjustment']['low_price'],
                lambda _: jax.lax.cond(
                    action_val == self.parameters['price_levels'] - 1,
                    lambda _: self.parameters['state_adjustment']['high_price'],
                    lambda _: jax.random.normal(rng_key) * self.parameters['state_adjustment']['medium_price_std'],
                    None
                ),
                None
            )
            
            new_state = jnp.clip(
                state_val + adjustment,
                self.parameters['state_min'],
                self.parameters['state_max']
            )
            return carry, new_state
        
        # Split RNG for each firm
        rng_keys = jax.random.split(rng_key, self.parameters['n_firms'])
        
        # Use scan for vectorized state updates
        _, new_states = jax.lax.scan(
            update_single_firm,
            None,
            (jnp.arange(self.parameters['n_firms']), state[:, 0], action, rng_keys)
        )
        
        return new_states.reshape(state.shape)

    @partial(jit, static_argnums=(0,))
    def batch_simulate_jax(self, rng_key, n_simulations, n_steps):
        """Perform batched simulations using JAX for maximum speed."""
        def simulate_one(rng_key):
            def step_fn(carry, x):
                state, key = carry
                key, subkey = jax.random.split(key)
                
                # Use random policy for simulation
                action = jax.random.randint(
                    subkey,
                    (self.parameters['n_firms'],),
                    0,
                    self.parameters['price_levels']
                )
                
                next_state, rewards = self._step_jax(state, action, subkey)
                return (next_state, key), (state, action, rewards)
            
            # Initial state
            key, subkey = jax.random.split(rng_key)
            initial_state = jax.random.uniform(
                subkey,
                shape=(self.parameters['n_firms'], self.parameters['state_dim']),
                minval=self.parameters['state_min'],
                maxval=self.parameters['state_max']
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

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current states: {self.state}")
            print(f"JAX-optimized BBL Environment")
            print(f"Parameters: revenue_multiplier={self.parameters['revenue_multiplier']:.2f}, "
                  f"cost_multiplier={self.parameters['cost_multiplier']:.2f}")
        return None


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Create JAX-optimized environment
    env_jax = BBLGameEnvJAX(n_firms=5, beta=0.95)
    
    print("Testing JAX-optimized BBL Environment")
    print("=" * 50)
    
    # Test basic functionality
    state, _ = env_jax.reset(seed=42)
    print(f"Initial state shape: {state.shape}")
    
    # Test single step
    action = [1, 0, 2, 1, 0]  # Example actions for 5 firms
    next_state, rewards, done, truncated, _ = env_jax.step(action)
    print(f"Rewards: {rewards}")
    print(f"Next state shape: {next_state.shape}")
    
    # Benchmark batched simulation
    print("\nBenchmarking batched simulation:")
    start_time = time.time()
    trajectories = env_jax.batch_simulate_jax(n_simulations=1000, n_steps=100)
    end_time = time.time()
    
    print(f"Simulated {1000} episodes of {100} steps each")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Trajectory shape: {trajectories[0].shape}")  # (states, actions, rewards)
    
    # Test equilibrium finding
    print("\nFinding equilibrium with JAX optimization:")
    start_time = time.time()
    policy, value_fn = env_jax.find_equilibrium_jax(n_grid_points=20, max_iter=100)
    end_time = time.time()
    
    print(f"Equilibrium computation time: {end_time - start_time:.4f} seconds")
    print(f"Policy shape: {policy.shape}")
    print(f"Value function shape: {value_fn.shape}") 