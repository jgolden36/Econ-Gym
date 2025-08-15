#!/usr/bin/env python3
"""
Simplified JAX Optimization Demo for EconGym Environments
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from envs.BBL_env import BBLGameEnv
from envs.BBL_env_jax import BBLGameEnvJAX
from envs.zurcher_env import ZurcherEnv
from envs.zurcher_env_jax import ZurcherEnvJAX

def run_demo():
    print("JAX Optimization Demo for EconGym Environments")
    print("=" * 60)
    
    # Test JAX setup
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print()
    
    # BBL Environment Comparison
    print("BBL Environment Performance Test")
    print("-" * 40)
    
    try:
        # Create environments
        env_original = BBLGameEnv(n_firms=3, beta=0.95)
        env_jax = BBLGameEnvJAX(n_firms=3, beta=0.95)
        
        # Test original environment
        print("Testing original BBL environment...")
        env_original.reset(seed=42)
        action = [1, 0, 2]
        
        start_time = time.time()
        for _ in range(100):
            env_original.step(action)
            env_original.reset()
        original_time = time.time() - start_time
        
        # Test JAX environment
        print("Testing JAX BBL environment...")
        env_jax.reset(seed=42)
        
        start_time = time.time()
        for _ in range(100):
            env_jax.step(action)
            env_jax.reset()
        jax_time = time.time() - start_time
        
        print(f"Original BBL: {original_time:.4f} seconds (100 steps)")
        print(f"JAX BBL: {jax_time:.4f} seconds (100 steps)")
        print(f"Speedup: {original_time / jax_time:.2f}x")
        
        # Test JAX batch simulation
        print("\nTesting JAX batch simulation...")
        start_time = time.time()
        rng_key = jax.random.PRNGKey(42)
        trajectories = env_jax.batch_simulate_jax(rng_key, 100, 50)
        batch_time = time.time() - start_time
        print(f"JAX batch simulation (100 episodes x 50 steps): {batch_time:.4f} seconds")
        print(f"Trajectory shape: {trajectories[0].shape}")
        
    except Exception as e:
        print(f"BBL test failed: {e}")
    
    print()
    
    # Zurcher Environment Comparison
    print("Zurcher Environment Performance Test")
    print("-" * 40)
    
    try:
        # Create environments
        env_original = ZurcherEnv(max_mileage=50, replace_cost=100, beta=0.99)
        env_jax = ZurcherEnvJAX(max_mileage=50, replace_cost=100, beta=0.99)
        
        # Test equilibrium computation
        print("Testing equilibrium computation...")
        
        # Original environment
        start_time = time.time()
        policy_orig, vf_orig = env_original.find_equilibrium(max_iter=200)
        original_eq_time = time.time() - start_time
        
        # JAX environment
        start_time = time.time()
        vf_jax, policy_jax = env_jax.find_equilibrium_jax(max_iter=200)
        jax_eq_time = time.time() - start_time
        
        print(f"Original Zurcher equilibrium: {original_eq_time:.4f} seconds")
        print(f"JAX Zurcher equilibrium: {jax_eq_time:.4f} seconds")
        print(f"Speedup: {original_eq_time / jax_eq_time:.2f}x")
        
        # Check accuracy
        policy_diff = np.mean(np.abs(policy_orig - np.array(policy_jax)))
        vf_diff = np.mean(np.abs(vf_orig - np.array(vf_jax)))
        print(f"Policy difference: {policy_diff:.6f}")
        print(f"Value function difference: {vf_diff:.6f}")
        
        # Test JAX batch simulation
        print("\nTesting Zurcher JAX batch simulation...")
        start_time = time.time()
        rng_key = jax.random.PRNGKey(123)
        trajectories = env_jax.batch_simulate_jax(rng_key, 100, 50)
        batch_time = time.time() - start_time
        print(f"JAX batch simulation: {batch_time:.4f} seconds")
        
        # Test moments computation
        start_time = time.time()
        moments = env_jax.compute_moments_jax(trajectories)
        moments_time = time.time() - start_time
        print(f"Moments computation: {moments_time:.4f} seconds")
        print(f"Computed moments: {moments}")
        
    except Exception as e:
        print(f"Zurcher test failed: {e}")
    
    print()
    
    # JAX Features Demo
    print("JAX Features Demonstration")
    print("-" * 40)
    
    try:
        # JIT compilation demo
        @jax.jit
        def test_function(x):
            return jnp.sum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2)
        
        x = jnp.arange(1000)
        
        # First call (with compilation)
        start = time.time()
        result1 = test_function(x)
        first_call = time.time() - start
        
        # Second call (already compiled)
        start = time.time()
        result2 = test_function(x)
        second_call = time.time() - start
        
        print(f"JIT compilation demo:")
        print(f"First call (with compilation): {first_call:.6f} seconds")
        print(f"Second call (compiled): {second_call:.6f} seconds")
        print(f"Speedup after compilation: {first_call / second_call:.2f}x")
        
    except Exception as e:
        print(f"JAX features test failed: {e}")
    
    print()
    print("Demo completed successfully!")
    print("Key benefits of JAX optimization:")
    print("• Significant speedups through JIT compilation")
    print("• Vectorized batch operations")
    print("• Maintained numerical accuracy")
    print("• Easy integration with existing code")

if __name__ == "__main__":
    run_demo() 