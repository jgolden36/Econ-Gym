#!/usr/bin/env python3
"""
Simple test script to verify JAX setup and environment imports
"""

print("Testing JAX setup...")

try:
    import jax
    import jax.numpy as jnp
    print(f"✓ JAX imported successfully (version {jax.__version__})")
    print(f"✓ JAX devices: {jax.devices()}")
except ImportError as e:
    print(f"✗ Failed to import JAX: {e}")
    exit(1)

print("\nTesting environment imports...")

try:
    from envs.BBL_env_jax import BBLGameEnvJAX
    print("✓ BBL JAX environment imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BBL JAX: {e}")

try:
    from envs.zurcher_env_jax import ZurcherEnvJAX
    print("✓ Zurcher JAX environment imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Zurcher JAX: {e}")

print("\nTesting basic functionality...")

try:
    # Test BBL environment
    env_bbl = BBLGameEnvJAX(n_firms=3, beta=0.95)
    state, _ = env_bbl.reset(seed=42)
    print(f"✓ BBL environment created and reset: state shape {state.shape}")
    
    # Test single step
    action = [1, 0, 2]
    next_state, rewards, done, truncated, _ = env_bbl.step(action)
    print(f"✓ BBL step completed: rewards shape {rewards.shape}")
    
except Exception as e:
    print(f"✗ BBL environment test failed: {e}")

try:
    # Test Zurcher environment
    env_zurcher = ZurcherEnvJAX(max_mileage=50, replace_cost=100, beta=0.99)
    state, _ = env_zurcher.reset(seed=42)
    print(f"✓ Zurcher environment created and reset: state {state}")
    
    # Test single step
    action = 0
    next_state, reward, done, truncated, _ = env_zurcher.step(action)
    print(f"✓ Zurcher step completed: reward {reward}")
    
except Exception as e:
    print(f"✗ Zurcher environment test failed: {e}")

print("\nTesting JAX compilation...")

try:
    @jax.jit
    def simple_func(x):
        return jnp.sum(x ** 2)
    
    x = jnp.array([1.0, 2.0, 3.0])
    result = simple_func(x)
    print(f"✓ JAX JIT compilation works: result = {result}")
    
except Exception as e:
    print(f"✗ JAX compilation test failed: {e}")

print("\nAll tests completed!") 