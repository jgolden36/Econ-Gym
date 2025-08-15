import time
import jax
from envs.zurcher_env_jax import ZurcherEnvJAX
from envs.BBL_env_jax import BBLGameEnvJAX

print("Quick JAX Test")
print("=" * 30)

# Test Zurcher JAX
print("Testing Zurcher JAX...")
env = ZurcherEnvJAX(max_mileage=20, replace_cost=50)
state, _ = env.reset()
print(f"✓ Zurcher reset: {state}")

next_state, reward, done, trunc, info = env.step(0)
print(f"✓ Zurcher step: state={next_state}, reward={reward:.2f}")

# Test batch simulation
start = time.time()
rng_key = jax.random.PRNGKey(42)
trajectories = env.batch_simulate_jax(rng_key, 10, 5)
batch_time = time.time() - start
print(f"✓ Batch simulation (10x5): {batch_time:.4f} seconds")

# Test BBL JAX
print("\nTesting BBL JAX...")
env_bbl = BBLGameEnvJAX(n_firms=2)
state, _ = env_bbl.reset()
print(f"✓ BBL reset: shape={state.shape}")

next_state, rewards, done, trunc, info = env_bbl.step([0, 1])
print(f"✓ BBL step: rewards={rewards}")

print("\n✓ All tests passed!")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}") 