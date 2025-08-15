"""
JAX Optimization Demo for EconGym Environments

This script demonstrates the performance improvements achieved by using JAX-optimized
versions of the BBL and Zurcher environments. It compares execution times and shows
how to use the JAX-optimized features.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from econgym.envs.BBL_env import BBLGameEnv
from econgym.envs.BBL_env_jax import BBLGameEnvJAX
from econgym.envs.zurcher_env import ZurcherEnv
from econgym.envs.zurcher_env_jax import ZurcherEnvJAX


def benchmark_bbl_environments():
    """Compare performance between original and JAX-optimized BBL environments."""
    print("=" * 60)
    print("BBL Environment Performance Comparison")
    print("=" * 60)
    
    # Test parameters
    n_firms = 5
    n_simulations = 500
    n_steps = 100
    
    # Create environments
    env_original = BBLGameEnv(n_firms=n_firms, beta=0.95)
    env_jax = BBLGameEnvJAX(n_firms=n_firms, beta=0.95)
    
    print(f"Testing with {n_firms} firms, {n_simulations} simulations, {n_steps} steps each")
    
    # Test single step performance
    print("\n1. Single Step Performance:")
    
    # Original environment
    env_original.reset(seed=42)
    action = np.random.randint(0, 3, size=n_firms)
    start_time = time.time()
    for _ in range(1000):
        env_original.step(action)
        env_original.reset()
    original_step_time = time.time() - start_time
    
    # JAX environment
    env_jax.reset(seed=42)
    start_time = time.time()
    for _ in range(1000):
        env_jax.step(action)
        env_jax.reset()
    jax_step_time = time.time() - start_time
    
    print(f"Original environment: {original_step_time:.4f} seconds (1000 steps)")
    print(f"JAX environment: {jax_step_time:.4f} seconds (1000 steps)")
    print(f"Speedup: {original_step_time / jax_step_time:.2f}x")
    
    # Test batch simulation (JAX only)
    print("\n2. Batch Simulation (JAX only):")
    start_time = time.time()
    rng_key = jax.random.PRNGKey(42)
    trajectories = env_jax.batch_simulate_jax(rng_key, n_simulations, n_steps)
    batch_sim_time = time.time() - start_time
    
    print(f"JAX batch simulation: {batch_sim_time:.4f} seconds")
    print(f"Trajectory shape: {trajectories[0].shape}")  # (states, actions, rewards)
    print(f"Total data points: {n_simulations * n_steps * n_firms}")
    
    return {
        'original_step_time': original_step_time,
        'jax_step_time': jax_step_time,
        'batch_sim_time': batch_sim_time
    }


def benchmark_zurcher_environments():
    """Compare performance between original and JAX-optimized Zurcher environments."""
    print("\n" + "=" * 60)
    print("Zurcher Environment Performance Comparison")
    print("=" * 60)
    
    # Test parameters
    max_mileage = 100
    n_simulations = 1000
    n_steps = 100
    
    # Create environments
    env_original = ZurcherEnv(max_mileage=max_mileage, replace_cost=200, beta=0.99)
    env_jax = ZurcherEnvJAX(max_mileage=max_mileage, replace_cost=200, beta=0.99)
    
    print(f"Testing with max_mileage={max_mileage}, {n_simulations} simulations, {n_steps} steps each")
    
    # Test equilibrium computation
    print("\n1. Equilibrium Computation:")
    
    # Original environment
    start_time = time.time()
    policy_orig, vf_orig = env_original.find_equilibrium(max_iter=500)
    original_eq_time = time.time() - start_time
    
    # JAX environment
    start_time = time.time()
    vf_jax, policy_jax = env_jax.find_equilibrium_jax(max_iter=500)
    jax_eq_time = time.time() - start_time
    
    print(f"Original environment: {original_eq_time:.4f} seconds")
    print(f"JAX environment: {jax_eq_time:.4f} seconds")
    print(f"Speedup: {original_eq_time / jax_eq_time:.2f}x")
    
    # Verify results are similar
    policy_diff = np.mean(np.abs(policy_orig - np.array(policy_jax)))
    vf_diff = np.mean(np.abs(vf_orig - np.array(vf_jax)))
    print(f"Policy difference: {policy_diff:.6f}")
    print(f"Value function difference: {vf_diff:.6f}")
    
    # Test batch simulation (JAX only)
    print("\n2. Batch Simulation (JAX only):")
    start_time = time.time()
    rng_key = jax.random.PRNGKey(42)
    trajectories = env_jax.batch_simulate_jax(rng_key, n_simulations, n_steps)
    batch_sim_time = time.time() - start_time
    
    print(f"JAX batch simulation: {batch_sim_time:.4f} seconds")
    print(f"Trajectory shape: {trajectories[0].shape}")
    
    # Test moments computation
    print("\n3. Moments Computation:")
    start_time = time.time()
    moments = env_jax.compute_moments_jax(trajectories)
    moments_time = time.time() - start_time
    
    print(f"JAX moments computation: {moments_time:.4f} seconds")
    print(f"Computed moments: {moments}")
    
    return {
        'original_eq_time': original_eq_time,
        'jax_eq_time': jax_eq_time,
        'batch_sim_time': batch_sim_time,
        'moments_time': moments_time,
        'policy_diff': policy_diff,
        'vf_diff': vf_diff
    }


def demonstrate_jax_features():
    """Demonstrate advanced JAX features in the optimized environments."""
    print("\n" + "=" * 60)
    print("Advanced JAX Features Demonstration")
    print("=" * 60)
    
    # Create JAX environments
    bbl_env = BBLGameEnvJAX(n_firms=3, beta=0.95)
    zurcher_env = ZurcherEnvJAX(max_mileage=50, replace_cost=100, beta=0.99)
    
    print("\n1. JAX Random Number Generation:")
    # Show how JAX RNG works
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    random_numbers = [jax.random.normal(k) for k in keys]
    print(f"Generated random numbers: {random_numbers}")
    
    print("\n2. JIT Compilation Benefits:")
    # Show compilation time vs execution time
    @jax.jit
    def expensive_computation(x):
        return jnp.sum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2)
    
    x = jnp.arange(10000)
    
    # First call (includes compilation)
    start = time.time()
    result1 = expensive_computation(x)
    first_call_time = time.time() - start
    
    # Second call (already compiled)
    start = time.time()
    result2 = expensive_computation(x)
    second_call_time = time.time() - start
    
    print(f"First call (with compilation): {first_call_time:.6f} seconds")
    print(f"Second call (compiled): {second_call_time:.6f} seconds")
    print(f"Speedup after compilation: {first_call_time / second_call_time:.2f}x")
    
    print("\n3. Vectorized Operations:")
    # Demonstrate vectorization in Zurcher environment
    start = time.time()
    rng_key = jax.random.PRNGKey(123)
    batch_trajectories = zurcher_env.batch_simulate_jax(rng_key, 100, 50)
    vectorized_time = time.time() - start
    
    print(f"Vectorized batch simulation (100 episodes): {vectorized_time:.4f} seconds")
    print(f"Average time per episode: {vectorized_time / 100:.6f} seconds")
    
    return {
        'first_call_time': first_call_time,
        'second_call_time': second_call_time,
        'vectorized_time': vectorized_time
    }


def plot_performance_comparison(bbl_results, zurcher_results):
    """Create visualizations of the performance improvements."""
    print("\n" + "=" * 60)
    print("Creating Performance Comparison Plots")
    print("=" * 60)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: BBL Step Time Comparison
    categories = ['Original', 'JAX']
    times = [bbl_results['original_step_time'], bbl_results['jax_step_time']]
    colors = ['skyblue', 'lightcoral']
    
    ax1.bar(categories, times, color=colors)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('BBL Environment: Step Time Comparison\n(1000 steps)')
    ax1.grid(True, alpha=0.3)
    
    # Add speedup annotation
    speedup = bbl_results['original_step_time'] / bbl_results['jax_step_time']
    ax1.text(0.5, max(times) * 0.8, f'Speedup: {speedup:.2f}x', 
             ha='center', fontsize=12, weight='bold')
    
    # Plot 2: Zurcher Equilibrium Computation
    categories = ['Original', 'JAX']
    times = [zurcher_results['original_eq_time'], zurcher_results['jax_eq_time']]
    
    ax2.bar(categories, times, color=colors)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Zurcher Environment: Equilibrium Computation')
    ax2.grid(True, alpha=0.3)
    
    speedup = zurcher_results['original_eq_time'] / zurcher_results['jax_eq_time']
    ax2.text(0.5, max(times) * 0.8, f'Speedup: {speedup:.2f}x', 
             ha='center', fontsize=12, weight='bold')
    
    # Plot 3: JAX-only Features
    jax_features = ['Batch Simulation', 'Moments Computation']
    jax_times = [zurcher_results['batch_sim_time'], zurcher_results['moments_time']]
    
    ax3.bar(jax_features, jax_times, color='lightgreen')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('JAX-Only Features (Zurcher Environment)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy Comparison
    metrics = ['Policy Diff', 'Value Function Diff']
    differences = [zurcher_results['policy_diff'], zurcher_results['vf_diff']]
    
    ax4.bar(metrics, differences, color='lightyellow')
    ax4.set_ylabel('Mean Absolute Difference')
    ax4.set_title('Accuracy: Original vs JAX Implementation')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('jax_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance comparison plot saved as 'jax_performance_comparison.png'")


def main():
    """Main demonstration function."""
    print("JAX Optimization Demo for EconGym Environments")
    print("This demo shows the performance improvements from JAX optimization")
    print("\nPress Enter to continue...")
    input()
    
    # Run benchmarks
    bbl_results = benchmark_bbl_environments()
    zurcher_results = benchmark_zurcher_environments()
    jax_features = demonstrate_jax_features()
    
    # Create visualizations
    plot_performance_comparison(bbl_results, zurcher_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BBL Environment Speedup: {bbl_results['original_step_time'] / bbl_results['jax_step_time']:.2f}x")
    print(f"Zurcher Equilibrium Speedup: {zurcher_results['original_eq_time'] / zurcher_results['jax_eq_time']:.2f}x")
    print(f"JIT Compilation Speedup: {jax_features['first_call_time'] / jax_features['second_call_time']:.2f}x")
    print(f"Policy Accuracy: {zurcher_results['policy_diff']:.2e} mean absolute difference")
    print(f"Value Function Accuracy: {zurcher_results['vf_diff']:.2e} mean absolute difference")
    
    print("\nKey Benefits of JAX Optimization:")
    print("• Significant speedups through JIT compilation")
    print("• Vectorized batch operations")
    print("• Automatic differentiation for calibration")
    print("• Maintained numerical accuracy")
    print("• Seamless integration with existing code")


if __name__ == "__main__":
    main() 