import numpy as np
import matplotlib.pyplot as plt
from econgym.envs.zurcher_env import ZurcherEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from econgym.envs.zurcher_analysis import compute_value_iteration_results

def train_ppo(env, total_timesteps=100000):
    """Train a PPO agent on the Zurcher environment"""
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create and train PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    return model

def get_ppo_value_function(model, states):
    """Get the value function from a trained PPO model."""
    values = []
    for state in states:
        obs = np.array([state])
        with model.policy.no_grad():
            value = model.policy.critic(obs)[0]
        values.append(value.item())
    return np.array(values)

def get_ppo_policy(model, states):
    """Get the policy (replacement probabilities) from a trained PPO model."""
    probs = []
    for state in states:
        obs = np.array([state])
        with model.policy.no_grad():
            action_probs = model.policy.get_distribution(obs).distribution.probs
        probs.append(action_probs[0, 1].item())  # Probability of replacement (action 1)
    return np.array(probs)

def normalize_value_function(values):
    """Normalize the value function to show increasing pattern."""
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

def main():
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define states (normalized to 0-100 scale)
    states = np.arange(101)  # 0 to 100 mileage
    
    # Create environment with paper's parameters
    env = ZurcherEnv(
        max_mileage=400000,  # Paper's scale
        replace_cost=11.7257,  # Paper's replacement cost
        beta=0.9999,  # Paper's discount factor
        poisson_lambda=1.0  # Paper's Poisson parameter
    )
    
    # Get value iteration results
    print("Computing value iteration results...")
    states, values_vi, probs_vi = compute_value_iteration_results(env)
    
    # Train PPO with more timesteps for better convergence
    print("\nTraining PPO agent...")
    model = train_ppo(env, total_timesteps=500000)  # Increased timesteps
    
    # Get PPO results
    print("\nComputing PPO results...")
    values_ppo = get_ppo_value_function(model, states)
    probs_ppo = get_ppo_policy(model, states)
    
    # Normalize value functions
    values_vi_norm = normalize_value_function(values_vi)
    values_ppo_norm = normalize_value_function(values_ppo)
    
    # Create figure with two subplots
    plt.style.use('seaborn')  # Use seaborn style for better aesthetics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot normalized value functions
    ax1.plot(states, values_vi_norm, 'b-', linewidth=2, label='Value Iteration')
    ax1.plot(states, values_ppo_norm, 'r--', linewidth=2, label='PPO')
    ax1.set_xlabel('Mileage (×1000)', fontsize=12)
    ax1.set_ylabel('Normalized Value', fontsize=12)
    ax1.set_title('Value Functions Comparison', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot replacement probabilities
    ax2.plot(states, probs_vi, 'b-', linewidth=2, label='Value Iteration')
    ax2.plot(states, probs_ppo, 'r--', linewidth=2, label='PPO')
    ax2.set_xlabel('Mileage (×1000)', fontsize=12)
    ax2.set_ylabel('Replacement Probability', fontsize=12)
    ax2.set_title('Replacement Probabilities', fontsize=14, pad=15)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Keep', 'Replace'], fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot with full path
    plot_path = os.path.join(plots_dir, 'zurcher_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved as: {plot_path}")
    
    # Print statistics
    print("\nValue Function Statistics:")
    print("\nValue Iteration:")
    print(f"Mean value: {np.mean(values_vi):.2f}")
    print(f"Min value: {np.min(values_vi):.2f}")
    print(f"Max value: {np.max(values_vi):.2f}")
    
    print("\nPPO:")
    print(f"Mean value: {np.mean(values_ppo):.2f}")
    print(f"Min value: {np.min(values_ppo):.2f}")
    print(f"Max value: {np.max(values_ppo):.2f}")
    
    # Print replacement statistics
    print("\nReplacement Policy Statistics:")
    print("\nValue Iteration:")
    print(f"Mean replacement probability: {np.mean(probs_vi):.2f}")
    print(f"Max replacement probability: {np.max(probs_vi):.2f}")
    
    print("\nPPO:")
    print(f"Mean replacement probability: {np.mean(probs_ppo):.2f}")
    print(f"Max replacement probability: {np.max(probs_ppo):.2f}")

if __name__ == "__main__":
    main() 