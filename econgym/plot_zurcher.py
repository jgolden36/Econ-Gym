import numpy as np
import matplotlib.pyplot as plt
from econgym.envs.zurcher_env import ZurcherEnv

def plot_zurcher_analysis():
    # Create environments with different beta values
    env_beta_high = ZurcherEnv(beta=0.9999)
    env_beta_zero = ZurcherEnv(beta=0.0)
    
    # Find equilibrium for both environments
    policy_high, value_net_high = env_beta_high.find_equilibrium()
    policy_zero, value_net_zero = env_beta_zero.find_equilibrium()
    
    # Generate states for plotting (using actual mileage values)
    states = np.arange(0, env_beta_high.parameters['max_mileage'] + 1, 1000)  # Plot every 1000 miles
    
    # Get value functions
    values_high = [env_beta_high.get_value_function(s) for s in states]
    values_zero = [env_beta_zero.get_value_function(s) for s in states]
    
    # Get replacement probabilities
    obs = np.array([[s] for s in states], dtype=np.float32)
    probs_high = policy_high[states]
    probs_zero = policy_zero[states]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot value functions
    ax1.plot(states/1000, values_high, 'b-', label='β = 0.9999', linewidth=2)
    ax1.plot(states/1000, values_zero, 'r--', label='β = 0.0', linewidth=2)
    ax1.set_xlabel('Mileage (thousands)', fontsize=12)
    ax1.set_ylabel('Value Function', fontsize=12)
    ax1.set_title('Value Function Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot replacement probabilities
    ax2.plot(states/1000, probs_high, 'b-', label='β = 0.9999', linewidth=2)
    ax2.plot(states/1000, probs_zero, 'r--', label='β = 0.0', linewidth=2)
    ax2.set_xlabel('Mileage (thousands)', fontsize=12)
    ax2.set_ylabel('Replacement Probability', fontsize=12)
    ax2.set_title('Replacement Probability Comparison', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add maintenance cost function for reference
    maintenance_cost = env_beta_high.parameters['maintenance_cost_slope'] * states
    ax1.plot(states/1000, -maintenance_cost, 'g:', label='Maintenance Cost', linewidth=1.5, alpha=0.7)
    ax1.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('zurcher_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_zurcher_analysis() 