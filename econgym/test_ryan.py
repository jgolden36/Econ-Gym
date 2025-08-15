import numpy as np
from econgym.envs.Ryan_env import RyanEnv
from econgym.wrappers.value_function import ValueFunctionWrapper
from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

def main():
    # Create the base environment
    env = RyanEnv(n_firms=3, A=10.0, B=0.5, tau=1.0, beta=0.95)
    
    # Wrap with value function wrapper
    vf_env = ValueFunctionWrapper(env)
    
    # Configure MBRL
    mbrl_config = MBRLConfig(
        hidden_dim=128,
        num_layers=2,
        horizon=3,
        num_particles=20
    )
    
    # Wrap with MBRL
    mbrl_env = MBRLWrapper(vf_env, mbrl_config)
    
    # Find equilibrium
    print("Computing equilibrium...")
    policy, V = env.find_equilibrium(plot=False)  # Don't plot yet
    print("Equilibrium policy shape:", policy.shape)
    print("Equilibrium value function shape:", V.shape)
    
    # Simulate using equilibrium policy
    print("\nSimulating with equilibrium policy...")
    states, rewards = env.simulate(n_periods=1000, plot=False)  # Don't plot yet
    print("Average cost shocks:", np.mean(states, axis=0))
    print("Average firm profits:", np.mean(rewards, axis=0))
    
    # Train MBRL
    print("\nTraining MBRL...")
    for episode in range(100):
        state, _ = mbrl_env.reset()
        episode_reward = 0
        
        for step in range(50):
            # Plan action using learned model
            action = mbrl_env.plan_action(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = mbrl_env.step(action)
            episode_reward += np.mean(reward)  # Average reward across firms
            
            # Train dynamics model
            mbrl_env.train_dynamics_model()
            
            if done or truncated:
                break
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward}")
    
    # Generate all plots after simulation is complete
    print("\nGenerating plots...")
    
    # Plot equilibrium results
    env.plot_equilibrium(policy, V)
    env.plot_transition_probabilities()
    
    # Plot simulation results
    env.plot_simulation_results(states, rewards)
    
    print("\nSimulation complete! Check the generated plots.")

if __name__ == "__main__":
    main() 