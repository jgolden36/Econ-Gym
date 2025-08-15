import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from econgym.envs.TANK_env import TANKEnv
import os

def main():
    try:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create TANK environment
        env = TANKEnv(
            beta=0.99,      # Discount factor
            alpha=0.33,     # Capital share
            delta=0.025,    # Depreciation rate
            phi_pi=1.5,     # Taylor rule coefficient on inflation
            phi_y=0.5,      # Taylor rule coefficient on output gap
            rho_a=0.9,      # Persistence of productivity shock
            rho_r=0.8       # Persistence of monetary policy shock
        )

        # Reset environment
        state, _ = env.reset()

        # Run simulation for 100 periods
        n_periods = 100
        states = np.zeros((n_periods, env.observation_space.shape[0], 3))
        rewards = np.zeros((n_periods, env.observation_space.shape[0]))
        actions = np.zeros((n_periods, env.observation_space.shape[0]))

        # Store initial state
        states[0] = state

        # Run simulation
        for t in range(1, n_periods):
            # Get optimal actions from equilibrium policy
            actions[t-1] = env.get_policy(states[t-1])
            
            # Step environment
            states[t], rewards[t-1], done, truncated, info = env.step(actions[t-1])
            
            if done or truncated:
                break

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot capital for each agent type
        for agent_type in range(env.observation_space.shape[0]):
            axes[0, 0].plot(states[:t, agent_type, 0], label=f'Agent {agent_type}')
        axes[0, 0].set_title('Capital Stock')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Capital')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot productivity for each agent type
        for agent_type in range(env.observation_space.shape[0]):
            axes[0, 1].plot(states[:t, agent_type, 1], label=f'Agent {agent_type}')
        axes[0, 1].set_title('Productivity')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Log Productivity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot interest rate for each agent type
        for agent_type in range(env.observation_space.shape[0]):
            axes[1, 0].plot(states[:t, agent_type, 2], label=f'Agent {agent_type}')
        axes[1, 0].set_title('Interest Rate')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Interest Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot consumption for each agent type
        for agent_type in range(env.observation_space.shape[0]):
            axes[1, 1].plot(actions[:t, agent_type], label=f'Agent {agent_type}')
        axes[1, 1].set_title('Consumption')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Consumption')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        
        # Save plot in the plots directory
        plot_path = os.path.join(plots_dir, 'tank_simulation.png')
        plt.savefig(plot_path)
        print(f"\nPlot saved to: {plot_path}")
        
        # Print some statistics
        print("\nSimulation Statistics:")
        print(f"Number of periods simulated: {t}")
        print("\nFinal State:")
        env.render()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 