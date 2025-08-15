import numpy as np
from econgym.envs.RepeatedPrisoners_Env import RepeatedPrisonersEnv
from econgym.solvers.prisoners_dqn_solver import PrisonersDQNSolver
import matplotlib.pyplot as plt

def plot_training_progress(rewards, title="Training Progress"):
    """Plot the training rewards and metrics over time."""
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Average Reward', color='blue')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    
    # Add moving average
    window_size = 50
    if len(rewards) > window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                label=f'{window_size}-Episode Moving Average', 
                color='red', linestyle='--')
        plt.legend()
    
    # Add second subplot for reward distribution
    plt.subplot(2, 1, 2)
    plt.hist(rewards, bins=50, alpha=0.75, color='green')
    plt.title("Reward Distribution")
    plt.xlabel("Reward Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Create environment
    env = RepeatedPrisonersEnv(rounds=1000)
    
    # Create DQN solver with custom configuration
    config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_dim': 128
    }
    solver = PrisonersDQNSolver(env, config)
    
    # Train the agent
    print("Training DQN agent...")
    policy, value_fn = solver.solve_equilibrium(max_iter=1000)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    state, _ = env.reset()
    done = False
    total_reward = 0
    actions_history = []
    
    while not done:
        # Get actions from both players using the trained policy
        action1 = solver.select_action(state, training=False)
        action2 = solver.select_action(state, training=False)
        actions = (action1, action2)
        
        # Take step in environment
        state, reward, done, _, info = env.step(actions)
        total_reward += np.mean(reward)
        actions_history.append(actions)
    
    # Print results
    print(f"\nTotal reward: {total_reward:.2f}")
    print("\nAction history:")
    for i, (a1, a2) in enumerate(actions_history):
        print(f"Round {i+1}: Player 1: {'Cooperate' if a1 == 0 else 'Defect'}, "
              f"Player 2: {'Cooperate' if a2 == 0 else 'Defect'}")
    
    # Plot training progress
    plot_training_progress(solver.episode_rewards, "DQN Training Progress")

if __name__ == "__main__":
    main() 