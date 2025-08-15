import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import traceback
import os
import time
from collections import deque
import random

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

print("Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("Project root:", project_root)

try:
    from econgym.envs.RepeatedPrisoners_Env import RepeatedPrisonersEnv
    from econgym.envs.zurcher_env import ZurcherEnv
    from econgym.solvers.prisoners_dqn_solver import PrisonersDQNSolver, DQNNetwork
    print("Successfully imported all required modules")
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    print("Trying to import modules individually...")
    try:
        import econgym.envs.RepeatedPrisoners_Env
        print("Successfully imported RepeatedPrisoners_Env")
    except ImportError as e:
        print(f"Error importing RepeatedPrisoners_Env: {str(e)}")
    try:
        import econgym.envs.zurcher_env
        print("Successfully imported zurcher_env")
    except ImportError as e:
        print(f"Error importing zurcher_env: {str(e)}")
    try:
        import econgym.solvers.prisoners_dqn_solver
        print("Successfully imported prisoners_dqn_solver")
    except ImportError as e:
        print(f"Error importing prisoners_dqn_solver: {str(e)}")
    sys.exit(1)

def debug_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()

def safe_print(*args, **kwargs):
    """Print with error handling and forced flush"""
    try:
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"Error in safe_print: {str(e)}", flush=True)
        sys.stdout.flush()

class TransferDQNNetwork(DQNNetwork):
    """Modified DQN network for transfer learning to Zurcher model."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__(input_dim, output_dim, hidden_dim)
        # Add an adapter layer for the Zurcher model's state space
        self.adapter = torch.nn.Linear(1, input_dim)  # Zurcher state is 1D
        debug_print(f"Initialized TransferDQNNetwork with input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, x):
        try:
            # Ensure input is 2D [batch_size, 1]
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            if x.dim() == 2 and x.size(1) == 1:
                x = self.adapter(x)  # Convert Zurcher state to Prisoner's state space
            return super().forward(x)
        except Exception as e:
            debug_print(f"Error in forward pass: {str(e)}")
            debug_print(f"Input shape: {x.shape}")
            raise

def plot_value_functions(states, vf_standard, vf_transfer, title="Value Function Comparison"):
    """Plot value functions from both approaches."""
    plt.figure(figsize=(10, 6))
    plt.plot(states, vf_standard, 'b-', label='Standard Value Iteration')
    plt.plot(states, vf_transfer, 'r--', label='Transfer Learning')
    plt.xlabel('Mileage')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_learning_curves(episode_rewards_transfer, episode_rewards_no_transfer, title="Learning Curves"):
    """Plot learning curves comparing transfer vs no transfer."""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards_transfer, 'r-', label='With Transfer')
    plt.plot(episode_rewards_no_transfer, 'b--', label='No Transfer')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_dqn(env, network, config, max_episodes=1000):
    """Train DQN on environment."""
    try:
        debug_print("\nInitializing DQN training...")
        optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
        memory = deque(maxlen=config['memory_size'])
        epsilon = config['epsilon_start']
        episode_rewards = []
        
        debug_print(f"Training configuration:")
        debug_print(f"Learning rate: {config['learning_rate']}")
        debug_print(f"Gamma: {config['gamma']}")
        debug_print(f"Epsilon start: {config['epsilon_start']}")
        debug_print(f"Memory size: {config['memory_size']}")
        debug_print(f"Batch size: {config['batch_size']}")
        debug_print(f"Max episodes: {max_episodes}")
        
        max_steps_per_episode = 100  # Add maximum steps per episode
        
        for episode in range(max_episodes):
            try:
                debug_print(f"\nStarting episode {episode}")
                state, _ = env.reset()  # Unpack state and info
                state = state[0]  # Get scalar state from array
                episode_reward = 0
                done = False
                step_count = 0
                
                debug_print(f"Initial state: {state}")
                
                while not done and step_count < max_steps_per_episode:  # Add step limit
                    try:
                        step_count += 1
                        # Epsilon-greedy action selection
                        if random.random() < epsilon:
                            action = env.action_space.sample()
                            debug_print(f"Step {step_count}: Random action {action} (epsilon: {epsilon:.3f})")
                        else:
                            with torch.no_grad():
                                state_tensor = torch.FloatTensor([state]).unsqueeze(1)  # Shape: [1, 1]
                                q_values = network(state_tensor)
                                action = q_values.argmax().item()
                                debug_print(f"Step {step_count}: Q-values {q_values.numpy()}, Action {action}")
                        
                        # Take action
                        next_state, reward, done, _, _ = env.step(action)
                        next_state = next_state[0]  # Get scalar state from array
                        debug_print(f"Step {step_count}: State {state} -> {next_state}, Reward {reward}, Done {done}")
                        
                        # Store transition
                        memory.append((state, action, reward, next_state, done))
                        
                        # Train if enough samples
                        if len(memory) >= config['batch_size']:
                            try:
                                # Sample batch
                                batch = random.sample(memory, config['batch_size'])
                                states, actions, rewards, next_states, dones = zip(*batch)
                                
                                # Convert to tensors
                                states = np.array(states)
                                next_states = np.array(next_states)
                                state_batch = torch.FloatTensor(states).unsqueeze(1)  # Shape: [batch_size, 1]
                                action_batch = torch.LongTensor(actions)
                                reward_batch = torch.FloatTensor(rewards)
                                next_state_batch = torch.FloatTensor(next_states).unsqueeze(1)  # Shape: [batch_size, 1]
                                done_batch = torch.FloatTensor(dones)
                                
                                # Compute Q(s_t, a)
                                state_action_values = network(state_batch).gather(1, action_batch.unsqueeze(1))
                                
                                # Compute V(s_{t+1})
                                with torch.no_grad():
                                    next_state_values = network(next_state_batch).max(1)[0]
                                
                                # Compute expected Q values
                                expected_state_action_values = reward_batch + (1 - done_batch) * config['gamma'] * next_state_values
                                
                                # Compute loss and update
                                loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                                
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                
                                if step_count % 10 == 0:  # Print loss every 10 steps
                                    debug_print(f"Step {step_count}: Training loss: {loss.item():.4f}")
                            except Exception as e:
                                debug_print(f"Error in training step {step_count}:")
                                debug_print(traceback.format_exc())
                                continue
                        
                        state = next_state
                        episode_reward += reward
                        
                        # Add a small delay to prevent overwhelming output
                        time.sleep(0.01)
                        
                    except Exception as e:
                        debug_print(f"Error in step {step_count}:")
                        debug_print(traceback.format_exc())
                        continue
                
                # Update epsilon
                epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
                episode_rewards.append(episode_reward)
                
                debug_print(f"\nEpisode {episode} completed:")
                debug_print(f"Total steps: {step_count}")
                debug_print(f"Episode reward: {episode_reward:.2f}")
                debug_print(f"Current epsilon: {epsilon:.3f}")
                debug_print(f"Average reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
            
            except Exception as e:
                debug_print(f"Error in episode {episode}:")
                debug_print(traceback.format_exc())
                continue
        
        return episode_rewards
    
    except Exception as e:
        debug_print("Error in train_dqn:")
        debug_print(traceback.format_exc())
        return []

def main():
    try:
        # Configuration with consistent network dimensions
        config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 1000,
            'batch_size': 32,
            'hidden_dim': 128,  # Keep original size to match existing networks
            'max_episodes': 50,
            'target_update': 10
        }
        
        # 1. Train DQN on Prisoner's Dilemma
        debug_print("\n=== Starting Prisoner's Dilemma Training ===")
        prisoners_env = RepeatedPrisonersEnv(rounds=5)
        dqn_solver = PrisonersDQNSolver(prisoners_env, config)
        debug_print("Prisoner's Dilemma environment created")
        debug_print("Starting Prisoner's Dilemma training...")
        policy, value_fn = dqn_solver.solve_equilibrium(max_iter=100)
        debug_print("Prisoner's Dilemma training completed")
        
        # 2. Create Zurcher environment
        debug_print("\n=== Creating Zurcher Environment ===")
        zurcher_env = ZurcherEnv(max_mileage=50, replace_cost=500, beta=0.9)
        debug_print("Zurcher environment created with parameters:")
        debug_print(f"Max mileage: {zurcher_env.parameters['max_mileage']}")
        debug_print(f"Replace cost: {zurcher_env.parameters['replace_cost']}")
        debug_print(f"Beta: {zurcher_env.parameters['beta']}")
        
        # 3. Train with transfer learning
        debug_print("\n=== Starting Transfer Learning Training ===")
        transfer_net = TransferDQNNetwork(
            input_dim=dqn_solver.state_dim,
            output_dim=2,
            hidden_dim=config['hidden_dim']
        )
        
        # Copy weights from pre-trained network
        debug_print("\nCopying weights from pre-trained network...")
        try:
            with torch.no_grad():
                source_params = list(dqn_solver.policy_net.network.parameters())
                target_params = list(transfer_net.network.parameters())
                
                debug_print(f"Source network parameters: {len(source_params)}")
                debug_print(f"Target network parameters: {len(target_params)}")
                
                for i, (transfer_param, dqn_param) in enumerate(zip(target_params, source_params)):
                    debug_print(f"Copying layer {i}: {dqn_param.shape} -> {transfer_param.shape}")
                    transfer_param.copy_(dqn_param)
            debug_print("Weights copied successfully")
        except Exception as e:
            debug_print(f"Error copying weights: {str(e)}")
            debug_print(traceback.format_exc())
            return
        
        # Train with transfer
        debug_print("\nStarting transfer learning training...")
        episode_rewards_transfer = train_dqn(zurcher_env, transfer_net, config, max_episodes=config['max_episodes'])
        debug_print("Transfer learning training complete")
        
        if not episode_rewards_transfer:
            debug_print("Transfer learning training failed")
            return
        
        # 4. Train without transfer (from scratch)
        debug_print("\n=== Starting No-Transfer Training ===")
        no_transfer_net = TransferDQNNetwork(
            input_dim=dqn_solver.state_dim,
            output_dim=2,
            hidden_dim=config['hidden_dim']
        )
        episode_rewards_no_transfer = train_dqn(zurcher_env, no_transfer_net, config, max_episodes=config['max_episodes'])
        debug_print("No-transfer training complete")
        
        if not episode_rewards_no_transfer:
            debug_print("No-transfer training failed")
            return
        
        # 5. Compare performance
        debug_print("\n=== Performance Comparison ===")
        transfer_avg_reward = np.mean(episode_rewards_transfer)
        no_transfer_avg_reward = np.mean(episode_rewards_no_transfer)
        transfer_final_reward = np.mean(episode_rewards_transfer[-10:])
        no_transfer_final_reward = np.mean(episode_rewards_no_transfer[-10:])
        
        debug_print(f"Transfer Learning:")
        debug_print(f"  Average reward across all episodes: {transfer_avg_reward:.2f}")
        debug_print(f"  Average reward in last 10 episodes: {transfer_final_reward:.2f}")
        debug_print(f"\nNo-Transfer Learning:")
        debug_print(f"  Average reward across all episodes: {no_transfer_avg_reward:.2f}")
        debug_print(f"  Average reward in last 10 episodes: {no_transfer_final_reward:.2f}")
        
        if transfer_final_reward > no_transfer_final_reward:
            debug_print("\nTransfer learning outperformed no-transfer learning!")
            improvement = ((transfer_final_reward - no_transfer_final_reward) / abs(no_transfer_final_reward)) * 100
            debug_print(f"Improvement: {improvement:.1f}%")
        else:
            debug_print("\nNo-transfer learning performed better than transfer learning.")
            difference = ((no_transfer_final_reward - transfer_final_reward) / abs(transfer_final_reward)) * 100
            debug_print(f"Difference: {difference:.1f}%")
        
        # 6. Plot learning curves
        debug_print("\n=== Plotting Results ===")
        plot_learning_curves(episode_rewards_transfer, episode_rewards_no_transfer)
        
        # 7. Compare final policies
        debug_print("\n=== Comparing Policies ===")
        states = np.arange(zurcher_env.parameters['max_mileage'] + 1)
        policy_transfer = []
        policy_no_transfer = []
        
        debug_print("\nEvaluating policies...")
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor([state]).unsqueeze(1)
                # Get transfer policy
                q_values = transfer_net(state_tensor)
                action = q_values.argmax().item()
                policy_transfer.append(action)
                
                # Get no-transfer policy
                q_values = no_transfer_net(state_tensor)
                action = q_values.argmax().item()
                policy_no_transfer.append(action)
        
        policy_transfer = np.array(policy_transfer)
        policy_no_transfer = np.array(policy_no_transfer)
        
        debug_print("\nPolicy comparison:")
        debug_print(f"Transfer policy replacement rate: {np.mean(policy_transfer):.2%}")
        debug_print(f"No-transfer policy replacement rate: {np.mean(policy_no_transfer):.2%}")
        
    except Exception as e:
        debug_print("Error in main:")
        debug_print(traceback.format_exc())

if __name__ == "__main__":
    main() 