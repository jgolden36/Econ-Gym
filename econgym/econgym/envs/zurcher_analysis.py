import numpy as np
import matplotlib.pyplot as plt
from econgym.envs.zurcher_env import ZurcherEnv
from econgym.envs.zurcher_fpml import ZurcherFPML
import os
import torch
from collections import deque, defaultdict
import random
import time
import pandas as pd

class DynaQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class EnvironmentModel:
    """Learned model of the environment for Dyna-Q."""
    def __init__(self, state_dim, action_dim):
        self.model = defaultdict(lambda: defaultdict(lambda: (0, 0)))  # state -> action -> (next_state, reward)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def update(self, state, action, next_state, reward):
        """Update the model with a new experience."""
        self.model[state][action] = (next_state, reward)
    
    def sample(self):
        """Sample a random state-action pair from the model."""
        if not self.model:
            return None, None, None, None
        
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        next_state, reward = self.model[state][action]
        return state, action, next_state, reward

def train_dyna_q(env, config, max_episodes=1000):
    """Train Dyna-Q on environment."""
    print("Initializing Dyna-Q training...")
    network = DynaQNetwork(1, 2)  # 1 input (mileage), 2 outputs (keep/replace)
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    memory = deque(maxlen=config['memory_size'])
    model = EnvironmentModel(1, 2)  # 1D state space, 2 actions
    epsilon = config['epsilon_start']
    episode_rewards = []
    
    print(f"Starting training for {max_episodes} episodes...")
    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state[0]  # Get scalar state from array
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            if steps % 100 == 0:
                print(f"Episode {episode}, Step {steps}")
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor([state]).unsqueeze(1)
                    q_values = network(state_tensor)
                    action = q_values.argmax().item()
            
            # Take action in real environment
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state[0]
            
            # Store transition
            memory.append((state, action, reward, next_state, done))
            
            # Update model
            model.update(state, action, next_state, reward)
            
            # Train on real experience
            if len(memory) >= config['batch_size']:
                batch = random.sample(memory, config['batch_size'])
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states = torch.FloatTensor(states).unsqueeze(1)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states).unsqueeze(1)
                dones = torch.FloatTensor(dones)
                
                # Compute Q(s_t, a)
                state_action_values = network(states).gather(1, actions.unsqueeze(1))
                
                # Compute V(s_{t+1})
                with torch.no_grad():
                    next_state_values = network(next_states).max(1)[0]
                
                # Compute expected Q values
                expected_state_action_values = rewards + (1 - dones) * config['gamma'] * next_state_values
                
                # Compute loss and update
                loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Dyna-Q: Additional training on simulated experiences
            if len(model.model) > 0:  # Only do planning if we have some model data
                for _ in range(config['n_planning_steps']):
                    # Sample from model
                    sim_state, sim_action, sim_next_state, sim_reward = model.sample()
                    if sim_state is None:
                        continue
                    
                    # Convert to tensors
                    sim_state_tensor = torch.FloatTensor([sim_state]).unsqueeze(1)
                    sim_next_state_tensor = torch.FloatTensor([sim_next_state]).unsqueeze(1)
                    
                    # Compute Q(s_t, a)
                    sim_state_action_value = network(sim_state_tensor).gather(1, torch.LongTensor([[sim_action]]))
                    
                    # Compute V(s_{t+1})
                    with torch.no_grad():
                        sim_next_state_value = network(sim_next_state_tensor).max(1)[0]
                    
                    # Compute expected Q value
                    sim_expected_value = sim_reward + config['gamma'] * sim_next_state_value
                    
                    # Compute loss and update
                    sim_loss = torch.nn.MSELoss()(sim_state_action_value, sim_expected_value.unsqueeze(1))
                    
                    optimizer.zero_grad()
                    sim_loss.backward()
                    optimizer.step()
            
            state = next_state
            episode_reward += reward
        
        # Update epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:  # Print more frequently
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {steps}")
            print(f"Model size: {len(model.model)} states")
    
    print("Training completed!")
    return network, episode_rewards

def get_dyna_q_value_function(network, states):
    """Get value function from trained Dyna-Q network."""
    values = []
    with torch.no_grad():
        for state in states:
            state_tensor = torch.FloatTensor([state]).unsqueeze(1)
            q_values = network(state_tensor)
            values.append(float(q_values.max()))
    return values

def get_dyna_q_policy(network, states):
    """Get policy from trained Dyna-Q network."""
    probs = []
    with torch.no_grad():
        for state in states:
            state_tensor = torch.FloatTensor([state]).unsqueeze(1)
            q_values = network(state_tensor)
            action = q_values.argmax().item()
            probs.append(1.0 if action == 1 else 0.0)
    return probs

def get_value_function(env, states):
    """Helper function to get value function for all states"""
    values = []
    for state in states:
        value = env.get_value_function(state)
        values.append(float(value))
    return values

def get_replacement_probabilities(env, states):
    """Helper function to get replacement probabilities directly from policy"""
    probs = []
    for state in states:
        # Policy is deterministic, so probability is 1 if policy says replace
        prob = 1.0 if env.get_policy(state) == 1 else 0.0
        probs.append(prob)
    return probs

def normalize_value_function(values):
    """Normalize value function to [0,1] range and invert to show increasing pattern"""
    min_val = np.min(values)
    max_val = np.max(values)
    normalized = (values - min_val) / (max_val - min_val)
    return 1 - normalized  # Invert to show increasing pattern

def generate_synthetic_data(env, n_samples=1000):
    """Generate synthetic data for estimation."""
    data = []
    for _ in range(n_samples):
        state, _ = env.reset()
        state = state[0]  # Get scalar state from array
        action = env.get_policy(state)
        data.append((state, action))
    return data

def load_bus_data(file_path):
    """Load and preprocess bus1234 data."""
    # Read the data
    df = pd.read_csv(file_path)
    
    # Sort by bus ID and time to ensure proper ordering
    df = df.sort_values(['id', 'year', 'month'])
    
    # Create state-action pairs using actual mileage values
    data = list(zip(df['miles'].astype(int), df['replace'].astype(int)))
    
    # Get unique bus IDs for potential future use
    bus_ids = df['id'].unique()
    
    print(f"Data summary:")
    print(f"Number of buses: {len(bus_ids)}")
    print(f"Time period: {df['year'].min()}-{df['year'].max()}")
    print(f"Mileage range: {df['miles'].min()} to {df['miles'].max()}")
    print(f"Number of replacements: {df['replace'].sum()}")
    
    return data, df['miles'].max()

def compute_value_iteration_results(env):
    """Compute value iteration results for the Zurcher environment."""
    # Define states (normalized to 0-100 scale)
    states = np.arange(101)  # 0 to 100 mileage
    
    # Find equilibrium using value function iteration
    value_function, policy = env.find_equilibrium(tol=1e-6, max_iter=1000)
    
    # Get values and probabilities for the normalized states
    values = []
    probs = []
    
    for state in states:
        # Scale state to match environment's scale
        scaled_state = int(state * env.parameters['max_mileage'] / 100)
        values.append(value_function[scaled_state])
        probs.append(policy[scaled_state])
    
    return np.array(states), np.array(values), np.array(probs)

def main():
    try:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load bus data
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'bus1234.csv')
        print("Loading bus data...")
        bus_data, max_miles = load_bus_data(data_path)
        print(f"Loaded {len(bus_data)} observations with maximum mileage of {max_miles}")
        
        # Define states
        states = np.arange(101)  # 0 to 100 normalized mileage
        
        # Create environments with different beta values
        print("\nInitializing environments...")
        env_high_beta = ZurcherEnv(beta=0.9999)
        env_zero_beta = ZurcherEnv(beta=0.0)
        
        # Create FPML estimators
        print("Creating FPML estimators...")
        fpml_high = ZurcherFPML(beta=0.9999, max_mileage=int(max_miles))
        fpml_zero = ZurcherFPML(beta=0.0, max_mileage=int(max_miles))
        
        # Estimate using FPML
        print("\nEstimating with FPML for β=0.9999...")
        try:
            fpml_results_high = fpml_high.estimate(bus_data)
            print("FPML estimation results for β=0.9999:")
            print(f"Estimated parameters: {fpml_results_high['estimated_parameters']}")
            print(f"Log-likelihood: {fpml_results_high['log_likelihood']:.2f}")
        except Exception as e:
            print(f"Error in FPML estimation for β=0.9999: {str(e)}")
            return
        
        print("\nEstimating with FPML for β=0.0...")
        try:
            fpml_results_zero = fpml_zero.estimate(bus_data)
            print("FPML estimation results for β=0.0:")
            print(f"Estimated parameters: {fpml_results_zero['estimated_parameters']}")
            print(f"Log-likelihood: {fpml_results_zero['log_likelihood']:.2f}")
        except Exception as e:
            print(f"Error in FPML estimation for β=0.0: {str(e)}")
            return
        
        # Get FPML value functions and policies
        print("\nComputing value functions and policies...")
        values_high = fpml_results_high['value_function']
        values_zero = fpml_results_zero['value_function']
        probs_high = fpml_results_high['policy']
        probs_zero = fpml_results_zero['policy']
        
        # Train Dyna-Q for both environments
        dyna_q_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 64,
            'n_planning_steps': 50  # Number of planning steps for Dyna-Q
        }
        
        print("\nTraining Dyna-Q for β=0.9999...")
        try:
            dyna_q_high, rewards_high = train_dyna_q(env_high_beta, dyna_q_config, max_episodes=1000)
            print("Dyna-Q training completed for β=0.9999")
        except Exception as e:
            print(f"Error in Dyna-Q training for β=0.9999: {str(e)}")
            return
        
        print("\nTraining Dyna-Q for β=0.0...")
        try:
            dyna_q_zero, rewards_zero = train_dyna_q(env_zero_beta, dyna_q_config, max_episodes=1000)
            print("Dyna-Q training completed for β=0.0")
        except Exception as e:
            print(f"Error in Dyna-Q training for β=0.0: {str(e)}")
            return
        
        # Get Dyna-Q value functions and policies
        print("\nComputing Dyna-Q value functions and policies...")
        dyna_q_values_high = get_dyna_q_value_function(dyna_q_high, states)
        dyna_q_values_zero = get_dyna_q_value_function(dyna_q_zero, states)
        dyna_q_probs_high = get_dyna_q_policy(dyna_q_high, states)
        dyna_q_probs_zero = get_dyna_q_policy(dyna_q_zero, states)
        
        # Normalize value functions
        print("\nNormalizing value functions...")
        values_high_norm = normalize_value_function(values_high)
        values_zero_norm = normalize_value_function(values_zero)
        dyna_q_values_high_norm = normalize_value_function(dyna_q_values_high)
        dyna_q_values_zero_norm = normalize_value_function(dyna_q_values_zero)
        
        # Create figure with four subplots
        print("\nCreating plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot normalized value functions comparison
        ax1.plot(states, values_high_norm, 'b-', label='FPML β=0.9999')
        ax1.plot(states, dyna_q_values_high_norm, 'b--', label='Dyna-Q β=0.9999')
        ax1.set_xlabel('Normalized Mileage')
        ax1.set_ylabel('Normalized Value')
        ax1.set_title('Value Functions (β=0.9999)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot replacement probabilities comparison
        ax2.step(states, probs_high, 'b-', label='FPML β=0.9999', where='post')
        ax2.step(states, dyna_q_probs_high, 'b--', label='Dyna-Q β=0.9999', where='post')
        ax2.set_xlabel('Normalized Mileage')
        ax2.set_ylabel('Replacement Decision')
        ax2.set_title('Replacement Policy (β=0.9999)')
        ax2.set_ylim([-0.1, 1.1])
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Keep', 'Replace'])
        ax2.grid(True)
        ax2.legend()
        
        # Plot normalized value functions for β=0.0
        ax3.plot(states, values_zero_norm, 'r-', label='FPML β=0.0')
        ax3.plot(states, dyna_q_values_zero_norm, 'r--', label='Dyna-Q β=0.0')
        ax3.set_xlabel('Normalized Mileage')
        ax3.set_ylabel('Normalized Value')
        ax3.set_title('Value Functions (β=0.0)')
        ax3.grid(True)
        ax3.legend()
        
        # Plot replacement probabilities for β=0.0
        ax4.step(states, probs_zero, 'r-', label='FPML β=0.0', where='post')
        ax4.step(states, dyna_q_probs_zero, 'r--', label='Dyna-Q β=0.0', where='post')
        ax4.set_xlabel('Normalized Mileage')
        ax4.set_ylabel('Replacement Decision')
        ax4.set_title('Replacement Policy (β=0.0)')
        ax4.set_ylim([-0.1, 1.1])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Keep', 'Replace'])
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot with full path
        plot_path = os.path.join(plots_dir, 'zurcher_comparison_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlot saved as: {plot_path}")
        
        # Print statistics
        print("\nValue Function Statistics:")
        print("\nFPML β=0.9999:")
        print(f"Mean value: {np.mean(values_high):.2f}")
        print(f"Min value: {np.min(values_high):.2f}")
        print(f"Max value: {np.max(values_high):.2f}")
        
        print("\nDyna-Q β=0.9999:")
        print(f"Mean value: {np.mean(dyna_q_values_high):.2f}")
        print(f"Min value: {np.min(dyna_q_values_high):.2f}")
        print(f"Max value: {np.max(dyna_q_values_high):.2f}")
        
        print("\nFPML β=0.0:")
        print(f"Mean value: {np.mean(values_zero):.2f}")
        print(f"Min value: {np.min(values_zero):.2f}")
        print(f"Max value: {np.max(values_zero):.2f}")
        
        print("\nDyna-Q β=0.0:")
        print(f"Mean value: {np.mean(dyna_q_values_zero):.2f}")
        print(f"Min value: {np.min(dyna_q_values_zero):.2f}")
        print(f"Max value: {np.max(dyna_q_values_zero):.2f}")
        
        # Print replacement statistics
        print("\nReplacement Policy Statistics:")
        print("\nFPML β=0.9999:")
        print(f"Replacement threshold: {np.where(np.array(probs_high) == 1)[0][0] if any(probs_high) else 'Never'}")
        print(f"Number of replacement states: {sum(probs_high)}")
        
        print("\nDyna-Q β=0.9999:")
        print(f"Replacement threshold: {np.where(np.array(dyna_q_probs_high) == 1)[0][0] if any(dyna_q_probs_high) else 'Never'}")
        print(f"Number of replacement states: {sum(dyna_q_probs_high)}")
        
        print("\nFPML β=0.0:")
        print(f"Replacement threshold: {np.where(np.array(probs_zero) == 1)[0][0] if any(probs_zero) else 'Never'}")
        print(f"Number of replacement states: {sum(probs_zero)}")
        
        print("\nDyna-Q β=0.0:")
        print(f"Replacement threshold: {np.where(np.array(dyna_q_probs_zero) == 1)[0][0] if any(dyna_q_probs_zero) else 'Never'}")
        print(f"Number of replacement states: {sum(dyna_q_probs_zero)}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 