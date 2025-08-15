import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from collections import deque, defaultdict
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from econgym.envs.zurcher_env import ZurcherEnv
from econgym.envs.RepeatedPrisoners_Env import RepeatedPrisonersEnv
from econgym.solvers.prisoners_dqn_solver import PrisonersDQNSolver, DQNNetwork
from econgym.core.utils import seed_everything

class TransferDQNNetwork(DQNNetwork):
    """Modified DQN network for transfer learning to Zurcher model."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__(input_dim, output_dim, hidden_dim)
        # Add an adapter layer for the Zurcher model's state space
        self.adapter = torch.nn.Linear(1, input_dim)  # Zurcher state is 1D
    
    def forward(self, x):
        # Ensure input is 2D [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.dim() == 2 and x.size(1) == 1:
            x = self.adapter(x)  # Convert Zurcher state to Prisoner's state space
        return super().forward(x)

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

def train_dqn(env, network, config, max_episodes=1000):
    """Train DQN on environment."""
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    memory = deque(maxlen=config['memory_size'])
    epsilon = config['epsilon_start']
    episode_rewards = []
    
    max_steps_per_episode = 100
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state[0]  # Get scalar state from array
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            step_count += 1
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor([state]).unsqueeze(1)
                    q_values = network(state_tensor)
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state[0]
            
            # Store transition
            memory.append((state, action, reward, next_state, done))
            
            # Train if enough samples
            if len(memory) >= config['batch_size']:
                # Sample batch
                batch = random.sample(memory, config['batch_size'])
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states = np.array(states)
                next_states = np.array(next_states)
                state_batch = torch.FloatTensor(states).unsqueeze(1)
                action_batch = torch.LongTensor(actions)
                reward_batch = torch.FloatTensor(rewards)
                next_state_batch = torch.FloatTensor(next_states).unsqueeze(1)
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
            
            state = next_state
            episode_reward += reward
        
        # Update epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    return episode_rewards

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

def run_zurcher_demo():
    print("\n=== Running Zurcher Bus Replacement Model Demo ===")
    
    # Create environment with more realistic parameters
    env = ZurcherEnv(
        max_mileage=100,
        replace_cost=200,  # Moderate replacement cost
        beta=0.99  # High discount factor
    )
    
    # Set maintenance cost parameters
    env.parameters['maintenance_cost_base'] = 5.0  # Lower base maintenance cost
    env.parameters['maintenance_cost_slope'] = 2.0  # Steeper increase with mileage
    
    # Generate states for analysis
    states = np.linspace(0, env.parameters['max_mileage'], 101)
    
    # Plot maintenance costs vs replacement costs
    mileages = np.linspace(0, env.parameters['max_mileage'], 101)
    maintenance_costs = [env.maintenance_cost(m) for m in mileages]
    replacement_costs = [env.parameters['replace_cost']] * len(mileages)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mileages, maintenance_costs, 'b-', label='Maintenance Cost')
    plt.plot(mileages, replacement_costs, 'r--', label='Replacement Cost')
    plt.xlabel('Mileage')
    plt.ylabel('Cost')
    plt.title('Maintenance vs Replacement Costs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print maintenance costs at different mileages
    print("\nMaintenance costs at different mileages:")
    for m in [0, 25, 50, 75, 100]:
        print(f"Mileage {m}: {env.maintenance_cost(m):.2f}")
    
    # Configuration for Dyna-Q
    config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 1000,
        'batch_size': 32,
        'n_planning_steps': 5
    }
    
    # Train Dyna-Q agent
    print("\nTraining Dyna-Q agent...")
    network, episode_rewards = train_dyna_q(env, config, max_episodes=200)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Dyna-Q Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()
    
    # Get value functions and policies
    print("\nComputing value functions and policies...")
    dyna_q_values = get_dyna_q_value_function(network, states)
    dyna_q_policy = get_dyna_q_policy(network, states)
    
    # Plot value functions
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(states, dyna_q_values, 'b-', label='Dyna-Q Value Function')
    plt.xlabel('Mileage')
    plt.ylabel('Value')
    plt.title('Value Functions')
    plt.legend()
    plt.grid(True)
    
    # Plot policies
    plt.subplot(2, 1, 2)
    plt.plot(states, dyna_q_policy, 'r-', label='Dyna-Q Policy')
    plt.xlabel('Mileage')
    plt.ylabel('Replacement Probability')
    plt.title('Replacement Policies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Simulate with trained policy
    print("\nSimulating with trained policy...")
    state, _ = env.reset()
    state = state[0]
    done = False
    total_reward = 0
    mileages = []
    replacements = 0
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor([state]).unsqueeze(1)
            q_values = network(state_tensor)
            action = q_values.argmax().item()
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state[0]
        
        mileages.append(state)
        if action == 1:
            replacements += 1
        
        state = next_state
        total_reward += reward
    
    # Print simulation results
    print(f"\nSimulation Results:")
    print(f"Average mileage: {np.mean(mileages):.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Replacement frequency: {replacements/len(mileages):.3f}")
    
    # Plot mileage trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(mileages)
    plt.title('Mileage Trajectory')
    plt.xlabel('Time')
    plt.ylabel('Mileage')
    plt.grid(True)
    plt.show()

def run_prisoners_demo():
    print("\n=== Running Repeated Prisoner's Dilemma Demo ===")
    
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
    print("\nTraining DQN agent...")
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
        
        # Take step in environment (Gymnasium 5-tuple)
        state, reward, done, _, info = env.step(actions)
        total_reward += np.mean(reward)
        actions_history.append(actions)
    
    # Print results
    print(f"\nTotal reward: {total_reward:.2f}")
    print("\nAction history (last 10 rounds):")
    for i, (a1, a2) in enumerate(actions_history[-10:]):
        print(f"Round {i+1}: Player 1: {'Cooperate' if a1 == 0 else 'Defect'}, "
              f"Player 2: {'Cooperate' if a2 == 0 else 'Defect'}")
    
    # Plot training progress
    plot_training_progress(solver.episode_rewards, "Prisoner's Dilemma DQN Training Progress")

def run_transfer_demo():
    print("\n=== Running Transfer Learning Demo ===")
    
    # Configuration
    config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 1000,
        'batch_size': 32,
        'hidden_dim': 128,
        'max_episodes': 50,
        'target_update': 10
    }
    
    # 1. Train DQN on Prisoner's Dilemma
    print("\nTraining on Prisoner's Dilemma...")
    prisoners_env = RepeatedPrisonersEnv(rounds=5)
    dqn_solver = PrisonersDQNSolver(prisoners_env, config)
    policy, value_fn = dqn_solver.solve_equilibrium(max_iter=100)
    
    # 2. Create Zurcher environment
    print("\nCreating Zurcher environment...")
    zurcher_env = ZurcherEnv(max_mileage=50, replace_cost=500, beta=0.9)
    
    # 3. Train with transfer learning
    print("\nTraining with transfer learning...")
    transfer_net = TransferDQNNetwork(
        input_dim=dqn_solver.state_dim,
        output_dim=2,
        hidden_dim=config['hidden_dim']
    )
    
    # Copy weights from pre-trained network except adapter layer
    print("\nCopying weights from pre-trained network...")
    pretrained_dict = dqn_solver.policy_net.state_dict()
    transfer_dict = transfer_net.state_dict()
    
    # Filter out adapter layer parameters
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in transfer_dict and 'adapter' not in k}
    transfer_dict.update(pretrained_dict)
    transfer_net.load_state_dict(transfer_dict)
    
    # Train on Zurcher environment with transfer
    print("\nFine-tuning on Zurcher environment...")
    rewards_transfer = train_dqn(zurcher_env, transfer_net, config, max_episodes=200)
    
    # Train from scratch for comparison
    print("\nTraining from scratch for comparison...")
    scratch_net = DQNNetwork(1, 2, config['hidden_dim'])
    rewards_scratch = train_dqn(zurcher_env, scratch_net, config, max_episodes=200)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_transfer, 'r-', label='With Transfer')
    plt.plot(rewards_scratch, 'b--', label='From Scratch')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Transfer Learning vs Training from Scratch')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("="*50)
    print("Running Economic Models Demo")
    print("="*50)
    seed_everything(42)
    
    # Run Prisoner's Dilemma demo
    run_prisoners_demo()
    
    # Run Transfer Learning demo
    run_transfer_demo()
    
    # Run Zurcher demo
    run_zurcher_demo()

if __name__ == "__main__":
    main() 