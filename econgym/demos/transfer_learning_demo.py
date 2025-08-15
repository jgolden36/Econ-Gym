import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from econgym.envs.zurcher_env import ZurcherEnv
from econgym.envs.aiyagari_env import AiyagariEnv
from econgym.wrappers.value_function import ValueFunctionWrapper
from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

class ValueFunctionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_value_function(env, network, n_episodes=1000, lr=0.001):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_loss = 0
        
        while not done:
            # Get target value from environment
            target_value = env.get_value_function(state)
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Forward pass
            predicted_value = network(state_tensor)
            
            # Compute loss
            loss = criterion(predicted_value, torch.FloatTensor([target_value]))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Take step in environment
            action = env.get_policy(state)
            state, _, done, _ = env.step(action)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Loss: {total_loss/100:.4f}")

def calibrate_with_network(env, network, targets, n_simulations=1000):
    """
    Calibrate model parameters using the trained value function network.
    """
    def objective(params):
        # Update environment parameters
        env.parameters['alpha'] = params[0]
        env.parameters['delta'] = params[1]
        env.parameters['phi_pi'] = params[2]
        
        # Simulate using the network for value function approximation
        states = []
        rewards = []
        state = env.reset()
        
        for _ in range(n_simulations):
            # Use network to predict value function
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            predicted_value = network(state_tensor).item()
            
            # Use predicted value to guide action selection
            action = env.get_policy(state)
            state, reward, done, _ = env.step(action)
            
            states.append(state)
            rewards.append(reward)
            
            if done:
                break
        
        # Compute moments
        mean_output = np.mean([env.parameters['A'] * (K ** env.parameters['alpha']) 
                             for K in np.array(states)[:, 0]])
        mean_consumption = np.mean(rewards)
        mean_capital = np.mean(np.array(states)[:, 0])
        
        # Compute squared differences from targets
        error = 0
        if 'mean_output' in targets:
            error += (mean_output - targets['mean_output'])**2
        if 'mean_consumption' in targets:
            error += (mean_consumption - targets['mean_consumption'])**2
        if 'mean_capital' in targets:
            error += (mean_capital - targets['mean_capital'])**2
            
        return error
    
    from scipy.optimize import minimize
    
    # Initial guess and bounds
    x0 = [
        env.parameters['alpha'],
        env.parameters['delta'],
        env.parameters['phi_pi']
    ]
    bounds = [
        (0.2, 0.4),    # alpha
        (0.01, 0.05),  # delta
        (1.0, 2.0)     # phi_pi
    ]
    
    # Optimize
    result = minimize(objective, x0, method="BFGS", bounds=bounds)
    
    return {
        'success': result.success,
        'message': result.message,
        'optimal_parameters': {
            'alpha': result.x[0],
            'delta': result.x[1],
            'phi_pi': result.x[2]
        },
        'objective_value': result.fun
    }

def estimate_with_network(env, network, data, moment_function, weight_matrix, n_simulations=1000):
    """
    Estimate model parameters using GMM with the trained value function network.
    """
    def gmm_objective(params):
        # Update parameters
        env.parameters['alpha'] = params[0]
        env.parameters['delta'] = params[1]
        env.parameters['phi_pi'] = params[2]
        
        # Simulate using the network for value function approximation
        states = []
        rewards = []
        state = env.reset()
        
        for _ in range(n_simulations):
            # Use network to predict value function
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            predicted_value = network(state_tensor).item()
            
            # Use predicted value to guide action selection
            action = env.get_policy(state)
            state, reward, done, _ = env.step(action)
            
            states.append(state)
            rewards.append(reward)
            
            if done:
                break
        
        # Compute simulated moments
        sim_moments = moment_function(env, np.array(states), np.array(rewards))
        
        # Compute GMM objective
        diff = sim_moments - data
        return diff @ weight_matrix @ diff
    
    from scipy.optimize import minimize
    
    # Initial guess and bounds
    x0 = [
        env.parameters['alpha'],
        env.parameters['delta'],
        env.parameters['phi_pi']
    ]
    bounds = [
        (0.2, 0.4),    # alpha
        (0.01, 0.05),  # delta
        (1.0, 2.0)     # phi_pi
    ]
    
    # Optimize
    result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds)
    
    return {
        'success': result.success,
        'message': result.message,
        'estimated_parameters': {
            'alpha': result.x[0],
            'delta': result.x[1],
            'phi_pi': result.x[2]
        },
        'objective_value': result.fun
    }

def main():
    # Initialize environments
    zurcher_env = ZurcherEnv()
    aiyagari_env = AiyagariEnv()
    
    # Create value function networks
    zurcher_network = ValueFunctionNetwork(input_dim=1)  # Zurcher has 1D state
    aiyagari_network = ValueFunctionNetwork(input_dim=2)  # Aiyagari has 2D state
    
    # Train on Zurcher environment
    print("Training value function on Zurcher environment...")
    train_value_function(zurcher_env, zurcher_network)
    
    # Save pre-trained weights
    torch.save(zurcher_network.state_dict(), 'zurcher_value_function.pth')
    
    # Initialize Aiyagari network with random weights
    aiyagari_network_random = ValueFunctionNetwork(input_dim=2)
    
    # Initialize Aiyagari network with transferred weights
    aiyagari_network_transfer = ValueFunctionNetwork(input_dim=2)
    # Transfer weights from first layer (assuming similar feature extraction)
    aiyagari_network_transfer.network[0].weight.data = zurcher_network.network[0].weight.data.repeat(2, 1)
    aiyagari_network_transfer.network[0].bias.data = zurcher_network.network[0].bias.data
    
    # Compare training times
    print("\nTraining Aiyagari with random initialization...")
    start_time = time.time()
    train_value_function(aiyagari_env, aiyagari_network_random)
    random_time = time.time() - start_time
    
    print("\nTraining Aiyagari with transferred weights...")
    start_time = time.time()
    train_value_function(aiyagari_env, aiyagari_network_transfer)
    transfer_time = time.time() - start_time
    
    print("\nResults:")
    print(f"Training time with random initialization: {random_time:.2f} seconds")
    print(f"Training time with transferred weights: {transfer_time:.2f} seconds")
    print(f"Speedup: {random_time/transfer_time:.2f}x")
    
    # Compare final performance
    print("\nComparing final performance:")
    
    # Test on some sample states
    test_states = [
        np.array([0.5, 0.5]),  # Low capital, low productivity
        np.array([0.8, 0.8]),  # High capital, high productivity
        np.array([0.5, 0.8]),  # Low capital, high productivity
        np.array([0.8, 0.5])   # High capital, low productivity
    ]
    
    print("\nValue function predictions:")
    print("State\t\tRandom\t\tTransferred\tTrue")
    for state in test_states:
        random_pred = aiyagari_network_random(torch.FloatTensor(state).unsqueeze(0)).item()
        transfer_pred = aiyagari_network_transfer(torch.FloatTensor(state).unsqueeze(0)).item()
        true_value = aiyagari_env.get_value_function(state)
        print(f"{state}\t{random_pred:.4f}\t{transfer_pred:.4f}\t{true_value:.4f}")
    
    # Demonstrate calibration with networks
    print("\nDemonstrating calibration with networks...")
    
    # Define calibration targets
    calibration_targets = {
        'mean_output': 1.0,
        'mean_consumption': 0.7,
        'mean_capital': 0.5
    }
    
    # Calibrate with random network
    print("\nCalibrating with random network...")
    start_time = time.time()
    random_calibration = calibrate_with_network(aiyagari_env, aiyagari_network_random, calibration_targets)
    random_calibration_time = time.time() - start_time
    
    # Calibrate with transferred network
    print("\nCalibrating with transferred network...")
    start_time = time.time()
    transfer_calibration = calibrate_with_network(aiyagari_env, aiyagari_network_transfer, calibration_targets)
    transfer_calibration_time = time.time() - start_time
    
    print("\nCalibration Results:")
    print(f"Random network calibration time: {random_calibration_time:.2f} seconds")
    print(f"Transferred network calibration time: {transfer_calibration_time:.2f} seconds")
    print(f"Calibration speedup: {random_calibration_time/transfer_calibration_time:.2f}x")
    
    # Demonstrate estimation with networks
    print("\nDemonstrating estimation with networks...")
    
    # Generate synthetic data
    np.random.seed(42)
    synthetic_data = np.random.normal(0.7, 0.1, 1000)  # Example moment data
    
    # Define moment function
    def moment_function(env, states, rewards):
        return np.array([
            np.mean(rewards),
            np.std(rewards),
            np.mean(states[:, 0])  # Mean capital
        ])
    
    # Define weight matrix
    weight_matrix = np.eye(3)
    
    # Estimate with random network
    print("\nEstimating with random network...")
    start_time = time.time()
    random_estimation = estimate_with_network(aiyagari_env, aiyagari_network_random, 
                                           synthetic_data, moment_function, weight_matrix)
    random_estimation_time = time.time() - start_time
    
    # Estimate with transferred network
    print("\nEstimating with transferred network...")
    start_time = time.time()
    transfer_estimation = estimate_with_network(aiyagari_env, aiyagari_network_transfer,
                                             synthetic_data, moment_function, weight_matrix)
    transfer_estimation_time = time.time() - start_time
    
    print("\nEstimation Results:")
    print(f"Random network estimation time: {random_estimation_time:.2f} seconds")
    print(f"Transferred network estimation time: {transfer_estimation_time:.2f} seconds")
    print(f"Estimation speedup: {random_estimation_time/transfer_estimation_time:.2f}x")

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

if __name__ == "__main__":
    main() 