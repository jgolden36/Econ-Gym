import numpy as np
from econgym.envs.bbl_env import BBLGameEnv
from econgym.wrappers.value_function import ValueFunctionWrapper
from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

def main():
    # Create the base environment
    env = BBLGameEnv(n_firms=3, state_dim=1, beta=0.95)
    
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
    policy, V = env.find_equilibrium()
    print("Equilibrium policy shape:", policy.shape)
    print("Equilibrium value function shape:", V.shape)
    
    # Simulate using equilibrium policy
    print("\nSimulating with equilibrium policy...")
    states, rewards = env.simulate(n_periods=1000)
    print("Average firm states:", np.mean(states, axis=0))
    print("Average firm profits:", np.mean(rewards, axis=0))
    
    # Calibrate the model
    print("\nCalibrating model...")
    targets = {
        'avg_price': 1.5,
        'avg_profit': 2.0,
        'market_concentration': 0.4
    }
    calibration_results = env.calibrate(targets)
    print("Calibration results:", calibration_results)
    
    # Estimate using GMM
    print("\nEstimating model...")
    data = np.array([1.5, 2.0, 0.4])  # Example empirical moments
    moment_function = lambda env, states, rewards: np.array([
        np.mean([np.mean(a) for a in states]),  # average price
        np.mean(rewards),                      # average profit
        np.mean([np.max(a) / np.sum(a) for a in states])  # market concentration
    ])
    weight_matrix = np.eye(3)
    estimation_results = env.estimate(data, moment_function, weight_matrix)
    print("Estimation results:", estimation_results)
    
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

if __name__ == "__main__":
    main() 