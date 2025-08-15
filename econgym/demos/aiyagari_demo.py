import numpy as np
from econgym.envs.aiyagari_env import AiyagariEnv
from econgym.wrappers.value_function import ValueFunctionWrapper
from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

def main():
    # Create the base environment
    env = AiyagariEnv(r=0.02, a_max=20.0, shock_vals=[0.5, 1.5], shock_probs=[0.5, 0.5])
    
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
    print("Average assets:", np.mean(states[:, 0]))
    print("Average consumption:", np.mean(rewards))
    
    # Calibrate the model
    print("\nCalibrating model...")
    targets = {
        'mean_assets': 10.0,
        'std_assets': 2.0
    }
    calibration_results = env.calibrate(targets)
    print("Calibration results:", calibration_results)
    
    # Estimate using GMM
    print("\nEstimating model...")
    data = np.array([10.0, 2.0])  # Example empirical moments
    moment_function = lambda env, states: np.array([
        np.mean(states[:, 0]),  # mean assets
        np.std(states[:, 0])    # std assets
    ])
    weight_matrix = np.eye(2)
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
            episode_reward += reward
            
            # Train dynamics model
            mbrl_env.train_dynamics_model()
            
            if done or truncated:
                break
                
            state = next_state
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    main() 