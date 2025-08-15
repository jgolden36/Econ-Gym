import numpy as np
from econgym.envs.zurcher_env import ZurcherEnv
from econgym.wrappers.value_function import ValueFunctionWrapper
from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

def main():
    # Create the base environment
    env = ZurcherEnv(max_mileage=100, replace_cost=500, beta=0.9)
    
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
    print("Average mileage:", np.mean(states))
    print("Average replacement frequency:", np.mean(policy))
    
    # Calibrate the model
    print("\nCalibrating model...")
    targets = {
        'replacement_frequency': 0.2,
        'avg_mileage': 30.0
    }
    calibration_results = env.calibrate(targets)
    print("Calibration results:", calibration_results)
    
    # Estimate using GMM
    print("\nEstimating model...")
    data = np.array([0.2, 30.0])  # Example empirical moments
    moment_function = lambda env, policy: np.array([
        np.mean(policy),  # replacement frequency
        np.mean([m for m, a in enumerate(policy) if a == 0])  # average mileage
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
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    main() 