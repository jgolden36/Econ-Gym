import numpy as np
from econgym.core.base_env import EconEnv
from econgym.core.MLestimation import MLestimation
import gymnasium.spaces as spaces

class SimpleEconEnv(EconEnv):
    """
    A simple economic environment for demonstration purposes.
    The environment has a single parameter theta that affects the reward distribution.
    """
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.parameters = {'theta': 0.0}  # Initialize parameter
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([0.0], dtype=np.float32)
        return self.state, {}
        
    def step(self, action):
        # Simple dynamics: state follows a random walk with drift
        drift = self.parameters['theta']
        self.state += drift + np.random.normal(0, 1)
        
        # Reward is higher when state is close to 0
        reward = -np.abs(self.state[0])
        
        done = False
        truncated = False
        info = {}
        
        return self.state, reward, done, truncated, info
        
    def transition_probability(self, obs, action, next_obs):
        """
        Compute the transition probability P(next_obs | obs, action)
        """
        drift = self.parameters['theta']
        mean = obs + drift
        return np.exp(-0.5 * (next_obs - mean)**2) / np.sqrt(2 * np.pi)
        
    def reward_probability(self, obs, action, reward):
        """
        Compute the reward probability P(reward | obs, action)
        """
        expected_reward = -np.abs(obs[0])
        return np.exp(-0.5 * (reward - expected_reward)**2) / np.sqrt(2 * np.pi)
        
    def _get_expected_reward(self, obs, action):
        """
        Get the expected reward for a state-action pair
        """
        return -np.abs(obs[0])

def main():
    # Create environment
    env = SimpleEconEnv()
    
    # Create ML estimation solver
    mle_solver = MLestimation(env, acceleration=None)  # Can use 'numba' or 'jax' if available
    
    # Collect some data
    data = mle_solver.collect_data(num_episodes=50, max_steps=20)
    
    # Initial parameter guess
    initial_params = np.array([0.0])
    
    # Run estimation
    estimated_params, opt_result = mle_solver.solve(
        initial_params,
        method='L-BFGS-B',
        options={'disp': True}
    )
    
    print("\nEstimation Results:")
    print(f"True parameter value: {env.parameters['theta']}")
    print(f"Estimated parameter value: {estimated_params[0]}")
    print(f"Optimization success: {opt_result.success}")
    print(f"Final objective value: {opt_result.fun}")
    
    # Test the estimated parameters
    env.parameters = {'theta': estimated_params[0]}
    test_states, test_rewards = env.simulate(n_periods=100)
    print(f"\nTest simulation mean reward: {np.mean(test_rewards)}")

if __name__ == "__main__":
    main() 