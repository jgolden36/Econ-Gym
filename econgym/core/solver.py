# econgym/core/solver.py
"""
Generic routines for solving dynamic programs and reinforcement learning problems
for economic environments. Supports both traditional DP methods and RLlib algorithms.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
import gymnasium as gym
from .base_env import EconEnv

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.algorithms.algorithm import Algorithm
except ImportError:
    ray = None

def value_iteration(env: EconEnv, 
                   discount_factor: float = 0.95, 
                   max_iter: int = 1000, 
                   tol: float = 1e-6,
                   equilibrium_check: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value iteration solver for economic environments.
    
    Args:
        env: EconEnv environment
        discount_factor: Discount factor (gamma)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        equilibrium_check: Whether to check for equilibrium convergence
        
    Returns:
        Tuple of (value_function, policy)
    """
    # Strictly require tabular (discrete) spaces and a state-conditional transition interface.
    if not hasattr(env.observation_space, 'n') or not hasattr(env.action_space, 'n'):
        raise ValueError(
            "Value iteration requires discrete state and action spaces. "
            "Use a model-specific solver or discretize your environment."
        )

    # Prefer a full transition model if provided
    transition_probs = getattr(env, 'transition_probs', None)
    reward_function = getattr(env, 'reward_function', None)
    step_from_state = getattr(env, 'step_from_state', None)
    if transition_probs is None or reward_function is None:
        if step_from_state is None:
            raise NotImplementedError(
                "Provide either (transition_probs, reward_function) or step_from_state on the env."
            )
        
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize value function and policy
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    
    # Store equilibrium state if available
    equilibrium_state = None
    if equilibrium_check and hasattr(env, 'find_equilibrium'):
        try:
            equilibrium_state = env.find_equilibrium()
        except NotImplementedError:
            pass
    
    for iteration in range(max_iter):
        V_new = np.copy(V)
        
        for state in range(n_states):
            # Compute value for each action from this explicit state
            action_values = np.zeros(n_actions)
            for action in range(n_actions):
                if transition_probs is not None and reward_function is not None:
                    # Expectation over next states
                    expected = 0.0
                    r = reward_function(state, action)
                    for next_state, prob in transition_probs(state, action):
                        expected += prob * (r + discount_factor * V[int(next_state)])
                    action_values[action] = expected
                else:
                    next_state, reward, done, truncated, _ = step_from_state(state, action)
                    action_values[action] = reward + discount_factor * V[int(next_state)]
            
            # Update value function and policy
            V_new[state] = np.max(action_values)
            policy[state] = np.argmax(action_values)
        
        # Check convergence
        if np.max(np.abs(V - V_new)) < tol:
            print(f"Value iteration converged in {iteration + 1} iterations")
            
            # Check if we've reached equilibrium
            if equilibrium_state is not None:
                current_state = env._equilibrium_state
                if current_state is not None and np.allclose(current_state, equilibrium_state, atol=tol):
                    print("Reached economic equilibrium")
            break
            
        V = V_new
    
    return V, policy

def policy_iteration(env: EconEnv,
                    discount_factor: float = 0.95,
                    max_iter: int = 1000,
                    tol: float = 1e-6,
                    equilibrium_check: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Policy iteration solver for economic environments.
    
    Args:
        env: EconEnv environment
        discount_factor: Discount factor (gamma)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        equilibrium_check: Whether to check for equilibrium convergence
        
    Returns:
        Tuple of (value_function, policy)
    """
    if not hasattr(env.observation_space, 'n') or not hasattr(env.action_space, 'n'):
        raise ValueError(
            "Policy iteration requires discrete state and action spaces. "
            "Use a model-specific solver or discretize your environment."
        )

    transition_probs = getattr(env, 'transition_probs', None)
    reward_function = getattr(env, 'reward_function', None)
    step_from_state = getattr(env, 'step_from_state', None)
    if transition_probs is None or reward_function is None:
        if step_from_state is None:
            raise NotImplementedError(
                "Provide either (transition_probs, reward_function) or step_from_state on the env."
            )
        
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize random policy
    policy = np.random.randint(0, n_actions, size=n_states)
    V = np.zeros(n_states)
    
    # Store equilibrium state if available
    equilibrium_state = None
    if equilibrium_check and hasattr(env, 'find_equilibrium'):
        try:
            equilibrium_state = env.find_equilibrium()
        except NotImplementedError:
            pass
    
    for iteration in range(max_iter):
        # Policy evaluation
        while True:
            V_new = np.copy(V)
            for state in range(n_states):
                action = policy[state]
                if transition_probs is not None and reward_function is not None:
                    r = reward_function(state, action)
                    expected = 0.0
                    for next_state, prob in transition_probs(state, action):
                        expected += prob * (r + discount_factor * V[int(next_state)])
                    V_new[state] = expected
                else:
                    next_state, reward, done, truncated, _ = step_from_state(state, action)
                    V_new[state] = reward + discount_factor * V[int(next_state)]
            
            if np.max(np.abs(V - V_new)) < tol:
                break
            V = V_new
        
        # Policy improvement
        policy_stable = True
        for state in range(n_states):
            old_action = policy[state]
            
            # Compute action values from this explicit state
            action_values = np.zeros(n_actions)
            for action in range(n_actions):
                if transition_probs is not None and reward_function is not None:
                    r = reward_function(state, action)
                    expected = 0.0
                    for next_state, prob in transition_probs(state, action):
                        expected += prob * (r + discount_factor * V[int(next_state)])
                    action_values[action] = expected
                else:
                    next_state, reward, done, truncated, _ = step_from_state(state, action)
                    action_values[action] = reward + discount_factor * V[int(next_state)]
            
            # Update policy
            policy[state] = np.argmax(action_values)
            if old_action != policy[state]:
                policy_stable = False
        
        if policy_stable:
            print(f"Policy iteration converged in {iteration + 1} iterations")
            
            # Check if we've reached equilibrium
            if equilibrium_state is not None:
                current_state = env._equilibrium_state
                if current_state is not None and np.allclose(current_state, equilibrium_state, atol=tol):
                    print("Reached economic equilibrium")
            break
    
    return V, policy

class EconRLlibSolver:
    """
    Wrapper for RLlib algorithms to solve economic reinforcement learning problems.
    """
    
    def __init__(self, 
                 algorithm: str = "PPO",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize RLlib solver for economic environments.
        
        Args:
            algorithm: RLlib algorithm to use ("PPO", "DQN", or "SAC")
            config: Optional configuration dictionary
        """
        if ray is None:
            raise ImportError("Ray and RLlib must be installed to use EconRLlibSolver")
            
        self.algorithm = algorithm
        self.config = config or {}
        
    def train(self, 
              env: EconEnv,
              num_iterations: int = 100,
              checkpoint_freq: int = 10,
              equilibrium_check: bool = True) -> Algorithm:
        """
        Train an RLlib algorithm on the economic environment.
        
        Args:
            env: EconEnv environment
            num_iterations: Number of training iterations
            checkpoint_freq: Frequency of checkpointing
            equilibrium_check: Whether to check for equilibrium convergence
            
        Returns:
            Trained algorithm
        """
        # Configure algorithm with economic-specific settings
        if self.algorithm == "PPO":
            config = (PPOConfig()
                     .environment(env)
                     .training(**self.config)
                     .framework("torch")
                     .rollouts(num_rollout_workers=2)
                     .evaluation(evaluation_interval=5)
                     .training(gamma=0.99,  # Economic discount factor
                             lr=0.0003,
                             train_batch_size=4000))
        elif self.algorithm == "DQN":
            config = (DQNConfig()
                     .environment(env)
                     .training(**self.config)
                     .framework("torch")
                     .rollouts(num_rollout_workers=2)
                     .evaluation(evaluation_interval=5)
                     .training(gamma=0.99,
                             lr=0.0001,
                             train_batch_size=32))
        elif self.algorithm == "SAC":
            config = (SACConfig()
                     .environment(env)
                     .training(**self.config)
                     .framework("torch")
                     .rollouts(num_rollout_workers=2)
                     .evaluation(evaluation_interval=5)
                     .training(gamma=0.99,
                             lr=0.0003,
                             train_batch_size=256))
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Create and train algorithm
        algo = config.build()
        
        # Store equilibrium state if available
        equilibrium_state = None
        if equilibrium_check and hasattr(env, 'find_equilibrium'):
            try:
                equilibrium_state = env.find_equilibrium()
            except NotImplementedError:
                pass
        
        for i in range(num_iterations):
            result = algo.train()
            print(f"Iteration {i+1}/{num_iterations}")
            print(f"Episode reward mean: {result['episode_reward_mean']}")
            
            # Check for equilibrium convergence
            if equilibrium_state is not None:
                current_state = env._equilibrium_state
                if current_state is not None and np.allclose(current_state, equilibrium_state, atol=1e-6):
                    print("Reached economic equilibrium")
            
            if (i + 1) % checkpoint_freq == 0:
                checkpoint = algo.save()
                print(f"Checkpoint saved at {checkpoint}")
        
        return algo
    
    def evaluate(self, 
                algo: Algorithm,
                env: EconEnv,
                num_episodes: int = 10,
                equilibrium_check: bool = True) -> Dict[str, float]:
        """
        Evaluate trained algorithm on economic environment.
        
        Args:
            algo: Trained algorithm
            env: EconEnv environment
            num_episodes: Number of evaluation episodes
            equilibrium_check: Whether to check for equilibrium convergence
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        equilibrium_reached = 0
        
        # Get equilibrium state if available
        equilibrium_state = None
        if equilibrium_check and hasattr(env, 'find_equilibrium'):
            try:
                equilibrium_state = env.find_equilibrium()
            except NotImplementedError:
                pass
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = algo.compute_single_action(obs)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            
            # Check if equilibrium was reached
            if equilibrium_state is not None:
                current_state = env._equilibrium_state
                if current_state is not None and np.allclose(current_state, equilibrium_state, atol=1e-6):
                    equilibrium_reached += 1
        
        metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }
        
        if equilibrium_check and equilibrium_state is not None:
            metrics["equilibrium_reached_ratio"] = equilibrium_reached / num_episodes
        
        return metrics

# Example usage
if __name__ == "__main__":
    from examples.ml_estimation_example import SimpleEconEnv
    
    # Create a simple economic environment
    env = SimpleEconEnv()
    
    # Try value iteration
    try:
        V, policy = value_iteration(env, equilibrium_check=True)
        print("\nValue iteration results:")
        print(f"Value function shape: {V.shape}")
        print(f"Policy shape: {policy.shape}")
    except ValueError as e:
        print(f"Value iteration failed: {e}")
    
    # Try RLlib solver
    try:
        solver = EconRLlibSolver(algorithm="PPO")
        algo = solver.train(env, num_iterations=10, equilibrium_check=True)
        results = solver.evaluate(algo, env, equilibrium_check=True)
        print("\nEvaluation results:", results)
    except ImportError as e:
        print(f"RLlib solver failed: {e}")