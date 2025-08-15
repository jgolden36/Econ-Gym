import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List
from econgym.core.base_env import EconEnv
from typing import Iterable

class ZurcherEnv(EconEnv):
    """
    Harold Zurcher Bus Replacement Model with:
      - stochastic mileage increments (Poisson)
      - infinite horizon (no terminal)
      - random starting state
      - integer Box observation for consistency
    """
    def __init__(self,
                 max_mileage: int = 400000,  # Increased to match paper's scale
                 replace_cost: float = 11.7257,  # Original paper's replacement cost
                 beta: float = 0.9999,  # Original paper's discount factor
                 poisson_lambda: float = 1.0):  # Original paper's Poisson parameter
        super().__init__()
        self.parameters = {
            'max_mileage': max_mileage,
            'replace_cost': replace_cost,
            'beta': beta,
            'maintenance_cost_base': 0.0,  # Original paper's maintenance cost parameters
            'maintenance_cost_slope': 0.0001,  # Original paper's maintenance cost parameters
            'poisson_lambda': poisson_lambda
        }

        # Use an integer Box so PPO sees the same dtype at train & eval
        self.observation_space = spaces.Box(
            low=0,
            high=max_mileage,
            shape=(1,),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(2)
        self.np_random = np.random.default_rng()
        self.state = None
        
        # Initialize value function and policy
        self.value_function = None
        self.policy = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # random start so PPO visits all mileage states
        self.state = self.np_random.integers(
            low=0,
            high=self.parameters['max_mileage'] + 1
        )
        return np.array([self.state], dtype=np.int32), {}

    def step(self, action):
        m = int(self.state)
        a = int(action) if not isinstance(action, (list, np.ndarray)) else int(action[0])

        # if choose replace OR you hit max_mileage, force-reset
        if a == 1 or m >= self.parameters['max_mileage']:
            reward = -self.parameters['replace_cost']
            next_m = 0
        else:
            # stochastic mileage increment (Poisson process)
            jump = self.np_random.poisson(self.parameters['poisson_lambda'])
            next_m = min(m + jump, self.parameters['max_mileage'])
            reward = -self.maintenance_cost(m)

        self.state = next_m
        done = False    # infinite-horizon
        return np.array([next_m], dtype=np.int32), reward, done, False, {}

    # Optional: discrete transition model for DP solvers
    def reward_function(self, state: int, action: int) -> float:
        m = int(state)
        if action == 1 or m >= self.parameters['max_mileage']:
            return -self.parameters['replace_cost']
        return -self.maintenance_cost(m)

    def transition_probs(self, state: int, action: int) -> Iterable[tuple[int, float]]:
        m = int(state)
        if action == 1 or m >= self.parameters['max_mileage']:
            # Replace or forced reset: deterministic to 0
            yield (0, 1.0)
            return
        # Keep: Poisson jump distribution truncated at max_mileage
        max_jump = 20
        poisson_probs = np.array([
            np.exp(-self.parameters['poisson_lambda']) *
            (self.parameters['poisson_lambda'] ** j) / np.math.factorial(j)
            for j in range(max_jump)
        ])
        poisson_probs = poisson_probs / np.sum(poisson_probs)
        for j, p in enumerate(poisson_probs):
            next_m = min(m + j, self.parameters['max_mileage'])
            yield (next_m, float(p))

    def maintenance_cost(self, mileage: int) -> float:
        """
        Maintenance cost function from the original paper:
        c(x) = θ1 * x where θ1 = 0.0001
        """
        return self.parameters['maintenance_cost_slope'] * mileage

    def find_equilibrium(self, tol=1e-6, max_iter=1000):
        """Find the equilibrium using value function iteration"""
        if self.value_function is not None:
            return self.value_function, self.policy

        # Initialize value function and policy
        n_states = self.parameters['max_mileage'] + 1
        value_function = np.zeros(n_states)
        policy = np.zeros(n_states, dtype=int)
        
        # Pre-compute Poisson probabilities for jumps
        max_jump = 20  # Consider jumps up to 20
        poisson_probs = np.array([
            np.exp(-self.parameters['poisson_lambda']) * 
            (self.parameters['poisson_lambda'] ** j) / np.math.factorial(j)
            for j in range(max_jump)
        ])
        poisson_probs = poisson_probs / np.sum(poisson_probs)
        
        # Value function iteration
        for iteration in range(max_iter):
            value_function_new = np.zeros(n_states)
            policy_new = np.zeros(n_states, dtype=int)
            
            # For each state
            for m in range(n_states):
                # Value of replacing
                replace_value = -self.parameters['replace_cost'] + self.parameters['beta'] * value_function[0]
                
                # Value of not replacing
                keep_value = -self.maintenance_cost(m)
                # Add expected future value
                for jump, prob in enumerate(poisson_probs):
                    next_m = min(m + jump, self.parameters['max_mileage'])
                    keep_value += self.parameters['beta'] * prob * value_function[next_m]
                
                # Choose best action
                if replace_value > keep_value:
                    value_function_new[m] = replace_value
                    policy_new[m] = 1
                else:
                    value_function_new[m] = keep_value
                    policy_new[m] = 0
            
            # Check convergence
            if np.max(np.abs(value_function_new - value_function)) < tol:
                print(f"Value function iteration converged after {iteration + 1} iterations")
                break
                
            value_function = value_function_new
            policy = policy_new
        
        self.value_function = value_function
        self.policy = policy
        return value_function, policy

    def get_value_function(self, state: int) -> float:
        """Get the value function for a given state"""
        if self.value_function is None:
            self.find_equilibrium()
        return float(self.value_function[state])

    def get_policy(self, state: int) -> int:
        """Get the optimal policy for a given state"""
        if self.policy is None:
            self.find_equilibrium()
        return int(self.policy[state])

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current mileage: {self.state}")
            print(f"Parameters: replace_cost={self.parameters['replace_cost']:.2f}, "
                  f"maintenance_cost_slope={self.parameters['maintenance_cost_slope']:.6f}")
        return None 