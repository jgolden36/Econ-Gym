import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import bisect
from econgym.core.base_env import EconEnv


class RepeatedPrisonersEnv(EconEnv):
    """
    A Gymnasium environment simulating a repeated Prisoner's Dilemma game.
    """
    def __init__(self, rounds=10):
        super(RepeatedPrisonersEnv, self).__init__()
        self.rounds = rounds
        self.current_round = 0

        # Two players, each {0,1}
        self.action_space = spaces.MultiDiscrete([2, 2])
        self.observation_space = spaces.Dict({
            "round": spaces.Discrete(rounds + 1),
            "last_actions": spaces.MultiDiscrete([2, 2])
        })

        self.history = []
        self.total_reward = np.array([0, 0])
        self.last_actions = (0, 0)

        self.payoff_matrix = {
            (0, 0): (3, 3),
            (0, 1): (0, 5),
            (1, 0): (5, 0),
            (1, 1): (1, 1)
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_round = 0
        self.history = []
        self.total_reward = np.array([0, 0])
        self.last_actions = (0, 0)
        return {"round": self.current_round, "last_actions": [0, 0]}, {}

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = tuple(int(x) for x in actions.tolist())
        if not (isinstance(actions, (list, tuple)) and len(actions) == 2):
            raise ValueError("Actions must be a pair (a1, a2) where each is 0 or 1.")
        if actions not in self.payoff_matrix:
            raise ValueError("Actions must be 0 (Cooperate) or 1 (Defect).")

        reward_pair = self.payoff_matrix[actions]
        self.total_reward += np.array(reward_pair)
        self.history.append({"round": self.current_round, "actions": actions, "reward": reward_pair})
        self.last_actions = actions
        self.current_round += 1

        done = self.current_round >= self.rounds
        obs = {"round": self.current_round, "last_actions": list(actions)}
        info = {"total_reward": self.total_reward.copy(), "history": list(self.history)}
        return obs, reward_pair, done, False, info

    def check_equilibrium(self):
        return self.last_actions == (1, 1)

# Example usage:
if __name__ == "__main__":
    env = RepeatedPrisonersEnv(rounds=5)
    obs = env.reset()
    print("Initial observation:", obs)
    
    done = False
    while not done:
        # For demonstration, let the players choose actions arbitrarily.
        # Here, we'll have both players alternate between cooperating (0) and defecting (1).
        actions = (1, 1) if env.current_round % 2 == 0 else (0, 0)
        obs, reward, done, info = env.step(actions)
        eq_status = env.check_equilibrium()
        print(f"Round {obs['round']}: Actions: {actions}, Reward: {reward}, Equilibrium? {eq_status}")
    
    print("Game finished. Total rewards:", info["total_reward"])

# Example usage:
if __name__ == "__main__":
    env = RepeatedPrisonersEnv(rounds=5)
    obs = env.reset()
    print("Initial observation:", obs)
    
    equilibrium_strategy = env.find_equilibrium()
    
    done = False
    while not done:
        # For demonstration, both players use the equilibrium strategy.
        action = equilibrium_strategy(obs)
        obs, reward, done, info = env.step((action, action))
        print(f"Round {obs['round']}: Action: {action}, Reward: {reward}")
    
    print("Game finished. Total rewards:", info["total_reward"])