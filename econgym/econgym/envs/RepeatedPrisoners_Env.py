import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import bisect
from econgym.core.base_env import EconEnv


class RepeatedPrisonersEnv(EconEnv):
    """
    A Gym environment simulating a repeated Prisoner's Dilemma game.
    
    - Each round, two players choose an action: 0 (Cooperate) or 1 (Defect).
    - The payoff matrix is defined as:
        (0, 0): (3, 3)
        (0, 1): (0, 5)
        (1, 0): (5, 0)
        (1, 1): (1, 1)
    - The game is repeated for a fixed number of rounds.
    """
    def __init__(self, rounds=10):
        super(RepeatedPrisonersEnv, self).__init__()
        self.rounds = rounds
        self.current_round = 0
        
        # Define action space: each agent can choose either 0 (Cooperate) or 1 (Defect)
        self.action_space = spaces.Discrete(2)
        
        # Define observation space: track the current round and the last actions taken.
        self.observation_space = spaces.Dict({
            "round": spaces.Discrete(rounds + 1),
            "last_actions": spaces.MultiDiscrete([2, 2])
        })
        
        # Initialize history and total rewards
        self.history = []
        self.total_reward = np.array([0, 0])
        self.last_actions = (0, 0)  # Default starting observation
        
        # Define the payoff matrix for the Prisoner's Dilemma
        self.payoff_matrix = {
            (0, 0): (3, 3),
            (0, 1): (0, 5),
            (1, 0): (5, 0),
            (1, 1): (1, 1)
        }
    
    def reset(self, *, seed=None, options=None):
        """Resets the environment to the initial state."""
        # Handle optional seeding locally to avoid calling abstract base reset
        if seed is not None:
            np.random.seed(seed)
        self.current_round = 0
        self.history = []
        self.total_reward = np.array([0, 0])
        self.last_actions = (0, 0)
        # Return observation and info per Gymnasium API
        return {"round": self.current_round, "last_actions": [0, 0]}, {}
    
    def step(self, actions):
        """
        Executes one round of the game.
        
        Parameters:
            actions (tuple or list): A pair of actions (one for each player), 
                                     where each action is 0 (Cooperate) or 1 (Defect).
        
        Returns:
            observation (dict): New state of the environment.
            reward (tuple): Rewards for the two players based on their actions.
            done (bool): Flag indicating if the game has ended.
            info (dict): Additional information including the full history and cumulative rewards.
        """
        if not (isinstance(actions, (list, tuple)) and len(actions) == 2):
            raise ValueError("Actions must be a list or tuple of two elements.")
        
        # Get the reward based on the current actions.
        reward = self.payoff_matrix.get((actions[0], actions[1]))
        if reward is None:
            raise ValueError("Invalid actions provided. Actions must be 0 (Cooperate) or 1 (Defect).")
        
        # Update cumulative rewards and record the history.
        self.total_reward += np.array(reward)
        self.history.append({
            "round": self.current_round,
            "actions": actions,
            "reward": reward
        })
        self.last_actions = actions
        self.current_round += 1
        
        # Determine if the game is over.
        done = self.current_round >= self.rounds
        
        # Create new observation.
        observation = {"round": self.current_round, "last_actions": list(actions)}
        
        # Additional info can include the history and cumulative rewards.
        info = {"total_reward": self.total_reward, "history": self.history}
        
        # Return 5-tuple per Gymnasium
        return observation, reward, done, False, info
    
    def check_equilibrium(self):
        """
        Checks whether the most recent round's actions constitute an equilibrium outcome.
        
        In the Prisoner's Dilemma, mutual defection (actions (1, 1)) is considered the Nash equilibrium.
        
        Returns:
            bool: True if the last actions are (1, 1), False otherwise.
        """
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