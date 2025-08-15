import gymnasium as gym
import abc
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

class EconEnv(gym.Env, metaclass=abc.ABCMeta):
    """
    Abstract base class for EconGym environments.
    Inherit from this to ensure a consistent interface for economic models.
    """

    def __init__(self):
        """
        Initialize the economic environment.
        """
        super().__init__()
        self._parameters: Dict[str, Any] = {}
        self._calibration_targets: Dict[str, Any] = {}
        self._estimation_data: Optional[np.ndarray] = None
        self._equilibrium_state: Optional[np.ndarray] = None
        self._num_params: int = 0

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, params: Dict[str, Any]):
        """
        Set model parameters.
        """
        self._parameters = params
        self._num_params = len(params)

    @property
    def calibration_targets(self) -> Dict[str, Any]:
        """
        Get the calibration targets for the model.
        """
        return self._calibration_targets

    @calibration_targets.setter
    def calibration_targets(self, targets: Dict[str, Any]):
        """
        Set calibration targets for the model.
        """
        self._calibration_targets = targets

    @property
    def estimation_data(self) -> Optional[np.ndarray]:
        """
        Get the data used for estimation.
        """
        return self._estimation_data

    @estimation_data.setter
    def estimation_data(self, data: np.ndarray):
        """
        Set the data used for estimation.
        """
        self._estimation_data = data

    @property
    def equilibrium_state(self) -> Optional[np.ndarray]:
        """
        Get the current equilibrium state.
        """
        return self._equilibrium_state

    @equilibrium_state.setter
    def equilibrium_state(self, state: np.ndarray):
        """
        Set the current equilibrium state.
        """
        self._equilibrium_state = state

    @abc.abstractmethod
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        Must return observation of the initial state and info dict.
        
        Args:
            seed: Optional random seed
            options: Optional configuration
            
        Returns:
            Tuple of (observation, info)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        """
        Take an action, return (next_state, reward, terminated, truncated, info).
        """
        raise NotImplementedError

    def find_equilibrium(self):
        """
        Optional: If your environment requires an equilibrium solve,
        you can implement a default method or raise NotImplementedError.
        """
        raise NotImplementedError("EconEnv.find_equilibrium not implemented.")

    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """
        Calibrate the model parameters to match target moments.
        
        Args:
            targets: Dictionary of target moments to match
            method: Optimization method to use
            **kwargs: Additional arguments for the optimizer
            
        Returns:
            Dictionary containing calibration results
        """
        raise NotImplementedError("EconEnv.calibrate not implemented.")

    def estimate(self, data: np.ndarray, moment_function: callable, 
                weight_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate model parameters using GMM.
        
        Args:
            data: Empirical data
            moment_function: Function that computes moments
            weight_matrix: Weighting matrix for GMM
            **kwargs: Additional arguments for estimation
            
        Returns:
            Dictionary containing estimation results
        """
        raise NotImplementedError("EconEnv.estimate not implemented.")

    def simulate(self, n_periods: int, policy: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model for a given number of periods.
        
        Args:
            n_periods: Number of periods to simulate
            policy: Optional policy function to use for actions
            
        Returns:
            Tuple of (states, rewards) arrays
        """
        states = []
        rewards = []
        state, _ = self.reset()
        
        for _ in range(n_periods):
            if policy is None:
                action = self.action_space.sample()
            else:
                action = policy(state)
                
            state, reward, done, truncated, _ = self.step(action)
            states.append(state)
            rewards.append(reward)
            
            if done or truncated:
                break
                
        return np.array(states), np.array(rewards)

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        """
        raise NotImplementedError("EconEnv.render not implemented.")

    def close(self):
        """
        Clean up any resources used by the environment.
        """
        pass


class EconEnvWrapper(EconEnv):
    """
    A wrapper for environments that conform to the Gym interface (or EconEnv)
    and extends them with an equilibrium finding method.
    
    If the wrapped environment already implements find_equilibrium, that
    method will be used. Otherwise, if an equilibrium_solver function is
    provided during initialization, it will be used to compute the equilibrium.
    """

    def __init__(self, env, equilibrium_solver=None):
        """
        Initialize the EconEnvWrapper.
        
        :param env: An instance of gym.Env or EconEnv that represents a specific economic model.
        :param equilibrium_solver: Optional. A function that accepts the environment as its argument
                                   and returns an equilibrium value.
        """
        super().__init__()
        self.env = env
        self.equilibrium_solver = equilibrium_solver
        # Propagate observation and action spaces from the wrapped environment.
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Propagate parameters and calibration targets if they exist
        if hasattr(env, 'parameters'):
            self._parameters = env.parameters
        if hasattr(env, 'calibration_targets'):
            self._calibration_targets = env.calibration_targets

    def reset(self, seed=None, options=None):
        """
        Reset the wrapped environment.
        
        Args:
            seed: Optional random seed
            options: Optional configuration
            
        Returns:
            Tuple of (observation, info)
        """
        # Delegate reset to the wrapped environment.
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """
        Take a step in the wrapped environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Delegate step to the wrapped environment.
        return self.env.step(action)

    def find_equilibrium(self):
        """
        Finds the equilibrium for the wrapped environment. It first checks if the underlying
        environment has its own implementation. If not, it uses the provided equilibrium_solver.
        """
        # If the underlying environment defines find_equilibrium, try using it.
        if hasattr(self.env, "find_equilibrium"):
            try:
                eq = self.env.find_equilibrium()
                return eq
            except NotImplementedError:
                # Fall through to try the provided solver.
                pass
        # If an equilibrium_solver function was provided, use it.
        if self.equilibrium_solver is not None:
            return self.equilibrium_solver(self.env)
        # Otherwise, indicate that no equilibrium method is available.
        raise NotImplementedError("No equilibrium solver available for the wrapped environment.")

    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """
        Calibrate the wrapped environment if it supports calibration.
        """
        if hasattr(self.env, "calibrate"):
            return self.env.calibrate(targets, method, **kwargs)
        raise NotImplementedError("Wrapped environment does not support calibration.")

    def estimate(self, data: np.ndarray, moment_function: callable, 
                weight_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate parameters of the wrapped environment if it supports estimation.
        """
        if hasattr(self.env, "estimate"):
            return self.env.estimate(data, moment_function, weight_matrix, **kwargs)
        raise NotImplementedError("Wrapped environment does not support estimation.")

    def simulate(self, n_periods: int, policy: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the wrapped environment.
        """
        if hasattr(self.env, "simulate"):
            return self.env.simulate(n_periods, policy)
        return super().simulate(n_periods, policy)

    def render(self, mode="human"):
        """
        Render the wrapped environment if it supports rendering.
        """
        if hasattr(self.env, "render"):
            return self.env.render(mode)
        raise NotImplementedError("Wrapped environment does not support rendering.")

    def close(self):
        """
        Close the wrapped environment.
        """
        if hasattr(self.env, "close"):
            self.env.close()


# ===== Example Usage =====
if __name__ == "__main__":
    # Define a specific economic environment that already implements find_equilibrium.
    class MySpecificEconEnv(EconEnv):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)
            self.state = None

        def reset(self, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            self.state = np.random.uniform(0, 10)
            return np.array([self.state], dtype=np.float32), {}

        def step(self, action):
            # Simple dynamics: increase or decrease the state.
            if action == 0:
                self.state -= 0.5
            else:
                self.state += 0.5
            # Reward is higher the closer the state is to 5.0.
            target = 5.0
            reward = -abs(self.state - target)
            done = abs(self.state - target) < 0.1
            truncated = False
            return np.array([self.state], dtype=np.float32), reward, done, truncated, {}

        def find_equilibrium(self):
            # For this example, the equilibrium is defined as state = 5.0.
            return 5.0

    # Create an instance of the specific environment.
    specific_env = MySpecificEconEnv()

    # Option 1: Wrap an environment that already has find_equilibrium implemented.
    wrapped_env = EconEnvWrapper(specific_env)
    print("Equilibrium from wrapped specific environment:", wrapped_env.find_equilibrium())

    # Option 2: Wrap an environment that does not implement find_equilibrium and supply a solver.
    class BasicEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)
            self.state = np.random.uniform(0, 10)

        def reset(self, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            self.state = np.random.uniform(0, 10)
            return np.array([self.state], dtype=np.float32), {}

        def step(self, action):
            self.state += 1 if action == 1 else -1
            reward = -abs(self.state - 5)
            done = self.state < 0 or self.state > 10
            truncated = False
            return np.array([self.state], dtype=np.float32), reward, done, truncated, {}

    # Define an equilibrium solver function for BasicEnv.
    def basic_env_equilibrium_solver(env):
        # In this toy example, define equilibrium as state = 5.
        return 5.0

    basic_env = BasicEnv()
    wrapped_basic_env = EconEnvWrapper(basic_env, equilibrium_solver=basic_env_equilibrium_solver)
    print("Equilibrium from wrapped basic environment:", wrapped_basic_env.find_equilibrium())