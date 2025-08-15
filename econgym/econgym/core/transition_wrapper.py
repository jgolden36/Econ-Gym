import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from econgym.core.base_env import EconEnv
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


@dataclass
class TransitionConfig:
    """Configuration for transition analysis."""
    # Transition parameters
    num_periods: int = 200  # Number of periods to simulate
    num_agents: int = 1000  # Number of agents to track
    burn_in: int = 50      # Number of periods to burn in before transition
    transition_period: int = 100  # Period at which the transition occurs
    
    # Analysis parameters
    compute_moments: bool = True  # Whether to compute moments during transition
    moments_to_track: List[str] = None  # Which moments to track
    save_path: Optional[str] = None  # Path to save transition plots
    
    # Optional: function to compute custom moments
    custom_moment_fn: Optional[Callable] = None


class TransitionWrapper(gym.Wrapper):
    """
    A wrapper for EconEnv that adds functionality to analyze transitions between steady states.
    This is particularly useful for analyzing the effects of policy changes (e.g., tax changes)
    in models like Aiyagari.
    """
    
    def __init__(self, env: EconEnv, config: Optional[TransitionConfig] = None):
        """
        Initialize the transition wrapper.
        
        Args:
            env: The underlying economic environment
            config: Configuration for transition analysis
        """
        super().__init__(env)
        self.config = config or TransitionConfig()
        
        # Initialize tracking variables
        self.transition_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'moments': []
        }
        
        # Store initial parameters
        self.initial_parameters = env.parameters.copy() if hasattr(env, 'parameters') else {}
        
        # Propagate spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Propagate parameters
        if hasattr(env, 'parameters'):
            self._parameters = env.parameters
        
        self.current_period = 0
        self.transition_occurred = False
        
        # Initialize arrays to store transition data
        self.mean_state = np.zeros(self.config.num_periods)
        self.std_state = np.zeros(self.config.num_periods)
        self.mean_action = np.zeros(self.config.num_periods)
        self.mean_reward = np.zeros(self.config.num_periods)
        
        # Store initial environment parameters
        self.initial_params = env.parameters.copy()
    
    def reset(self, seed=None, options=None):
        """Reset the environment and clear transition data."""
        self.transition_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'moments': []
        }
        self.current_period = 0
        self.transition_occurred = False
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Take a step in the environment and store transition data."""
        # Check if we need to switch to final parameters
        if self.current_period == self.config.transition_period and not self.transition_occurred:
            self.transition_occurred = True
            # Update environment parameters for transition
            if hasattr(self.env, 'parameters'):
                self.env.parameters.update(self.final_params)
        
        # Take step
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # Store transition data
        self.transition_data['states'].append(next_state)
        self.transition_data['actions'].append(action)
        self.transition_data['rewards'].append(reward)
        
        # Compute moments if enabled
        if self.config.compute_moments:
            self.mean_state[self.current_period] = np.mean(next_state[1])  # Mean assets
            self.std_state[self.current_period] = np.std(next_state[1])    # Asset volatility
            self.mean_action[self.current_period] = np.mean(action)        # Mean savings rate
            self.mean_reward[self.current_period] = reward                 # Mean utility
        
        self.current_period += 1
        return next_state, reward, done, truncated, info
    
    def _compute_moments(self, state: np.ndarray, action: np.ndarray, reward: float) -> Dict[str, float]:
        """Compute moments from the current state and action."""
        moments = {}
        
        # Default moments
        if isinstance(state, np.ndarray):
            moments['mean_state'] = np.mean(state)
            moments['std_state'] = np.std(state)
            if len(state.shape) > 1:
                moments['state_correlation'] = np.corrcoef(state.T)[0, 1]
        
        if isinstance(action, np.ndarray):
            moments['mean_action'] = np.mean(action)
            moments['std_action'] = np.std(action)
        
        moments['reward'] = reward
        
        # Custom moments
        if self.config.custom_moment_fn is not None:
            custom_moments = self.config.custom_moment_fn(state, action, reward)
            moments.update(custom_moments)
        
        return moments
    
    def simulate_transition(self, 
                          parameter_change: Dict[str, Any],
                          policy: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Simulate a transition between steady states following a parameter change.
        
        Args:
            parameter_change: Dictionary of parameters to change and their new values
            policy: Optional policy function to use for actions
            
        Returns:
            Dictionary containing transition data and analysis
        """
        # Store initial state
        initial_state, _ = self.env.reset()
        
        # Burn in to initial steady state
        for _ in range(self.config.burn_in):
            if policy is None:
                action = self.env.action_space.sample()
            else:
                action = policy(initial_state)[0]  # PPO predict returns (action, state)
            initial_state, _, _, _, _ = self.env.step(action)
        
        # Simulate transition
        current_state = initial_state
        for t in range(self.config.num_periods):
            # Apply parameter change at transition period
            if t == self.config.transition_period:
                if hasattr(self.env, 'parameters'):
                    for param, value in parameter_change.items():
                        self.env.parameters[param] = value
            
            # Take step
            if policy is None:
                action = self.env.action_space.sample()
            else:
                action = policy(current_state)[0]  # PPO predict returns (action, state)
            
            current_state, reward, done, truncated, _ = self.env.step(action)
            
            if done or truncated:
                break
        
        # Analyze transition
        analysis = self._analyze_transition()
        
        # Plot transition if save path is provided
        if self.config.save_path:
            self._plot_transition(analysis)
        
        return analysis
    
    def _analyze_transition(self) -> Dict[str, Any]:
        """Analyze the transition data."""
        analysis = {
            'initial_steady_state': {},
            'final_steady_state': {},
            'transition_path': {},
            'moments': {}
        }
        
        # Compute steady states
        burn_in = self.config.burn_in
        transition = self.config.transition_period
        
        # Initial steady state
        initial_states = self.transition_data['states'][:burn_in]
        initial_actions = self.transition_data['actions'][:burn_in]
        initial_rewards = self.transition_data['rewards'][:burn_in]
        
        analysis['initial_steady_state'] = {
            'mean_state': np.mean(initial_states),
            'std_state': np.std(initial_states),
            'mean_action': np.mean(initial_actions),
            'mean_reward': np.mean(initial_rewards)
        }
        
        # Final steady state
        final_states = self.transition_data['states'][transition:]
        final_actions = self.transition_data['actions'][transition:]
        final_rewards = self.transition_data['rewards'][transition:]
        
        analysis['final_steady_state'] = {
            'mean_state': np.mean(final_states),
            'std_state': np.std(final_states),
            'mean_action': np.mean(final_actions),
            'mean_reward': np.mean(final_rewards)
        }
        
        # Transition path
        if self.config.compute_moments:
            moments = self.transition_data['moments']
            for moment in moments[0].keys():
                values = [m[moment] for m in moments]
                analysis['transition_path'][moment] = values
        
        return analysis
    
    def _plot_transition(self, analysis: Dict[str, Any]):
        """Plot the transition analysis."""
        if not self.config.compute_moments:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Transition Analysis')
        
        # Plot state moments
        axes[0, 0].plot(analysis['transition_path']['mean_state'])
        axes[0, 0].set_title('Mean State')
        axes[0, 0].axvline(x=self.config.transition_period, color='r', linestyle='--')
        
        axes[0, 1].plot(analysis['transition_path']['std_state'])
        axes[0, 1].set_title('State Standard Deviation')
        axes[0, 1].axvline(x=self.config.transition_period, color='r', linestyle='--')
        
        # Plot action and reward moments
        axes[1, 0].plot(analysis['transition_path']['mean_action'])
        axes[1, 0].set_title('Mean Action')
        axes[1, 0].axvline(x=self.config.transition_period, color='r', linestyle='--')
        
        axes[1, 1].plot(analysis['transition_path']['reward'])
        axes[1, 1].set_title('Reward')
        axes[1, 1].axvline(x=self.config.transition_period, color='r', linestyle='--')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.config.save_path)
        plt.close()
    
    def find_equilibrium(self):
        """Find equilibrium using the underlying environment's method."""
        if hasattr(self.env, "find_equilibrium"):
            return self.env.find_equilibrium()
        raise NotImplementedError("No equilibrium solver available.")
    
    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """Calibrate the environment parameters."""
        if hasattr(self.env, "calibrate"):
            return self.env.calibrate(targets, method, **kwargs)
        raise NotImplementedError("Environment does not support calibration.")
    
    def estimate(self, data: np.ndarray, moment_function: callable, 
                weight_matrix: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Estimate environment parameters."""
        if hasattr(self.env, "estimate"):
            return self.env.estimate(data, moment_function, weight_matrix, **kwargs)
        raise NotImplementedError("Environment does not support estimation.")
    
    def render(self, mode="human"):
        """Render the environment."""
        if hasattr(self.env, "render"):
            return self.env.render(mode)
        return None
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.env, "close"):
            self.env.close()

    def set_final_params(self, final_params):
        """Set the final parameters for the transition."""
        self.final_params = final_params.copy()


# Example usage:
if __name__ == "__main__":
    from envs.aiyagari_env import AiyagariEnv
    
    # Create environment
    env = AiyagariEnv()
    
    # Configure transition analysis
    config = TransitionConfig(
        num_periods=200,
        num_agents=1000,
        burn_in=50,
        transition_period=100,
        compute_moments=True,
        save_path="transition_analysis.png"
    )
    
    # Wrap environment
    transition_env = TransitionWrapper(env, config)
    
    # Define parameter change (e.g., tax change)
    parameter_change = {
        'tau': 0.2  # New tax rate
    }
    
    # Simulate transition
    analysis = transition_env.simulate_transition(parameter_change)
    
    # Print results
    print("Initial Steady State:")
    print(analysis['initial_steady_state'])
    print("\nFinal Steady State:")
    print(analysis['final_steady_state']) 