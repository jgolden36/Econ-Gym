import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple, List
import networkx as nx
import matplotlib.pyplot as plt

class BlanchardKremer(EconEnv, NumpyRNGMixin):
    """
    Blanchard-Kremer Production Network Environment.
    
    This environment models a production network economy where:
    - Firms are connected in a directed network
    - Each firm requires inputs from other firms
    - Shocks can propagate through the network
    - Firms can experience disruptions
    
    The state space consists of:
    - Production capacity state for each firm (0 = disrupted, 1 = operational)
    
    The action space consists of:
    - Binary array indicating which firms to disrupt (0 = disrupt, 1 = keep operational)
    """
    def __init__(self, 
                 num_firms: int = 100,
                 connection_prob: float = 0.1,
                 shock_size: float = 0.5,
                 recovery_rate: float = 0.1):
        """
        Initialize the Blanchard-Kremer environment.
        
        Args:
            num_firms: Number of firms in the network
            connection_prob: Probability of connection between firms
            shock_size: Size of productivity shocks
            recovery_rate: Rate at which firms recover from disruptions
        """
        super().__init__()
        
        # Initialize parameters
        self.parameters = {
            'num_firms': num_firms,
            'connection_prob': connection_prob,
            'shock_size': shock_size,
            'recovery_rate': recovery_rate,
            'max_steps': 100
        }
        
        # Generate network
        self.network = self._generate_network()
        
        # Define observation space: binary state for each firm
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_firms,),
            dtype=np.float32
        )
        
        # Define action space: binary action for each firm
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_firms,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.steps = 0

    def _generate_network(self) -> nx.DiGraph:
        """
        Generate a random directed network of firms.
        
        Returns:
            Directed graph representing the production network
        """
        # Create a directed graph
        G = nx.gnp_random_graph(self.parameters['num_firms'], 
                               self.parameters['connection_prob'], 
                               directed=True)
        
        # Add input coefficients to edges
        for (u, v) in G.edges():
            G[u][v]['input_coefficient'] = np.random.uniform(0.1, 1.0)
            
        return G

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        self.reseed(seed)
            
        # Initialize all firms as operational
        self.state = np.ones(self.parameters['num_firms'], dtype=np.float32)
        self.steps = 0
        
        return self.state, {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one period of the economy.
        
        Args:
            actions: Binary array indicating which firms to disrupt
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Clip actions to valid range
        disruptions = np.clip(actions, 0, 1)
        
        # Apply disruptions
        self.state *= disruptions
        
        # Propagate disruptions through the network
        new_states = self.state.copy()
        for firm in range(self.parameters['num_firms']):
            suppliers = list(self.network.predecessors(firm))
            if suppliers:
                # Get required inputs and actual inputs
                required_inputs = np.array([self.network[supplier][firm]['input_coefficient'] 
                                         for supplier in suppliers])
                inputs_received = np.array([self.state[supplier] for supplier in suppliers])
                
                # Leontief production: constrained by minimum input
                input_ratios = inputs_received / required_inputs
                new_states[firm] = min(new_states[firm], input_ratios.min())
        
        # Update state
        self.state = new_states
        
        # Compute reward (total production)
        reward = np.sum(self.state)
        
        # Check termination conditions
        self.steps += 1
        done = self.check_equilibrium()
        truncated = self.steps >= self.parameters['max_steps']
        
        # Add additional information
        info = {
            'total_production': reward,
            'num_disrupted': np.sum(self.state < 0.1),
            'network_density': nx.density(self.network),
            'average_degree': np.mean([d for n, d in self.network.degree()])
        }
        
        return self.state, reward, done, truncated, info

    def check_equilibrium(self, threshold: float = 1e-3) -> bool:
        """
        Check if the economy has reached equilibrium.
        
        Args:
            threshold: Threshold for considering state changes
            
        Returns:
            True if equilibrium reached, False otherwise
        """
        return np.all(self.state < threshold) or np.all(self.state > 1 - threshold)

    def plot_network(self, save_path: Optional[str] = None) -> None:
        """
        Plot the production network.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(self.network)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.network, pos, 
                             node_color=self.state,
                             cmap=plt.cm.RdYlGn,
                             node_size=100)
        
        # Draw edges
        nx.draw_networkx_edges(self.network, pos, 
                             edge_color='gray',
                             alpha=0.5)
        
        plt.title('Production Network')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            print("\nCurrent State:")
            print(f"  Total Production: {np.sum(self.state):.2f}")
            print(f"  Number of Disrupted Firms: {np.sum(self.state < 0.1)}")
            print(f"  Network Density: {nx.density(self.network):.2f}")
            print(f"  Average Degree: {np.mean([d for n, d in self.network.degree()]):.2f}")
            
            print("\nParameters:")
            print(f"  Number of Firms: {self.parameters['num_firms']}")
            print(f"  Connection Probability: {self.parameters['connection_prob']:.2f}")
            print(f"  Shock Size: {self.parameters['shock_size']:.2f}")
            print(f"  Recovery Rate: {self.parameters['recovery_rate']:.2f}")

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
        from scipy.optimize import minimize
        
        def objective(params):
            # Update parameters
            self.parameters['connection_prob'] = params[0]
            self.parameters['shock_size'] = params[1]
            self.parameters['recovery_rate'] = params[2]
            
            # Regenerate network with new connection probability
            self.network = self._generate_network()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute moments
            mean_production = np.mean(rewards)
            mean_disrupted = np.mean([np.sum(s < 0.1) for s in states])
            network_density = nx.density(self.network)
            
            # Compute squared differences from targets
            error = 0
            if 'mean_production' in targets:
                error += (mean_production - targets['mean_production'])**2
            if 'mean_disrupted' in targets:
                error += (mean_disrupted - targets['mean_disrupted'])**2
            if 'network_density' in targets:
                error += (network_density - targets['network_density'])**2
                
            return error
        
        # Initial guess and bounds
        x0 = [
            self.parameters['connection_prob'],
            self.parameters['shock_size'],
            self.parameters['recovery_rate']
        ]
        bounds = [
            (0.01, 0.5),    # connection_prob
            (0.1, 1.0),     # shock_size
            (0.01, 0.5)     # recovery_rate
        ]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['connection_prob'] = result.x[0]
        self.parameters['shock_size'] = result.x[1]
        self.parameters['recovery_rate'] = result.x[2]
        
        # Regenerate network with final parameters
        self.network = self._generate_network()
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'connection_prob': result.x[0],
                'shock_size': result.x[1],
                'recovery_rate': result.x[2]
            },
            'objective_value': result.fun
        }

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
        from scipy.optimize import minimize
        
        def gmm_objective(params):
            # Update parameters
            self.parameters['connection_prob'] = params[0]
            self.parameters['shock_size'] = params[1]
            self.parameters['recovery_rate'] = params[2]
            
            # Regenerate network with new connection probability
            self.network = self._generate_network()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute simulated moments
            sim_moments = moment_function(self, states, rewards)
            
            # Compute GMM objective
            diff = sim_moments - data
            return diff @ weight_matrix @ diff
        
        # Initial guess and bounds
        x0 = [
            self.parameters['connection_prob'],
            self.parameters['shock_size'],
            self.parameters['recovery_rate']
        ]
        bounds = [
            (0.01, 0.5),    # connection_prob
            (0.1, 1.0),     # shock_size
            (0.01, 0.5)     # recovery_rate
        ]
        
        # Optimize
        result = minimize(gmm_objective, x0, method="BFGS", bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['connection_prob'] = result.x[0]
        self.parameters['shock_size'] = result.x[1]
        self.parameters['recovery_rate'] = result.x[2]
        
        # Regenerate network with final parameters
        self.network = self._generate_network()
        
        return {
            'success': result.success,
            'message': result.message,
            'estimated_parameters': {
                'connection_prob': result.x[0],
                'shock_size': result.x[1],
                'recovery_rate': result.x[2]
            },
            'objective_value': result.fun
        }

    def simulate(self, n_periods: int = 1000) -> Tuple[List[np.ndarray], List[float]]:
        """
        Simulate the model for a given number of periods.
        
        Args:
            n_periods: Number of periods to simulate
            
        Returns:
            Tuple of (states, rewards)
        """
        states = []
        rewards = []
        
        # Reset environment
        state, _ = self.reset()
        states.append(state)
        
        # Simulate forward
        for _ in range(n_periods):
            # Sample random actions
            action = self.action_space.sample()
            
            # Step environment
            state, reward, done, truncated, _ = self.step(action)
            
            # Store results
            states.append(state)
            rewards.append(reward)
            
            # Check termination
            if done or truncated:
                break
        
        return states, rewards

    def compute_moments(self, states: List[np.ndarray], rewards: List[float]) -> np.ndarray:
        """
        Compute key moments from simulation data.
        
        Args:
            states: List of states from simulation
            rewards: List of rewards from simulation
            
        Returns:
            Array of moments
        """
        # Convert to numpy arrays
        states = np.array(states)
        rewards = np.array(rewards)
        
        # Compute moments
        moments = np.array([
            np.mean(rewards),                    # Mean production
            np.std(rewards),                     # Production volatility
            np.mean([np.sum(s < 0.1) for s in states]),  # Mean number of disrupted firms
            nx.density(self.network),            # Network density
            np.mean([d for n, d in self.network.degree()])  # Average degree
        ])
        
        return moments

# Example usage:
if __name__ == "__main__":
    env = BlanchardKremer(num_firms=50)
    state = env.reset()
    print("Initial state:", state)
    
    # Take a sample action
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Info:", info)
    
    # Plot the network
    env.plot_network()
    
    # Example calibration
    targets = {
        'mean_production': 40.0,
        'mean_disrupted': 5.0,
        'network_density': 0.1
    }
    calibration_results = env.calibrate(targets)
    print("\nCalibration Results:", calibration_results)
    
    # Example estimation
    data = np.array([40.0, 5.0, 0.1, 0.1, 5.0])  # Example empirical moments
    weight_matrix = np.eye(5)  # Identity weight matrix
    estimation_results = env.estimate(data, env.compute_moments, weight_matrix)
    print("\nEstimation Results:", estimation_results)