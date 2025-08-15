import numpy as np
from typing import Dict, Any, Tuple, List, Type
from econgym.core.base_solver import BaseSolver

class TransferLearningSolver(BaseSolver):
    """
    Solver for transfer learning between different economic models.
    Can work with any solver that inherits from BaseSolver.
    """
    def __init__(self, env, source_solver: BaseSolver, target_solver_class: Type[BaseSolver]):
        """
        Initialize the transfer learning solver.
        
        Args:
            env: The target environment
            source_solver: The source model solver
            target_solver_class: The class of the target solver to use
        """
        super().__init__(env)
        self.source_solver = source_solver
        self.target_solver = target_solver_class(env)
        
    def solve_equilibrium(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the equilibrium using transfer learning.
        Uses the source model's solution to initialize the target model.
        """
        # Get source model's solution
        source_policy, source_value = self.source_solver.solve_equilibrium(tol, max_iter)
        
        # Initialize target model with source model's solution
        self.target_solver.equilibrium_policy = source_policy
        self.target_solver.equilibrium_value = source_value
        
        # Fine-tune the target model
        target_policy, target_value = self.target_solver.solve_equilibrium(tol, max_iter)
        
        self.equilibrium_value = target_value
        self.equilibrium_policy = target_policy
        self.equilibrium_state = np.array([target_value, target_policy], dtype=object)
        return target_policy, target_value

    def _update_parameters(self, params: np.ndarray) -> None:
        """Update environment parameters."""
        self.target_solver._update_parameters(params)

    def _get_parameter_bounds(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for optimization."""
        return self.target_solver._get_parameter_bounds()

    def _get_parameters_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return self.target_solver._get_parameters_dict(params)

    def _compute_moments(self, states: List[np.ndarray], rewards: List[np.ndarray]) -> Dict[str, float]:
        """Compute moments from simulated data."""
        # Get moments from both source and target models
        source_moments = self.source_solver._compute_moments(states, rewards)
        target_moments = self.target_solver._compute_moments(states, rewards)
        
        # Combine moments
        moments = {}
        moments.update({f'source_{k}': v for k, v in source_moments.items()})
        moments.update({f'target_{k}': v for k, v in target_moments.items()})
        
        return moments

    def _interpolate_value(self, state: np.ndarray) -> np.ndarray:
        """Interpolate value function for given state."""
        return self.target_solver._interpolate_value(state)

    def _interpolate_policy(self, state: np.ndarray) -> np.ndarray:
        """Interpolate policy for given state."""
        return self.target_solver._interpolate_policy(state)
        
    def transfer_parameters(self, param_mapping: Dict[str, str]) -> None:
        """
        Transfer parameters from source to target model using a mapping.
        
        Args:
            param_mapping: Dictionary mapping source parameter names to target parameter names
        """
        source_params = self.source_solver._get_parameters_dict(
            np.array([self.source_solver.env.parameters[p] for p in param_mapping.keys()])
        )
        
        # Update target parameters
        for source_param, target_param in param_mapping.items():
            if target_param in self.env.parameters:
                self.env.parameters[target_param] = source_params[source_param]
                
    def compute_transfer_metrics(self, states: List[np.ndarray], rewards: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute metrics to evaluate the transfer learning performance.
        
        Args:
            states: List of states from simulation
            rewards: List of rewards from simulation
            
        Returns:
            Dictionary of transfer learning metrics
        """
        # Get moments from both models
        source_moments = self.source_solver._compute_moments(states, rewards)
        target_moments = self.target_solver._compute_moments(states, rewards)
        
        # Compute transfer metrics
        metrics = {}
        for key in source_moments:
            if key in target_moments:
                # Compute relative difference
                metrics[f'{key}_transfer_error'] = abs(
                    (target_moments[key] - source_moments[key]) / source_moments[key]
                )
        
        # Add average transfer error
        metrics['avg_transfer_error'] = np.mean(list(metrics.values()))
        
        return metrics 