import functools
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

# Optional accelerators
try:  # numba is optional
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover - optional
    njit = None  # type: ignore

try:  # jax is optional
    import jax  # type: ignore
except Exception:  # pragma: no cover - optional
    jax = None  # type: ignore

try:
    from scipy.optimize import minimize  # type: ignore
except Exception as _exc:  # pragma: no cover - ensure import error is clear at call time
    minimize = None  # type: ignore


def _objective(params: np.ndarray, likelihood_function: Callable[[np.ndarray, Any], float], data: Any) -> float:
    """Negative log-likelihood for minimization."""
    return -float(likelihood_function(params, data))


class MLestimation:
    """
    Maximum likelihood estimation helper for Gymnasium-compatible environments.

    - Uses env.maximum_likelihood_estimate if available
    - Otherwise minimizes the negative log-likelihood provided by likelihood_function
    - Can optionally accelerate with Numba or JAX
    """

    def __init__(
        self,
        env: Any,
        likelihood_function: Optional[Callable[[np.ndarray, Any], float]] = None,
        data: Any = None,
        acceleration: Optional[str] = None,
    ) -> None:
        self.env = env
        self.likelihood_function = likelihood_function or getattr(env, "log_likelihood", None)
        if self.likelihood_function is None:
            raise ValueError(
                "No likelihood function provided and the environment does not have a 'log_likelihood' method."
            )
        self.data = data
        self.acceleration = acceleration
        if self.acceleration == "numba" and njit is None:
            raise ImportError("Numba is not installed but acceleration='numba' was specified.")
        if self.acceleration == "jax" and jax is None:
            raise ImportError("JAX is not installed but acceleration='jax' was specified.")

    def collect_data(self, num_episodes: int = 100, max_steps: int = 100) -> list:
        """
        Collect trajectories using random actions.
        Stores a list of episodes, each with (obs, action, reward, next_obs, done).
        Works with both Gym and Gymnasium step/reset conventions.
        """
        data = []
        for _ in range(num_episodes):
            episode = []
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                step_result = self.env.step(action)
                if isinstance(step_result, tuple) and len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated) or bool(truncated)
                else:
                    # Fallback for legacy 4-tuple API
                    next_obs, reward, done, info = step_result  # type: ignore[misc]
                episode.append((obs, action, reward, next_obs, bool(done)))
                obs = next_obs
                if bool(done):
                    break
            data.append(episode)
        self.data = data
        return data

    def solve(self, initial_params: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, Any]:
        """
        Run MLE and return (estimated_params, optimization_result).
        """
        # Delegate to environment-specific routine if available
        if hasattr(self.env, "maximum_likelihood_estimate"):
            return self.env.maximum_likelihood_estimate(initial_params, data=self.data, **kwargs)

        if self.data is None:
            self.collect_data()

        if minimize is None:
            raise ImportError("scipy is required for generic MLE; install with the 'plot' extra or scipy")

        objective: Callable[[np.ndarray], float]
        _base = functools.partial(_objective, likelihood_function=self.likelihood_function, data=self.data)
        if self.acceleration == "numba":
            objective = njit(_base)  # type: ignore[arg-type]
        elif self.acceleration == "jax":
            jitted = jax.jit(_base)

            def objective(params: np.ndarray) -> float:
                return float(jitted(params))

        else:
            objective = _base

        res = minimize(lambda p: objective(p), initial_params, **kwargs)
        return np.asarray(res.x), res


