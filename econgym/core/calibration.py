"""
calibration.py

Object-oriented calibration routine for macroeconomic models with improvements:
- Uses logging for progress messages.
- Adds input validation and error handling.
- Incorporates JAX JIT compilation.
- Improves multi-start initial guess generation.
- Comments and modular structure for flexibility.
"""

import numpy as np
import logging
import warnings
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Attempt to import Numba
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class CalibrationObjective:
    """
    Encapsulates the calibration objective function and related logic.
    Calibration minimizes the weighted squared distance between simulated and target data.
    """

    def __init__(self, env, data, target_function, weights=None, backend="numpy"):
        """
        Parameters
        ----------
        env : object
            Custom environment or model instance.
        data : ndarray or any custom data structure
            Empirical data or calibration targets.
        target_function : callable
            Function mapping (env, params) -> simulated targets.
        weights : ndarray or None
            Weight matrix for the calibration objective.
        backend : str
            One of {'numpy', 'jax', 'numba'} specifying the backend.
        """
        self.env = env
        self.data = data
        self.target_function = target_function
        self.backend = backend.lower()

        # Validate and set weights
        if weights is None:
            if isinstance(data, np.ndarray):
                # If data is 1D, treat length as dim; if 2D, use the second dim.
                dim = data.shape[0] if data.ndim == 1 else data.shape[1]
                self.weights = np.eye(dim)
            else:
                self.weights = np.eye(1)
        else:
            self.weights = weights

        self._prepare_objective()

    def _prepare_objective(self):
        """
        Set up the objective function based on the selected backend.
        """
        W = self.weights
        data = self.data
        env = self.env
        target_func = self.target_function

        def objective_numpy(params):
            sim = target_func(env, params)
            # Validate shapes if possible.
            if sim.shape != data.shape:
                raise ValueError(f"Shape mismatch: simulated {sim.shape} vs data {data.shape}")
            diff = sim - data
            return diff @ W @ diff

        self.obj_numpy = objective_numpy

        # Compile with Numba if requested and available.
        if NUMBA_AVAILABLE and self.backend == "numba":
            try:
                self.obj_numba = njit(objective_numpy)
            except Exception as e:
                warnings.warn(f"Numba njit failed, falling back to numpy: {e}")
                self.obj_numba = objective_numpy
        else:
            self.obj_numba = objective_numpy

        # JAX backend: convert weights and data to jax arrays and use jit.
        if JAX_AVAILABLE and self.backend == "jax":
            self.W_jax = jnp.array(W)
            self.data_jax = jnp.array(data)

            def objective_jax(params):
                sim = target_func(env, params)  # Ensure target_function is JAX-compatible.
                diff = sim - self.data_jax
                return diff @ self.W_jax @ diff

            self.obj_jax = jit(objective_jax)
            self.obj_jax_grad = jit(grad(objective_jax))
        else:
            self.obj_jax = None
            self.obj_jax_grad = None

    def __call__(self, params):
        """
        Evaluate the objective function at the given parameters.
        """
        if self.backend == "numpy":
            return self.obj_numpy(params)
        elif self.backend == "numba":
            return self.obj_numba(params)
        elif self.backend == "jax":
            if not JAX_AVAILABLE:
                raise RuntimeError("JAX not available. Install jax to use the 'jax' backend.")
            return float(self.obj_jax(jnp.array(params)))
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def gradient(self, params):
        """
        Return the gradient of the objective function if available.
        For the JAX backend, compute the gradient; otherwise, return None.
        """
        if self.backend == "jax" and JAX_AVAILABLE:
            grad_val = self.obj_jax_grad(jnp.array(params))
            return np.array(grad_val)
        else:
            # Optionally, implement finite-difference gradient here.
            return None


class Calibrator:
    """
    Organizes the calibration procedure:
    - Sets up the calibration objective.
    - Selects an optimization approach (including multi-starts and parallelization).
    """

    def __init__(self, env, data, target_function, weights=None, backend="numpy",
                 optimizer="BFGS", n_jobs=1):
        """
        Parameters
        ----------
        env : object
            Model or environment to calibrate.
        data : ndarray or custom object
            Empirical or target data.
        target_function : callable
            Maps (env, params) to simulated targets.
        weights : ndarray or None
            Weight matrix.
        backend : str
            One of {'numpy', 'numba', 'jax'}.
        optimizer : str
            SciPy optimizer method (e.g., 'BFGS', 'Nelder-Mead').
        n_jobs : int
            Number of parallel jobs for multi-start calibration.
        """
        self.env = env
        self.data = data
        self.target_function = target_function
        self.weights = weights
        self.backend = backend
        self.optimizer = optimizer
        self.n_jobs = n_jobs

        # Ensure the environment has an attribute 'num_params'
        if not hasattr(env, 'num_params'):
            raise AttributeError("Environment must have a 'num_params' attribute.")
        self.objective = CalibrationObjective(env, data, target_function, weights, backend)

    def run_calibration(self, initial_params=None, multiple_starts=1,
                        bounds=None, options=None, verbose=True):
        """
        Perform calibration using the chosen optimizer and backend.
        
        Parameters
        ----------
        initial_params : ndarray or None
            Initial guess for parameters.
        multiple_starts : int
            Number of random starting points for global search.
        bounds : list of tuple or None
            Bounds for parameters.
        options : dict or None
            Additional options for the optimizer.
        verbose : bool
            If True, log progress messages.

        Returns
        -------
        dict
            Dictionary with calibrated parameters, objective value, success flag, etc.
        """
        if initial_params is None:
            # Dummy initial guess (user should override with meaningful values)
            initial_params = np.zeros(self.env.num_params)

        if options is None:
            options = {}

        # Generate initial guesses for multi-start calibration.
        param_guesses = [self._draw_initial_guess() for _ in range(multiple_starts)]
        # Ensure the user's initial_params is included.
        param_guesses[0] = initial_params

        # Single start
        if multiple_starts <= 1:
            result = self._optimize(initial_params, bounds, options)
            return self._wrap_result(result, verbose)
        else:
            results = []
            if self.n_jobs > 1:
                logger.info(f"Running calibration with {multiple_starts} starts in parallel...")
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = {
                        executor.submit(self._optimize, guess, bounds, options): guess
                        for guess in param_guesses
                    }
                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            results.append(res)
                        except Exception as e:
                            logger.error(f"Optimization failed: {e}")
            else:
                logger.info(f"Running calibration with {multiple_starts} starts serially...")
                for guess in param_guesses:
                    results.append(self._optimize(guess, bounds, options))

            best_res = min(results, key=lambda r: r.fun)
            return self._wrap_result(best_res, verbose)

    def _optimize(self, init, bounds, options):
        """
        Helper function to run optimization for a given initial guess.
        """
        # For the JAX backend, pass gradient information if available.
        def fun_wrapped(p):
            return self.objective(p)

        if self.backend == "jax" and JAX_AVAILABLE:
            def grad_wrapped(p):
                return self.objective.gradient(p)
            res = minimize(fun_wrapped, np.array(init), jac=grad_wrapped,
                           method=self.optimizer, bounds=bounds, options=options)
        else:
            res = minimize(fun_wrapped, np.array(init),
                           method=self.optimizer, bounds=bounds, options=options)
        return res

    def _wrap_result(self, result, verbose):
        """
        Package the optimization result.
        """
        if verbose:
            logger.info(f"Calibration complete. Success: {result.success}, "
                        f"Objective: {result.fun}, Params: {result.x}")
        out = {
            "params": result.x,
            "objective_value": result.fun,
            "success": result.success,
            "message": result.message,
            "standard_errors": np.full_like(result.x, np.nan)  # Placeholder for standard errors.
        }
        return out

    def _draw_initial_guess(self):
        """
        Generate a random initial guess.
        Modify this method to incorporate domain-specific distributions or bounds.
        """
        return np.random.randn(self.env.num_params)