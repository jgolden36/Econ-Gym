import gymnasium as gym
import numpy as np
from gymnasium import spaces
from econgym.core.base_env import EconEnv
from econgym.core.utils import NumpyRNGMixin
from typing import Dict, Any, Optional, Tuple, List

# Optional heavy dependencies: import lazily/optionally
try:  # SB3 is optional
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
except Exception:  # pragma: no cover - optional
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore

try:  # Torch optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

try:  # Plotting optional
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore

try:  # SciPy optional for some routines
    from scipy.optimize import bisect  # type: ignore
    from scipy import sparse  # type: ignore
    from scipy.sparse import csr_matrix  # type: ignore
except Exception:  # pragma: no cover - optional
    bisect = None  # type: ignore
    sparse = None  # type: ignore
    csr_matrix = None  # type: ignore


class AiyagariEnv(EconEnv, NumpyRNGMixin):
    """
    A heterogeneous agent Aiyagari incomplete markets model environment.
    
    This environment models a heterogeneous agent economy where agents face uninsurable
    income risk and can save in a risk-free asset. The state space consists of the
    agent's current income shock and asset holdings. The model includes labor and capital
    taxes, and supports heterogeneous agents with different preferences and productivity.
    """
    def __init__(self, 
                 beta=0.96, 
                 risk_aversion=2.0, 
                 income_persistence=0.9,
                 labor_tax=0.2,
                 capital_tax=0.397,  # Initial capital tax rate
                 alpha=0.36,  # Capital share in production
                 delta=0.08,  # Depreciation rate
                 num_agent_types=2,  # Number of agent types
                 X=1.0,  # Productivity factor
                 income_grid_size=3,
                 asset_grid_size=800,
                 asset_max=100.0,
                 income_std=0.2):
        """
        Parameters:
            beta: Discount factor
            risk_aversion: Coefficient of relative risk aversion
            income_persistence: Persistence of income shocks
            labor_tax: Labor income tax rate
            capital_tax: Capital income tax rate
            alpha: Capital share in production
            delta: Depreciation rate
            num_agent_types: Number of heterogeneous agent types
            X: Productivity factor
        """
        super().__init__()
        # Initialize parameters
        self.parameters = {
            'beta': beta,
            'risk_aversion': risk_aversion,
            'income_persistence': income_persistence,
            'income_std': income_std,  # Standard deviation of income shocks
            'asset_min': 0.0,   # Minimum asset level (borrowing constraint)
            'asset_max': asset_max,  # Maximum asset level
            'interest_rate': 0.03,  # Initial interest rate (will be endogenized)
            'wage': 1.0,  # Wage rate (set from firm FOC in GE experiments)
            'income_grid_size': income_grid_size,  # Number of income grid points
            'asset_grid_size': asset_grid_size,  # Number of asset grid points
            'labor_tax': labor_tax,
            'capital_tax': capital_tax,
            'alpha': alpha,
            'delta': delta,
            'num_agent_types': num_agent_types,
            'X': X  # Productivity factor
        }
        
        # Create income process via Rouwenhorst to match persistence and volatility
        self.income_grid, self.income_transition = self._build_income_process(
            n_states=self.parameters['income_grid_size'],
            rho=self.parameters['income_persistence'],
            sigma_eps=self.parameters['income_std']
        )
        
        # Create asset grid with proper handling of asset_min
        if self.parameters['asset_min'] >= 0:
            # Exponential grid for non-negative assets
            self.asset_grid = np.exp(np.linspace(
                np.log(self.parameters['asset_min'] + 1e-6),
                np.log(self.parameters['asset_max']),
                self.parameters['asset_grid_size']
            )) - 1e-6
        else:
            # Linear grid when allowing borrowing
            self.asset_grid = np.linspace(
                self.parameters['asset_min'],
                self.parameters['asset_max'],
                self.parameters['asset_grid_size']
            )
        
        # Initialize agent types with different preferences
        self.agent_types = []
        for i in range(num_agent_types):
            # Different risk aversion and discount factors for each type
            agent_type = {
                'beta': beta * (1 + 0.05 * (i - (num_agent_types-1)/2)),  # Smaller variation in beta
                'risk_aversion': risk_aversion * (1 + 0.1 * (i - (num_agent_types-1)/2)),  # Smaller variation in risk aversion
                'productivity': 1.0 + 0.1 * i  # Smaller productivity differences
            }
            self.agent_types.append(agent_type)
        
        # Define observation space (income shock, asset holdings, agent type)
        self.observation_space = spaces.Box(
            low=np.array([self.income_grid[0], self.parameters['asset_min'], 0]),
            high=np.array([self.income_grid[-1], self.parameters['asset_max'], num_agent_types-1]),
            dtype=np.float32
        )
        
        # Define action space (savings rate between 0 and 1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        
        # Initialize solver
        self.solver = None
        # Vec env only if SB3 is available
        self.vec_env = DummyVecEnv([lambda: self]) if DummyVecEnv is not None else None
        
        # Initialize RNG and state
        self.rng = np.random.default_rng()
        self.reset()
        # Persist last converged policy to warm-start EGM across nearby parameters
        self.last_policy = None

    def _build_income_process(self, n_states: int, rho: float, sigma_eps: float):
        """
        Build a discrete approximation to AR(1) income using the Rouwenhorst method.
        y_t = rho * y_{t-1} + eps_t, eps ~ N(0, sigma_eps^2).
        Returns level grid exp(y) and transition matrix Pi.
        """
        p = (1 + rho) / 2
        q = p
        Pi = np.array([[1.0]])
        for m in range(2, n_states + 1):
            newPi = np.zeros((m, m))
            newPi[:m-1, :m-1] += p * Pi
            newPi[:m-1, 1:]  += (1 - p) * Pi
            newPi[1:, :m-1]  += (1 - q) * Pi
            newPi[1:, 1:]    += q * Pi
            newPi[1:-1, :]   /= 2.0
            Pi = newPi
        # Unconditional std of y is sigma_eps / sqrt(1 - rho^2)
        sigma_y = sigma_eps / max(np.sqrt(max(1e-12, 1 - rho**2)), 1e-8)
        y_max = sigma_y * np.sqrt(n_states - 1)
        y_grid = np.linspace(-y_max, y_max, n_states)
        income_grid = np.exp(y_grid)
        # Normalize mean income to 1 using stationary distribution of Pi
        try:
            eigvals, eigvecs = np.linalg.eig(Pi.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            stationary = np.real(eigvecs[:, idx])
            stationary = np.maximum(stationary, 0)
            if stationary.sum() == 0:
                stationary = np.ones_like(stationary)
            stationary = stationary / stationary.sum()
            mean_income = float(stationary @ income_grid)
            if np.isfinite(mean_income) and mean_income > 0:
                income_grid = income_grid / mean_income
        except Exception:
            # Fallback to equal-weights normalization
            mean_income = float(np.mean(income_grid))
            if np.isfinite(mean_income) and mean_income > 0:
                income_grid = income_grid / mean_income
        return income_grid, Pi

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        self.reseed(seed)
        # Initialize with random income shock, zero assets, and random agent type
        income = self.rng.choice(self.income_grid)
        assets = self.parameters['asset_min']
        agent_type = self.rng.integers(0, self.parameters['num_agent_types'])
        self.state = np.array([income, assets, agent_type], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        """
        Execute one time step in the environment dynamics.
        
        Args:
            action: Savings rate between 0 and 1
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Unpack state
        income, assets, agent_type = self.state
        agent_type = int(agent_type)  # Convert to integer for indexing
        
        # Get agent-specific parameters
        agent_params = self.agent_types[agent_type]
        
        # Calculate labor income (with productivity adjustment and wage)
        wage = float(self.parameters.get('wage', 1.0))
        # Income state is already in levels; do not exponentiate again
        income_level = float(income) * agent_params['productivity']
        labor_income = wage * income_level
        
        # Effective net return on assets (interest_rate is already net from equilibrium search)
        interest_rate = float(self.parameters['interest_rate'])
        net_return = interest_rate
        
        # After-tax labor income
        after_tax_labor = (1.0 - float(self.parameters['labor_tax'])) * labor_income
        
        # Total resources: principal plus after-tax return on assets, plus after-tax labor income
        total_resources = (1.0 + net_return) * assets + after_tax_labor
        
        # Calculate savings and consumption
        savings = action[0] * total_resources
        consumption = max(total_resources - savings, 1e-10)
        
        # Calculate utility using agent-specific risk aversion
        if agent_params['risk_aversion'] == 1:
            utility = np.log(consumption)
        else:
            utility = (consumption ** (1 - agent_params['risk_aversion'])) / (1 - agent_params['risk_aversion'])
        
        # Update income shock (AR(1) process)
        new_income = (self.parameters['income_persistence'] * income +
                      self.rng.normal(0, self.parameters['income_std']))
        new_income = np.clip(new_income, self.income_grid[0], self.income_grid[-1])
        
        # Update assets
        new_assets = savings
        
        # Ensure assets stay within bounds
        new_assets = np.clip(new_assets, self.parameters['asset_min'], self.parameters['asset_max'])
        
        # Update state
        self.state = np.array([new_income, new_assets, agent_type], dtype=np.float32)
        
        done = False
        truncated = False
        return self.state, utility, done, truncated, {}

    def find_equilibrium_interest_rate(self, tol=1e-6):
        """
        Find the equilibrium interest rate that clears the capital market.
        
        Args:
            tol: Tolerance for market clearing
            
        Returns:
            Equilibrium interest rate
        """
        cache: dict[float, float] = {}

        def excess_demand(r):
            # Deterministic cache to avoid re-solving at identical r
            key = float(np.round(r, 10))
            if key in cache:
                return cache[key]
            # r is the net household interest rate: r_net = (1 - tau_k) * r_pre_tax
            
            try:
                # Solve for stationary distribution
                # Set wage from firm FOCs given implied capital demand at this r.
                # IMPORTANT: Firms set r_pre_tax. Households see r_net = (1 - tau_k) * r_pre_tax.
                tau_k = float(self.parameters.get('capital_tax', 0.0))
                one_minus_tau = max(1e-8, 1.0 - tau_k)
                r_pre_tax = r / one_minus_tau
                # Store net rate so household net return inside EGM is consistent
                self.parameters['interest_rate'] = r
                L_supply = 1.0
                K_demand = ((self.parameters['alpha'] * self.parameters['X'])/(r_pre_tax + self.parameters['delta']))**(1/(1-self.parameters['alpha'])) * L_supply
                self.parameters['wage'] = (1.0 - self.parameters['alpha']) * self.parameters['X'] * (K_demand / L_supply)**(self.parameters['alpha'])

                _, _, stationary_dist = self.solve_egm()
                
                # Compute per-capita aggregate capital supply (normalize by total mass)
                asset_grid = self.asset_grid
                total_mass = np.sum(stationary_dist)
                if not np.isfinite(total_mass) or total_mass <= 0:
                    return np.inf if r < 0.05 else -np.inf
                K_supply = float(np.sum(stationary_dist * asset_grid[np.newaxis, np.newaxis, :]) / total_mass)
                
                # Compute aggregate capital demand using pre-tax rental rate
                L_supply = 1.0  # Normalized labor supply
                K_demand = ((self.parameters['alpha'] * self.parameters['X'])/(r_pre_tax + self.parameters['delta']))**(1/(1-self.parameters['alpha'])) * L_supply
                
                # Compute excess demand
                excess = K_supply - K_demand
                
                # Add numerical stability checks
                if not np.isfinite(excess) or K_supply < 0:
                    return np.inf if r < 0.05 else -np.inf
                
                print(f"\nAt r = {r:.6f}:")
                r_net = (1.0 - tau_k) * r_pre_tax
                print(f"K_supply = {K_supply:.6f}")
                print(f"K_demand = {K_demand:.6f}")
                print(f"wage = {self.parameters['wage']:.6f}, r_pre_tax = {r_pre_tax:.6f}, r_net = {r_net:.6f}")
                print(f"Excess = {excess:.6f}")
                
                cache[key] = float(excess)
                return cache[key]
                
            except (RuntimeWarning, RuntimeError, ValueError) as e:
                print(f"Warning: Numerical error at r = {r}: {str(e)}")
                val = np.inf if r < 0.05 else -np.inf
                cache[key] = float(val)
                return cache[key]
        
        # Try to find brackets for the root with improved and wider bounds
        r_min, r_max = 0.005, 0.25
        r_step = 0.005
        r = r_min
        
        # Track best solution found
        best_r = 0.03  # Default value
        best_excess = np.inf
        
        print("\nSearching for equilibrium interest rate...")
        while r <= r_max:
            try:
                y1 = excess_demand(r)
                y2 = excess_demand(min(r + r_step, r_max))
                
                # Update best solution if we find a better one
                if abs(y1) < abs(best_excess):
                    best_r = r
                    best_excess = y1
                
                if y1 * y2 < 0:  # Found a bracket
                    # Use bisection to find root
                    r_eq = bisect(excess_demand, r, r + r_step, rtol=tol)
                    
                    # Verify the solution
                    excess = excess_demand(r_eq)
                    if abs(excess) < abs(best_excess):
                        best_r = r_eq
                        best_excess = excess
                    
                    print(f"\nFound equilibrium interest rate: {best_r:.4f}")
                    print(f"Excess demand at equilibrium: {best_excess:.4f}")
                    return best_r
            except Exception as e:
                print(f"Warning: Error during root finding at r = {r}: {str(e)}")
            r += r_step
        
        # If no bracket found, try expanding the search range once
        if not np.isfinite(best_excess) or abs(best_excess) > tol:
            r2_min, r2_max = 0.001, 0.5
            r2_step = 0.01
            r = r2_min
            while r <= r2_max:
                try:
                    y1 = excess_demand(r)
                    y2 = excess_demand(min(r + r2_step, r2_max))
                    if abs(y1) < abs(best_excess):
                        best_r, best_excess = r, y1
                    if y1 * y2 < 0:
                        r_eq = bisect(excess_demand, r, min(r + r2_step, r2_max), rtol=tol)
                        excess = excess_demand(r_eq)
                        if abs(excess) < abs(best_excess):
                            best_r, best_excess = r_eq, excess
                        print(f"\nFound equilibrium interest rate: {best_r:.4f}")
                        print(f"Excess demand at equilibrium: {best_excess:.4f}")
                        return best_r
                except Exception:
                    pass
                r += r2_step
        
        # If we can't find a proper bracket, return the best solution found
        print(f"\nWarning: Could not find equilibrium interest rate. Using best found value: {best_r:.4f}")
        print(f"Excess demand at best value: {best_excess:.4f}")
        return best_r

    def find_equilibrium(self):
        """
        Find the stationary equilibrium using PPO from stable-baselines3.
        Returns the optimal policy and value function.
        """
        if self.solver is None:
            self.solver = PPO("MlpPolicy", self.vec_env, verbose=0)
        
        # Train the model
        self.solver.learn(total_timesteps=100000)
        
        # Get the policy and value function
        policy = self.solver.policy
        value_fn = self.solver.policy.value_net
        
        return policy, value_fn

    def get_policy(self, state):
        """
        Get the optimal action for a given state using the trained policy.
        """
        if self.solver is None:
            self.find_equilibrium()
        return self.solver.predict(state)[0]

    def get_value_function(self, state):
        """
        Get the value function for a given state using the trained model.
        """
        if self.solver is None:
            self.find_equilibrium()
        return self.solver.policy.value_net(torch.FloatTensor(state))

    def calibrate(self, targets: Dict[str, Any], method: str = "BFGS", **kwargs) -> Dict[str, Any]:
        """
        Calibrate the model parameters to match target moments.
        """
        from scipy.optimize import minimize
        
        def objective(params):
            # Update parameters
            self.parameters['risk_aversion'] = params[0]
            self.parameters['income_persistence'] = params[1]
            self.parameters['income_std'] = params[2]
            self.parameters['labor_tax'] = params[3]
            self.parameters['capital_tax'] = params[4]
            
            # Reset solver when parameters change
            self.solver = None
            
            # Find equilibrium
            policy, _ = self.find_equilibrium()
            
            # Simulate the model
            states, rewards = self.simulate(n_periods=1000)
            
            # Compute moments
            avg_consumption = np.mean([np.exp(state[0]) for state in states])
            consumption_volatility = np.std([np.exp(state[0]) for state in states])
            avg_assets = np.mean([state[1] for state in states])
            gini_coefficient = self._compute_gini_coefficient(states)
            
            # Compute squared differences from targets
            error = 0
            if 'avg_consumption' in targets:
                error += (avg_consumption - targets['avg_consumption'])**2
            if 'consumption_volatility' in targets:
                error += (consumption_volatility - targets['consumption_volatility'])**2
            if 'avg_assets' in targets:
                error += (avg_assets - targets['avg_assets'])**2
            if 'gini_coefficient' in targets:
                error += (gini_coefficient - targets['gini_coefficient'])**2
                
            return error
        
        # Get initial parameters and bounds
        x0 = [
            self.parameters['risk_aversion'],
            self.parameters['income_persistence'],
            self.parameters['income_std'],
            self.parameters['labor_tax'],
            self.parameters['capital_tax']
        ]
        bounds = [
            (1.0, 5.0),    # risk_aversion
            (0.7, 0.99),   # income_persistence
            (0.1, 0.5),    # income_std
            (0.0, 0.5),    # labor_tax
            (0.0, 0.5)     # capital_tax
        ]
        
        # Optimize
        result = minimize(objective, x0, method=method, bounds=bounds, **kwargs)
        
        # Update parameters with optimal values
        self.parameters['risk_aversion'] = result.x[0]
        self.parameters['income_persistence'] = result.x[1]
        self.parameters['income_std'] = result.x[2]
        self.parameters['labor_tax'] = result.x[3]
        self.parameters['capital_tax'] = result.x[4]
        
        return {
            'success': result.success,
            'message': result.message,
            'optimal_parameters': {
                'risk_aversion': result.x[0],
                'income_persistence': result.x[1],
                'income_std': result.x[2],
                'labor_tax': result.x[3],
                'capital_tax': result.x[4]
            },
            'objective_value': result.fun
        }

    def _compute_gini_coefficient(self, states):
        """Compute the Gini coefficient for asset holdings."""
        assets = [state[1] for state in states]
        assets = np.sort(assets)
        n = len(assets)
        index = np.arange(1, n + 1)
        return ((2 * np.sum(index * assets)) / (n * np.sum(assets))) - (n + 1) / n

    def weights_and_indices(self, A, gridA, na, ny):
        """
        Compute weights and indices for interpolation, similar to the Julia implementation.
        
        Args:
            A: Array of asset values
            gridA: Asset grid
            na: Number of asset grid points
            ny: Number of income grid points
            
        Returns:
            Tuple of (weights, indices)
        """
        # Initialize arrays
        weights = np.zeros((na, ny, 2))
        indices = np.zeros((na, ny), dtype=int)
        
        # For each asset value and income state
        for i in range(na):
            for y in range(ny):
                # Find the index of the last grid point less than or equal to A[i,y]
                idx = np.searchsorted(gridA, A[i,y], side='right') - 1
                
                # Handle boundary cases
                if idx < 0:
                    idx = 0
                    weights[i,y,0] = 1.0
                    weights[i,y,1] = 0.0
                elif idx >= len(gridA) - 1:
                    idx = len(gridA) - 2
                    weights[i,y,0] = 0.0
                    weights[i,y,1] = 1.0
                else:
                    # Compute weights for linear interpolation
                    weights[i,y,0] = (gridA[idx+1] - A[i,y]) / (gridA[idx+1] - gridA[idx])
                    weights[i,y,1] = 1.0 - weights[i,y,0]
                
                indices[i,y] = idx
        
        return weights, indices

    def solve_egm(self, max_iter=500, tol=1e-4):
        """
        Solve the household problem using the Endogenous Grid Method (EGM).
        Returns value function, policy function (as savings rates), and stationary distribution.
        """
        try:
            print("Starting solve_egm...", flush=True)
            
            # Initialize arrays
            na = self.parameters['asset_grid_size']
            ny = self.parameters['income_grid_size']
            n_agent_types = self.parameters['num_agent_types']
            
            print(f"Grid sizes - na: {na}, ny: {ny}, n_agent_types: {n_agent_types}", flush=True)
            
            # Initialize value function and policy (policy stores next period asset level a')
            V = np.zeros((n_agent_types, ny, na))
            policy = np.zeros((n_agent_types, ny, na))
            
            # Create endogenous grid for assets
            a_grid = self.asset_grid
            
            # Small constant to avoid division by zero
            eps = 1e-10
            
            # Initial guess for value function based on steady state consumption
            for agent_type in range(n_agent_types):
                agent_params = self.agent_types[agent_type]
                sigma = agent_params['risk_aversion']
                for y_idx, y in enumerate(self.income_grid):
                    # Calculate income (with wage)
                    wage = float(self.parameters.get('wage', 1.0))
                    income = wage * (y * agent_params['productivity'])
                    # Get returns
                    interest_rate = float(self.parameters['interest_rate'])
                    net_return = (1.0 - float(self.parameters['capital_tax'])) * interest_rate
                    for a_idx, a in enumerate(a_grid):
                        c_guess = max((1.0 + net_return) * a + (1.0 - self.parameters['labor_tax']) * income, eps)  # Ensure positive consumption
                        if sigma == 1:
                            V[agent_type, y_idx, a_idx] = np.log(c_guess) / (1 - agent_params['beta'])
                        else:
                            V[agent_type, y_idx, a_idx] = (c_guess**(1 - sigma)) / (1 - sigma) / (1 - agent_params['beta'])
            
            print(f"Asset grid range: [{a_grid[0]:.2f}, {a_grid[-1]:.2f}]", flush=True)
            
            # Initialize policy_old: use warm-start if available, else feasible savings rule
            if isinstance(self.last_policy, np.ndarray) and self.last_policy.shape == policy.shape:
                policy_old = np.array(self.last_policy, copy=True)
            else:
                policy_old = np.zeros_like(policy)
                for agent_type in range(n_agent_types):
                    agent_params = self.agent_types[agent_type]
                    wage0 = float(self.parameters.get('wage', 1.0))
                    r0 = float(self.parameters['interest_rate'])
                    net_r0 = r0
                    s0 = 0.3
                    for y_idx, y in enumerate(self.income_grid):
                        labor_inc0 = wage0 * (y * agent_params['productivity'])
                        y_net0 = (1.0 - self.parameters['labor_tax']) * labor_inc0
                        total_res = (1.0 + net_r0) * a_grid + y_net0
                        # Initial feasible guess: a' = s0 * total_resources (no division by 1+r)
                        a_prime0 = np.clip(s0 * total_res, self.parameters['asset_min'], self.parameters['asset_max'])
                        policy_old[agent_type, y_idx] = a_prime0
 
            # Policy iteration via EGM
            relax = 0.9  # fixed relaxation to reduce path dependence
            prev_diff = np.inf
            small_change_streak = 0
            for iteration in range(max_iter):
                policy_new = np.zeros_like(policy_old)
                
                for agent_type in range(n_agent_types):
                    agent_params = self.agent_types[agent_type]
                    beta = agent_params['beta']
                    sigma = agent_params['risk_aversion']
                    wage = float(self.parameters.get('wage', 1.0))
                    interest_rate = float(self.parameters['interest_rate'])
                    net_return = interest_rate  # interest_rate is already the net rate from equilibrium search
                    
                    # Debug: check if β(1+r_net) < 1 (should be ≥ 1 for interior solution)
                    beta_times_growth = beta * (1.0 + net_return)
                    if iteration == 0:
                        print(f"Agent {agent_type}: β={beta:.4f}, r_net={net_return:.4f}, β(1+r_net)={beta_times_growth:.4f}", flush=True)

                    # Marginal utility and its inverse
                    if sigma == 1:
                        marg_u = lambda c: 1.0 / np.maximum(c, eps)
                        inv_marg_u = lambda mu: 1.0 / np.maximum(mu, eps)
                    else:
                        marg_u = lambda c: np.minimum(np.maximum(c, eps) ** (-sigma), 1e10)
                        inv_marg_u = lambda mu: np.maximum(mu, eps) ** (-1.0 / sigma)

                    for y_idx, y in enumerate(self.income_grid):
                        labor_inc = wage * (y * agent_params['productivity'])
                        after_tax_labor = (1.0 - self.parameters['labor_tax']) * labor_inc

                        # Choice grid for next assets
                        a_prime_grid = a_grid

                        # Compute next period consumption c'(y', a') using last policy: c' = (1+r_net)a' + y'_net - a''
                        emu = np.zeros_like(a_prime_grid)
                        for y_next in range(ny):
                            prob = float(self.income_transition[y_idx, y_next])
                            next_after_tax_labor = (1.0 - self.parameters['labor_tax']) * wage * (self.income_grid[y_next] * agent_params['productivity'])
                            # Interpolate a'' = g_old(y', a')
                            g_old = policy_old[agent_type, y_next]
                            a_pp = np.interp(a_prime_grid, a_grid, g_old)
                            c_next = np.maximum((1.0 + net_return) * a_prime_grid + next_after_tax_labor - a_pp, eps)
                            emu += prob * marg_u(c_next)
                        
                        # Debug: log key values at first and last asset point
                        if iteration < 3 and (y_idx == 0 or y_idx == ny-1):
                            print(f"  y[{y_idx}]={self.income_grid[y_idx]:.3f}: EMU[0]={emu[0]:.6f}, EMU[-1]={emu[-1]:.6f}", flush=True)

                        # Current consumption from Euler with bounds to prevent overflow
                        mu_c = beta * (1.0 + net_return) * emu
                        mu_c = np.clip(mu_c, 1e-8, 1e8)
                        c = inv_marg_u(mu_c)
                        c = np.clip(c, eps, 1e8)  # Prevent NaN/inf in consumption
                        
                        # Debug: consumption and endogenous grid
                        if iteration < 3 and (y_idx == 0 or y_idx == ny-1):
                            print(f"    c[0]={c[0]:.6f}, c[-1]={c[-1]:.6f}", flush=True)
                        # Endogenous asset grid with bounds
                        a_endo = (c + a_prime_grid - after_tax_labor) / (1.0 + net_return)
                        a_endo = np.clip(a_endo, -1e6, 1e6)  # Prevent extreme values
                        
                        if iteration < 3 and (y_idx == 0 or y_idx == ny-1):
                            print(f"    a_endo[0]={a_endo[0]:.6f}, a_endo[-1]={a_endo[-1]:.6f}", flush=True)

                        # Ensure bounds and mild smoothing to keep a_endo monotone
                        a_endo = np.maximum(a_endo, self.parameters['asset_min'])
                        # Smooth small kinks
                        a_endo = 0.5 * a_endo + 0.5 * np.maximum.accumulate(a_endo)
                        # Map to exogenous asset grid
                        # Handle potential non-monotonicities by sorting
                        order = np.argsort(a_endo)
                        a_endo_sorted = a_endo[order]
                        a_prime_sorted = a_prime_grid[order]
                        # Extrapolate at boundaries
                        a_prime_interp = np.interp(a_grid, a_endo_sorted, a_prime_sorted, left=a_prime_sorted[0], right=a_prime_sorted[-1])
                        # Enforce feasibility: a' in [a_min, (1+r_net)a + y_net - eps]
                        resources = (1.0 + net_return) * a_grid + after_tax_labor
                        a_prime_feasible = np.clip(a_prime_interp, self.parameters['asset_min'], resources - 1e-6)
                        # Enforce monotonicity in a
                        a_prime_monotone = np.maximum.accumulate(a_prime_feasible)
                        # Check for NaN/inf and fallback if needed
                        if not np.all(np.isfinite(a_prime_monotone)):
                            print(f"Warning: NaN/inf in policy for agent {agent_type}, income {y_idx}. Using fallback.", flush=True)
                            a_prime_monotone = np.maximum(a_grid * 0.1, self.parameters['asset_min'])
                        policy_new[agent_type, y_idx] = a_prime_monotone
                        
                        if iteration < 3 and (y_idx == 0 or y_idx == ny-1):
                            print(f"    final a'[0]={a_prime_monotone[0]:.6f}, a'[-1]={a_prime_monotone[-1]:.6f}", flush=True)

                # Convergence check on policy
                max_diff = np.max(np.abs(policy_new - policy_old))
                if iteration % 50 == 0 or iteration < 10:
                    print(f"Iteration {iteration} - Max policy change: {max_diff:.6f}", flush=True)
                if max_diff < tol:
                    small_change_streak += 1
                else:
                    small_change_streak = 0
                if small_change_streak >= 5 or (max_diff < 5e-4 and iteration > 600):
                    policy = policy_new
                    print(f"Converged at iteration {iteration} with policy change {max_diff:.6f}", flush=True)
                    break
                # Fixed relaxation to reduce oscillations and path dependence
                policy_old = relax * policy_old + (1.0 - relax) * policy_new
                prev_diff = max_diff

            # If loop exited without triggering the convergence break, use latest smoothed policy
            if not np.any(policy):
                policy = policy_old

            # Policy evaluation to get V and consumption for welfare
            V = np.zeros_like(V)
            for eval_iter in range(200):
                V_new = np.zeros_like(V)
                for agent_type in range(n_agent_types):
                    agent_params = self.agent_types[agent_type]
                    beta = agent_params['beta']
                    sigma = agent_params['risk_aversion']
                    wage = float(self.parameters.get('wage', 1.0))
                    interest_rate = float(self.parameters['interest_rate'])
                    net_return = interest_rate
                    for y_idx, y in enumerate(self.income_grid):
                        labor_inc = wage * (y * agent_params['productivity'])
                        after_tax_labor = (1.0 - self.parameters['labor_tax']) * labor_inc
                        a_prime = policy[agent_type, y_idx]
                        c = np.maximum((1.0 + net_return) * a_grid + after_tax_labor - a_prime, eps)
                        if sigma == 1:
                            u = np.log(c)
                        else:
                            u = (c ** (1 - sigma)) / (1 - sigma)
                        # Continuation via interpolation of V
                        cont = np.zeros_like(a_grid)
                        for y_next in range(ny):
                            prob = float(self.income_transition[y_idx, y_next])
                            V_next = V[agent_type, y_next]
                            cont += prob * np.interp(a_prime, a_grid, V_next)
                        V_new[agent_type, y_idx] = u + beta * cont
                if np.max(np.abs(V_new - V)) < 1e-4:
                    V = V_new
                    break
                V = V_new
            
            print("Computing stationary distribution...", flush=True)
            # Compute stationary distribution
            stationary_dist = self._compute_stationary_distribution_from_policy(policy)
            print(f"Stationary distribution shape: {stationary_dist.shape}", flush=True)
            print(f"Stationary distribution sum: {np.sum(stationary_dist):.6f}", flush=True)
            # Save warm-start
            self.last_policy = np.array(policy, copy=True)
            
            # Debug: policy statistics by income state and agent type
            try:
                a_min = float(self.parameters['asset_min'])
                for agent_type in range(n_agent_types):
                    print(f"Policy stats (agent_type={agent_type}):", flush=True)
                    for y_idx, y in enumerate(self.income_grid):
                        a_prime_slice = policy[agent_type, y_idx]
                        min_ap = float(np.min(a_prime_slice))
                        max_ap = float(np.max(a_prime_slice))
                        frac_at_min = float(np.mean(np.isclose(a_prime_slice, a_min, atol=1e-10)))
                        print(f"  y[{y_idx}]={y:.4f} -> min(a')={min_ap:.6f}, max(a')={max_ap:.6f}, frac(a'=a_min)={frac_at_min:.3f}", flush=True)
            except Exception:
                pass

            # Debug: distribution-weighted averages of key quantities
            try:
                wage_dbg = float(self.parameters.get('wage', 1.0))
                interest_rate_dbg = float(self.parameters['interest_rate'])
                net_return_dbg = interest_rate_dbg  # already net rate
                # Broadcast grids
                a_grid_b = self.asset_grid.reshape(1, 1, -1)
                # After-tax labor income per (type, income)
                after_tax_labor = np.zeros((n_agent_types, ny, 1))
                for agent_type in range(n_agent_types):
                    prod = self.agent_types[agent_type]['productivity']
                    for y_idx, y in enumerate(self.income_grid):
                        labor_inc = wage_dbg * (float(y) * prod)
                        after_tax_labor[agent_type, y_idx, 0] = (1.0 - self.parameters['labor_tax']) * labor_inc
                resources = (1.0 + net_return_dbg) * a_grid_b + after_tax_labor
                consumption = np.maximum(resources - policy, 1e-12)
                total_mass = np.sum(stationary_dist)
                w = stationary_dist / max(total_mass, 1e-12)
                Ea = float(np.sum(w * a_grid_b))
                Eap = float(np.sum(w * policy))
                Eres = float(np.sum(w * resources))
                Ec = float(np.sum(w * consumption))
                min_c = float(np.min(consumption))
                print(f"Averages (dist-weighted): E[a]={Ea:.6f}, E[a']={Eap:.6f}, E[resources]={Eres:.6f}, E[c]={Ec:.6f}, min(c)={min_c:.6f}", flush=True)
                
                # Debug: distribution concentration
                mass_at_zero = float(np.sum(stationary_dist[:, :, 0]))  # mass at a=0
                mass_first_10 = float(np.sum(stationary_dist[:, :, :10]))  # mass at first 10 grid points
                total_mass = float(np.sum(stationary_dist))
                print(f"Distribution: mass_at_a=0: {mass_at_zero:.6f}, mass_first_10_points: {mass_first_10:.6f}, total: {total_mass:.6f}", flush=True)
            except Exception:
                pass
            
            return V, policy, stationary_dist
            
        except Exception as e:
            print("\nError in solve_egm:", flush=True)
            print(str(e), flush=True)
            import traceback
            traceback.print_exc()
            raise

    def solve_egm_one_step_given_next_policy(self, next_policy: np.ndarray) -> np.ndarray:
        """
        Perform a single EGM policy update using a provided next-period policy (perfect-foresight step).
        Uses current parameters (interest_rate as net, wage, taxes) and returns the updated policy for the current period.
        next_policy shape: (n_agent_types, ny, na)
        """
        # Set up grids and parameters
        na = self.parameters['asset_grid_size']
        ny = self.parameters['income_grid_size']
        n_agent_types = self.parameters['num_agent_types']
        a_grid = self.asset_grid
        eps = 1e-10

        policy_new = np.zeros((n_agent_types, ny, na))

        for agent_type in range(n_agent_types):
            agent_params = self.agent_types[agent_type]
            beta = agent_params['beta']
            sigma = agent_params['risk_aversion']
            wage = float(self.parameters.get('wage', 1.0))
            interest_rate = float(self.parameters['interest_rate'])  # net
            net_return = interest_rate

            if sigma == 1:
                marg_u = lambda c: 1.0 / np.maximum(c, eps)
                inv_marg_u = lambda mu: 1.0 / np.maximum(mu, eps)
            else:
                marg_u = lambda c: np.minimum(np.maximum(c, eps) ** (-sigma), 1e10)
                inv_marg_u = lambda mu: np.maximum(mu, eps) ** (-1.0 / sigma)

            for y_idx, y in enumerate(self.income_grid):
                labor_inc = wage * (y * agent_params['productivity'])
                after_tax_labor = (1.0 - self.parameters['labor_tax']) * labor_inc
                a_prime_grid = a_grid

                emu = np.zeros_like(a_prime_grid)
                for y_next in range(ny):
                    prob = float(self.income_transition[y_idx, y_next])
                    next_after_tax_labor = (1.0 - self.parameters['labor_tax']) * wage * (self.income_grid[y_next] * agent_params['productivity'])
                    g_next = next_policy[agent_type, y_next]
                    a_pp = np.interp(a_prime_grid, a_grid, g_next)
                    c_next = np.maximum((1.0 + net_return) * a_prime_grid + next_after_tax_labor - a_pp, eps)
                    emu += prob * marg_u(c_next)

                mu_c = beta * (1.0 + net_return) * emu
                mu_c = np.clip(mu_c, 1e-8, 1e8)
                c = inv_marg_u(mu_c)
                a_endo = (c + a_prime_grid - after_tax_labor) / (1.0 + net_return)
                a_endo = np.maximum(a_endo, self.parameters['asset_min'])
                order = np.argsort(a_endo)
                a_endo_sorted = a_endo[order]
                a_prime_sorted = a_prime_grid[order]
                a_prime_interp = np.interp(a_grid, a_endo_sorted, a_prime_sorted, left=a_prime_sorted[0], right=a_prime_sorted[-1])
                resources = (1.0 + net_return) * a_grid + after_tax_labor
                a_prime_feasible = np.clip(a_prime_interp, self.parameters['asset_min'], resources - 1e-8)
                a_prime_monotone = np.maximum.accumulate(a_prime_feasible)
                policy_new[agent_type, y_idx] = a_prime_monotone

        return policy_new
    
    def _compute_stationary_distribution_from_policy(self, policy):
        """
        Compute the stationary distribution from a given policy using sparse eigenvalue computation.
        
        Args:
            policy: Policy function of shape (n_agent_types, ny, na)
            
        Returns:
            Stationary distribution of shape (n_agent_types, ny, na)
        """
        n_agent_types = self.parameters['num_agent_types']
        ny = self.parameters['income_grid_size']
        na = self.parameters['asset_grid_size']
        
        # Initialize stationary distribution
        stationary_dist = np.zeros((n_agent_types, ny, na))
        
        for agent_type in range(n_agent_types):
            # Build joint transition matrix for this agent type
            P = self._build_joint_transition_matrix(policy[agent_type])

            # Deterministic power iteration for stationary distribution
            v = np.ones(P.shape[0], dtype=np.float64) / P.shape[0]
            for _ in range(1000):
                v_next = P.T @ v
                v_next = np.maximum(v_next, 0.0)
                s = float(v_next.sum())
                if s <= 0 or not np.isfinite(s):
                    v_next = np.ones_like(v_next) / v_next.size
                    s = 1.0
                else:
                    v_next = v_next / s
                if np.max(np.abs(v_next - v)) < 1e-12:
                    v = v_next
                    break
                v = v_next

            dist = v
            # Enforce non-negativity before normalization
            dist[~np.isfinite(dist)] = 0.0
            dist = np.maximum(dist, 0.0)
            total = np.sum(dist)
            if total <= 0:
                dist = np.ones_like(dist) / dist.size
            else:
                dist = dist / total  # Ensure proper normalization
            
            # Reshape back to (ny, na)
            dist = dist.reshape(ny, na)
            
            # Store in stationary distribution array
            stationary_dist[agent_type] = dist
        
        # Verify normalization
        total_mass = np.sum(stationary_dist)
        if not np.isclose(total_mass, n_agent_types, rtol=1e-5):
            print(f"Warning: Stationary distribution not properly normalized. Total mass: {total_mass:.6f}, Expected: {n_agent_types}", flush=True)
            # Renormalize
            stationary_dist = stationary_dist / total_mass * n_agent_types
        
        return stationary_dist

    def _build_joint_transition_matrix(self, policy):
        """
        Build the joint transition matrix for income and assets with proper interpolation.
        
        Args:
            policy: Policy function of shape (ny, na) representing savings rates
            
        Returns:
            Sparse transition matrix of shape (ny*na, ny*na)
        """
        ny = self.parameters['income_grid_size']
        na = self.parameters['asset_grid_size']
        
        # Initialize sparse matrix
        P = sparse.csr_matrix((ny*na, ny*na))
        
        # Build transition matrix with interpolation
        for y_curr in range(ny):
            for a_curr in range(na):
                # Get current state index
                curr_idx = y_curr * na + a_curr
                
                # Next assets directly from policy (a')
                a_next = np.clip(policy[y_curr, a_curr], self.parameters['asset_min'], self.parameters['asset_max'])
                
                # Find indices and weights for interpolation
                idx = np.searchsorted(self.asset_grid, a_next, side='right') - 1
                idx = np.clip(idx, 0, na-2)  # Ensure we have room for interpolation
                
                # Compute interpolation weights
                w0 = (self.asset_grid[idx+1] - a_next) / (self.asset_grid[idx+1] - self.asset_grid[idx])
                w1 = 1.0 - w0
                
                # For each possible next income state
                for y_next in range(ny):
                    # Use Markov transition from Rouwenhorst approximation
                    p = float(self.income_transition[y_curr, y_next])
                    
                    # Get next state indices for interpolation
                    next_idx0 = y_next * na + idx
                    next_idx1 = y_next * na + (idx + 1)
                    
                    # Add interpolated probabilities to transition matrix
                    P[curr_idx, next_idx0] += p * w0
                    P[curr_idx, next_idx1] += p * w1
        
        return P

    def find_stationary_distribution(self, policy=None, n_periods=1000, n_agents=1000):
        """
        Find the stationary distribution using EGM.
        
        Args:
            policy: Optional policy function to use. If None, uses EGM to solve.
            n_periods: Number of periods to simulate (only used if policy is provided)
            n_agents: Number of agents to simulate (only used if policy is provided)
            
        Returns:
            Tuple of (asset_distribution, income_distribution, joint_distribution)
        """
        if policy is None:
            # Use EGM to solve for the stationary distribution
            _, _, stationary_dist = self.solve_egm()
            
            # Compute marginal distributions
            n_agent_types = self.parameters['num_agent_types']
            n_income_points = self.parameters['income_grid_size']
            n_asset_points = self.parameters['asset_grid_size']
            
            # Asset distribution
            asset_dist = np.sum(stationary_dist, axis=1)  # Sum over income states
            
            # Income distribution
            income_dist = np.sum(stationary_dist, axis=2)  # Sum over asset states
            
            # Joint distribution is already in the right form
            joint_dist = stationary_dist
            
            return asset_dist, income_dist, joint_dist
        else:
            # Use the original simulation-based method if a policy is provided
            return super().find_stationary_distribution(policy, n_periods, n_agents)

    def plot_stationary_distribution(self, asset_dist, income_dist, joint_dist, save_path=None):
        """
        Plot the stationary distributions.
        
        Args:
            asset_dist: Asset distribution
            income_dist: Income distribution
            joint_dist: Joint distribution
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Stationary Distributions')
        
        # Plot asset distribution
        for agent_type in range(self.parameters['num_agent_types']):
            axes[0, 0].plot(self.asset_grid, asset_dist[agent_type], 
                          label=f'Agent Type {agent_type}')
        axes[0, 0].set_title('Asset Distribution')
        axes[0, 0].set_xlabel('Assets')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot income distribution
        for agent_type in range(self.parameters['num_agent_types']):
            axes[0, 1].plot(self.income_grid, income_dist[agent_type], 
                          label=f'Agent Type {agent_type}')
        axes[0, 1].set_title('Income Distribution')
        axes[0, 1].set_xlabel('Income')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot joint distribution for first agent type
        im = axes[1, 0].imshow(joint_dist[0].T, origin='lower', aspect='auto',
                              extent=[self.income_grid[0], self.income_grid[-1],
                                    self.asset_grid[0], self.asset_grid[-1]])
        axes[1, 0].set_title('Joint Distribution (Agent Type 0)')
        axes[1, 0].set_xlabel('Income')
        axes[1, 0].set_ylabel('Assets')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot joint distribution for second agent type
        im = axes[1, 1].imshow(joint_dist[1].T, origin='lower', aspect='auto',
                              extent=[self.income_grid[0], self.income_grid[-1],
                                    self.asset_grid[0], self.asset_grid[-1]])
        axes[1, 1].set_title('Joint Distribution (Agent Type 1)')
        axes[1, 1].set_xlabel('Income')
        axes[1, 1].set_ylabel('Assets')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            print(f"Current state: income={self.state[0]:.2f}, assets={self.state[1]:.2f}, agent_type={int(self.state[2])}")
            print(f"Parameters: risk_aversion={self.parameters['risk_aversion']:.2f}, "
                  f"income_persistence={self.parameters['income_persistence']:.2f}, "
                  f"income_std={self.parameters['income_std']:.2f}, "
                  f"labor_tax={self.parameters['labor_tax']:.2f}, "
                  f"capital_tax={self.parameters['capital_tax']:.2f}")
        return None 