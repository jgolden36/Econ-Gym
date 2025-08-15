import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from econgym.envs.aiyagari_env import AiyagariEnv
from econgym.core.transition_wrapper import TransitionWrapper, TransitionConfig
try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
except Exception:  # pragma: no cover - optional
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore
try:
    from scipy.optimize import bisect  # type: ignore
except Exception:  # pragma: no cover - optional
    bisect = None  # type: ignore


class AiyagariTransition:
    """
    A class to handle transition dynamics in the Aiyagari model.
    This class uses the TransitionWrapper to analyze how the economy
    transitions between steady states when tax parameters change.
    The interest rate is determined endogenously in equilibrium.
    """
    
    def __init__(self, 
                 initial_params: Dict[str, Any],
                 final_params: Dict[str, Any],
                 config: Optional[TransitionConfig] = None):
        """
        Initialize the transition analysis.
        
        Args:
            initial_params: Dictionary of initial parameter values (including taxes)
            final_params: Dictionary of final parameter values (including taxes)
            config: Configuration for transition analysis
        """
        # Extract X parameter if present
        X_initial = initial_params.pop('X', 1.0)
        X_final = final_params.pop('X', 1.0)
        
        # Create initial environment
        self.initial_env = AiyagariEnv(**initial_params)
        self.initial_env.parameters['X'] = X_initial
        
        # Create transition wrapper
        self.config = config or TransitionConfig(
            num_periods=200,
            num_agents=1000,
            burn_in=50,
            transition_period=100,
            compute_moments=True,
            save_path="aiyagari_transition.png"
        )
        self.transition_env = TransitionWrapper(self.initial_env, self.config)
        
        # Store parameters
        self.initial_params = initial_params
        self.final_params = final_params
        self.initial_params['X'] = X_initial
        self.final_params['X'] = X_final
        
        # Initialize solver
        self.solver = None
        self.vec_env = DummyVecEnv([lambda: self.transition_env])
        
        # Store equilibrium objects
        self.initial_equilibrium = None
        self.final_equilibrium = None
    
    def find_equilibrium_interest_rate(self, env: AiyagariEnv, tol: float = 1e-6) -> float:
        """
        Find the equilibrium interest rate that clears the capital market using an iterative approach.
        Caches household solutions to avoid unnecessary recomputation.
        
        Args:
            env: The Aiyagari environment
            tol: Tolerance for market clearing
            
        Returns:
            Equilibrium interest rate
        """
        print("Starting equilibrium search...", flush=True)
        
        # Initial guess for interest rate - start with a higher value
        r = 0.04  # Increased from 0.03
        env.parameters['interest_rate'] = r
        
        # Initialize labor supply
        L_supply = 1.0
        
        # Initialize difference and iteration counter
        difference_market_clearing = 100.0
        iteration = 0
        max_iterations = 1000  # Increased from 500
        
        # Cache for household solutions
        cached_r = None
        cached_stationary_dist = None
        r_change_threshold = 0.0005  # Increased from 0.0001
        
        # Small constant to avoid division by zero
        eps = 1e-8  # Decreased from 1e-10 for better numerical stability
        
        print(f"Initial interest rate: {r:.4f}", flush=True)
        
        # Track best solution
        best_r = r
        best_difference = float('inf')
        
        while difference_market_clearing > tol and iteration < max_iterations:
            try:
                # Update capital demand using firm's FOC: MPK = r + δ
                K_demand = ((env.parameters['alpha'] * env.parameters['X']) / 
                          max(r + env.parameters['delta'], eps))**(1/(1-env.parameters['alpha'])) * L_supply
                
                # Update wage using firm's FOC: MPL = w
                w = (1 - env.parameters['alpha']) * env.parameters['X'] * K_demand**env.parameters['alpha'] * L_supply**(-env.parameters['alpha'])
                
                # Check if we need to recompute household solution
                if cached_r is None or abs(r - cached_r) > r_change_threshold:
                    print(f"Recomputing household solution at r = {r:.4f}", flush=True)
                    # Solve for value function and policy using EGM
                    _, _, stationary_dist = env.solve_egm()
                    cached_r = r
                    cached_stationary_dist = stationary_dist
                else:
                    # Use cached solution
                    stationary_dist = cached_stationary_dist
                
                # Compute aggregate capital supply
                asset_grid = env.asset_grid.reshape(1, 1, -1)
                K_supply = np.sum(stationary_dist * asset_grid)
                
                # Compute market clearing difference
                difference_market_clearing = abs(K_supply - K_demand)
                
                # Update best solution if current is better
                if difference_market_clearing < best_difference:
                    best_difference = difference_market_clearing
                    best_r = r
                
                # Update interest rate with adaptive dampening
                # Use smaller adjustment when close to equilibrium
                dampening = 0.01 * (1.0 - min(1.0, difference_market_clearing / K_demand))
                adjustment = dampening * (K_supply - K_demand) / max(K_demand, eps)
                r_new = r + adjustment
                
                # Ensure r stays in reasonable bounds
                r_new = np.clip(r_new, -env.parameters['delta'] + eps, 0.5)
                
                # Check for oscillation
                if iteration > 1 and abs(r_new - r) < eps:
                    print("Warning: Interest rate oscillation detected, using best solution")
                    r = best_r
                    break
                
                r = r_new
                env.parameters['interest_rate'] = r
                
                iteration += 1
                
                if iteration % 5 == 0:
                    print(f"Iteration {iteration}:", flush=True)
                    print(f"  Market clearing difference = {difference_market_clearing:.6f}", flush=True)
                    print(f"  Current interest rate: {r:.6f}", flush=True)
                    print(f"  Capital supply: {K_supply:.6f}, Capital demand: {K_demand:.6f}", flush=True)
                    print(f"  Wage: {w:.6f}", flush=True)
                    print(f"  Stationary dist sum: {np.sum(stationary_dist):.6f}", flush=True)
                
            except (RuntimeWarning, RuntimeError, ValueError) as e:
                print(f"Warning: Numerical error at iteration {iteration}: {str(e)}", flush=True)
                # Use best solution found so far
                r = best_r
                break
        
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations reached. Using best solution found.")
            r = best_r
        
        # Update final capital and wage
        K_final = ((env.parameters['alpha'] * env.parameters['X'])/(r + env.parameters['delta']))**(1/(1-env.parameters['alpha'])) * L_supply
        w_final = (1 - env.parameters['alpha']) * env.parameters['X'] * K_final**env.parameters['alpha'] * L_supply**(-env.parameters['alpha'])
        
        print(f"\nEquilibrium found:", flush=True)
        print(f"Interest rate: {r:.4f}", flush=True)
        print(f"Capital: {K_final:.4f}", flush=True)
        print(f"Wage: {w_final:.4f}", flush=True)
        print(f"Capital to output ratio: {K_final / (K_final**env.parameters['alpha'] * L_supply**(1-env.parameters['alpha'])):.4f}", flush=True)
        
        return r
    
    def find_initial_equilibrium(self):
        """Find the initial steady state equilibrium."""
        print("\nFinding initial equilibrium...", flush=True)
        # Find equilibrium interest rate
        r_eq = self.find_equilibrium_interest_rate(self.initial_env)
        self.initial_env.parameters['interest_rate'] = r_eq
        
        print("Solving for value function and policy...", flush=True)
        # Solve for value function and policy
        V, g, stationary_dist = self.initial_env.solve_egm()
        
        self.initial_equilibrium = {
            'interest_rate': r_eq,
            'value_function': V,
            'policy': g,
            'stationary_dist': stationary_dist
        }
        
        return V, g
    
    def find_final_equilibrium(self):
        """Find the final steady state equilibrium."""
        print("\nFinding final equilibrium...", flush=True)
        # Create new environment with final parameters
        final_env = AiyagariEnv(**self.final_params)
        
        # Find equilibrium interest rate
        r_eq = self.find_equilibrium_interest_rate(final_env)
        final_env.parameters['interest_rate'] = r_eq
        
        print("Solving for value function and policy...", flush=True)
        # Solve for value function and policy
        V, g, stationary_dist = final_env.solve_egm()
        
        self.final_equilibrium = {
            'interest_rate': r_eq,
            'value_function': V,
            'policy': g,
            'stationary_dist': stationary_dist
        }
        
        return V, g
    
    def calculate_government_revenue(self, env: AiyagariEnv, stationary_dist: np.ndarray) -> float:
        """
        Calculate total government revenue from labor and capital taxes.
        
        Args:
            env: The Aiyagari environment
            stationary_dist: Stationary distribution of agents
            
        Returns:
            Total government revenue
        """
        # Calculate aggregate capital
        asset_grid = env.asset_grid.reshape(1, 1, -1)
        K = np.sum(stationary_dist * asset_grid)
        
        # Calculate aggregate labor (normalized to 1)
        L = 1.0
        
        # Calculate factor prices
        r = env.parameters['interest_rate']
        w = (1 - env.parameters['alpha']) * env.parameters['X'] * K**env.parameters['alpha'] * L**(-env.parameters['alpha'])
        
        # Calculate tax revenue
        labor_revenue = env.parameters['labor_tax'] * w * L
        capital_revenue = env.parameters['capital_tax'] * r * K
        
        return labor_revenue + capital_revenue

    def adjust_labor_tax(self, env: AiyagariEnv, stationary_dist: np.ndarray, target_revenue: float, tol: float = 1e-6) -> float:
        """
        Adjust labor tax to maintain target government revenue with improved convergence.
        
        Args:
            env: The Aiyagari environment
            stationary_dist: Stationary distribution of agents
            target_revenue: Target government revenue
            tol: Tolerance for revenue matching
            
        Returns:
            Adjusted labor tax rate
        """
        # Initial bounds for labor tax
        tax_min = 0.0
        tax_max = 1.0
        
        # Track best solution
        best_tax = env.parameters['labor_tax']
        best_revenue_diff = float('inf')
        
        # Maximum iterations
        max_iter = 50
        iteration = 0
        
        # Previous tax for dampening
        prev_tax = env.parameters['labor_tax']
        
        while tax_max - tax_min > tol and iteration < max_iter:
            # Try middle tax rate with dampening
            if iteration == 0:
                new_tax = (tax_min + tax_max) / 2
            else:
                # Use dampening to avoid oscillations
                dampening = 0.5 * (1.0 - min(1.0, iteration / max_iter))
                new_tax = prev_tax + dampening * ((tax_min + tax_max) / 2 - prev_tax)
            
            env.parameters['labor_tax'] = new_tax
            
            # Calculate revenue with current tax
            revenue = self.calculate_government_revenue(env, stationary_dist)
            revenue_diff = abs(revenue - target_revenue)
            
            # Update best solution
            if revenue_diff < best_revenue_diff:
                best_revenue_diff = revenue_diff
                best_tax = new_tax
            
            # Update bounds
            if revenue > target_revenue:
                tax_max = new_tax
            else:
                tax_min = new_tax
            
            prev_tax = new_tax
            iteration += 1
        
        # Use best solution found
        env.parameters['labor_tax'] = best_tax
        return best_tax

    def simulate_transition(self, policy: Optional[Any] = None) -> Dict[str, Any]:
        """
        Simulate the transition between steady states with improved convergence.
        """
        # Initialize arrays to store results
        n_periods = self.config.num_periods
        n_agents = self.config.num_agents
        
        mean_state = np.zeros(n_periods)
        std_state = np.zeros(n_periods)
        mean_action = np.zeros(n_periods)
        mean_reward = np.zeros(n_periods)
        labor_taxes = np.zeros(n_periods)
        government_revenues = np.zeros(n_periods)  # Track government revenue
        
        # Phase 1: Use EGM policies
        print("Phase 1: Using EGM policies for initial transition path...")
        current_env = self.initial_env
        current_policy = self.initial_equilibrium['policy']
        
        # Calculate initial government revenue
        initial_revenue = self.calculate_government_revenue(current_env, self.initial_equilibrium['stationary_dist'])
        print(f"Initial government revenue: {initial_revenue:.4f}")
        
        # Initialize agent states
        agent_states = []
        agent_types = []
        for _ in range(n_agents):
            state, _ = current_env.reset()
            agent_type = int(state[2])
            agent_states.append(state)
            agent_types.append(agent_type)
        
        # Previous labor tax for dampening
        prev_labor_tax = current_env.parameters['labor_tax']
        
        # Simulate transition using EGM policies
        for t in range(n_periods):
            if t == self.config.transition_period:
                # Switch to final environment and policy
                current_env = AiyagariEnv(**self.final_params)
                current_env.parameters['interest_rate'] = self.final_equilibrium['interest_rate']
                current_policy = self.final_equilibrium['policy']
                
                # Initialize states for the new environment
                for i in range(n_agents):
                    state, _ = current_env.reset()
                    agent_states[i] = state
                    agent_types[i] = int(state[2])
            
            # Simulate one period for all agents
            states = []
            actions = []
            rewards = []
            
            for i in range(n_agents):
                state = agent_states[i]
                agent_type = agent_types[i]
                
                income_idx = int(np.searchsorted(current_env.income_grid, state[0]))
                asset_idx = int(np.searchsorted(current_env.asset_grid, state[1]))
                
                income_idx = np.clip(income_idx, 0, len(current_env.income_grid) - 1)
                asset_idx = np.clip(asset_idx, 0, len(current_env.asset_grid) - 1)
                
                action = current_policy[agent_type, income_idx, asset_idx]
                action = np.array([action])
                
                next_state, reward, _, _, _ = current_env.step(action)
                
                agent_states[i] = next_state
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            
            # Compute moments
            mean_state[t] = np.mean([s[1] for s in states])
            std_state[t] = np.std([s[1] for s in states])
            mean_action[t] = np.mean([a[0] for a in actions])
            mean_reward[t] = np.mean(rewards)
            
            # Calculate current stationary distribution with more iterations
            _, _, stationary_dist = current_env.solve_egm(max_iter=2000, tol=1e-8)
            
            # Adjust labor tax to maintain government revenue
            labor_tax = self.adjust_labor_tax(current_env, stationary_dist, initial_revenue)
            
            # Apply dampening to labor tax changes
            if t > 0:
                dampening = 0.3  # Reduced from 0.5 for smoother transitions
                labor_tax = prev_labor_tax + dampening * (labor_tax - prev_labor_tax)
            
            labor_taxes[t] = labor_tax
            prev_labor_tax = labor_tax
            
            # Calculate and store current government revenue
            current_revenue = self.calculate_government_revenue(current_env, stationary_dist)
            government_revenues[t] = current_revenue
            
            if t % 10 == 0:
                print(f"Period {t}:")
                print(f"  Mean Assets = {mean_state[t]:.2f}")
                print(f"  Labor Tax = {labor_tax:.4f}")
                print(f"  Government Revenue = {current_revenue:.4f}")
                print(f"  Revenue Difference = {current_revenue - initial_revenue:.4f}")
        
        # Store results
        analysis = {
            'transition_path': {
                'mean_state': mean_state,
                'std_state': std_state,
                'mean_action': mean_action,
                'reward': mean_reward,
                'labor_tax': labor_taxes,
                'government_revenue': government_revenues
            },
            'initial_steady_state': {
                'mean_state': mean_state[self.config.transition_period-1],
                'std_state': std_state[self.config.transition_period-1],
                'mean_action': mean_action[self.config.transition_period-1],
                'labor_tax': labor_taxes[self.config.transition_period-1],
                'government_revenue': government_revenues[self.config.transition_period-1]
            },
            'final_steady_state': {
                'mean_state': mean_state[-1],
                'std_state': std_state[-1],
                'mean_action': mean_action[-1],
                'labor_tax': labor_taxes[-1],
                'government_revenue': government_revenues[-1]
            }
        }
        
        return analysis
    
    def plot_transition(self, analysis: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot the transition analysis with additional government revenue information.
        """
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Aiyagari Model Transition Analysis')
        
        # Plot state moments
        axes[0, 0].plot(analysis['transition_path']['mean_state'])
        axes[0, 0].set_title('Mean Assets')
        axes[0, 0].axvline(x=self.config.transition_period, color='r', linestyle='--')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(analysis['transition_path']['std_state'])
        axes[0, 1].set_title('Asset Standard Deviation')
        axes[0, 1].axvline(x=self.config.transition_period, color='r', linestyle='--')
        axes[0, 1].grid(True)
        
        # Plot action and labor tax
        axes[1, 0].plot(analysis['transition_path']['mean_action'])
        axes[1, 0].set_title('Mean Savings Rate')
        axes[1, 0].axvline(x=self.config.transition_period, color='r', linestyle='--')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(analysis['transition_path']['labor_tax'])
        axes[1, 1].set_title('Labor Tax Rate')
        axes[1, 1].axvline(x=self.config.transition_period, color='r', linestyle='--')
        axes[1, 1].grid(True)
        
        # Plot government revenue
        axes[2, 0].plot(analysis['transition_path']['government_revenue'])
        axes[2, 0].set_title('Government Revenue')
        axes[2, 0].axvline(x=self.config.transition_period, color='r', linestyle='--')
        axes[2, 0].grid(True)
        
        # Plot revenue difference from target
        initial_revenue = analysis['initial_steady_state']['government_revenue']
        revenue_diff = analysis['transition_path']['government_revenue'] - initial_revenue
        axes[2, 1].plot(revenue_diff)
        axes[2, 1].set_title('Revenue Difference from Target')
        axes[2, 1].axvline(x=self.config.transition_period, color='r', linestyle='--')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def main():
    """Example usage of the AiyagariTransition class."""
    print("Starting Aiyagari transition analysis...", flush=True)
    
    # Define initial parameters (high capital tax)
    initial_params = {
        'beta': 0.96,
        'risk_aversion': 2.0,
        'income_persistence': 0.9,
        'labor_tax': 0.2,
        'capital_tax': 0.397,  # Initial capital tax
        'alpha': 0.36,
        'delta': 0.08,
        'num_agent_types': 2,
        'X': 1.0  # Productivity factor
    }
    
    # Define final parameters (lower capital tax)
    final_params = initial_params.copy()
    final_params['capital_tax'] = 0.25  # Lower capital tax
    
    print("Creating transition analysis...", flush=True)
    # Create transition analysis
    trans = AiyagariTransition(initial_params, final_params)
    
    # Find initial equilibrium
    print("\nFinding initial equilibrium...", flush=True)
    V_initial, g_initial = trans.find_initial_equilibrium()
    
    # Find final equilibrium
    print("\nFinding final equilibrium...", flush=True)
    V_final, g_final = trans.find_final_equilibrium()
    
    # Simulate transition
    print("\nSimulating transition...", flush=True)
    analysis = trans.simulate_transition()
    
    # Plot results
    print("\nPlotting results...", flush=True)
    trans.plot_transition(analysis, save_path="aiyagari_transition.png")
    
    # Print summary statistics
    print("\nTransition Analysis Summary:")
    print(f"Initial Capital Tax: {initial_params['capital_tax']:.3f}")
    print(f"Final Capital Tax: {final_params['capital_tax']:.3f}")
    print(f"Initial Steady State Mean Assets: {analysis['initial_steady_state']['mean_state']:.4f}")
    print(f"Final Steady State Mean Assets: {analysis['final_steady_state']['mean_state']:.4f}")
    print(f"Initial Steady State Asset Volatility: {analysis['initial_steady_state']['std_state']:.4f}")
    print(f"Final Steady State Asset Volatility: {analysis['final_steady_state']['std_state']:.4f}")
    
    print("\nAnalysis complete! Results saved to aiyagari_transition.png", flush=True)


if __name__ == "__main__":
    main() 