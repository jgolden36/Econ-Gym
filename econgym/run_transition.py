from econgym.envs.aiyagari_transition import AiyagariTransition
from econgym.core.transition_wrapper import TransitionConfig
from stable_baselines3 import PPO
import sys
import traceback

def main():
    try:
        print("Starting main function...", flush=True)
        
        # Define initial and final parameters
        initial_params = {
            'beta': 0.96,  # Discount factor
            'risk_aversion': 2.0,  # CRRA utility parameter
            'income_persistence': 0.9,  # Income shock persistence
            'labor_tax': 0.2,  # Labor income tax rate
            'capital_tax': 0.397,  # Initial capital tax rate
            'alpha': 0.36,  # Capital share in production
            'delta': 0.08,  # Depreciation rate
            'num_agent_types': 2,  # Number of agent types
            'X': 1.0  # Productivity factor
        }
        
        print("Initial parameters defined", flush=True)
        
        final_params = {
            'beta': 0.96,
            'risk_aversion': 2.0,
            'income_persistence': 0.9,
            'labor_tax': 0.2,
            'capital_tax': 0.0,  # Final capital tax rate (removed)
            'alpha': 0.36,
            'delta': 0.08,
            'num_agent_types': 2,
            'X': 1.0
        }
        
        print("Final parameters defined", flush=True)
        
        # Create a faster configuration
        config = TransitionConfig(
            num_periods=100,  # Reduced from 200
            num_agents=500,   # Reduced from 1000
            burn_in=25,       # Reduced from 50
            transition_period=50,  # Reduced from 100
            compute_moments=True,
            save_path="aiyagari_transition.png"
        )
        
        print("Configuration created", flush=True)
        
        # Create transition analysis
        print("Creating transition analysis...", flush=True)
        transition = AiyagariTransition(initial_params, final_params, config)
        print("Transition analysis object created", flush=True)
        
        # Find equilibria
        print("\nFinding initial equilibrium...", flush=True)
        initial_policy, initial_value = transition.find_initial_equilibrium()
        print(f"Initial equilibrium interest rate: {transition.initial_equilibrium['interest_rate']:.4f}", flush=True)
        
        print("\nFinding final equilibrium...", flush=True)
        final_policy, final_value = transition.find_final_equilibrium()
        print(f"Final equilibrium interest rate: {transition.final_equilibrium['interest_rate']:.4f}", flush=True)
        
        # Simulate transition with reduced PPO timesteps
        print("\nSimulating transition...", flush=True)
        if transition.solver is None:
            transition.solver = PPO("MlpPolicy", transition.vec_env, verbose=0)
            transition.solver.learn(total_timesteps=50000)  # Reduced from 100000
        analysis = transition.simulate_transition()
        
        # Plot results
        print("\nPlotting transition...", flush=True)
        transition.plot_transition(analysis, "aiyagari_transition.png")
        
        # Print summary statistics
        print("\nTransition Analysis Summary:", flush=True)
        print("Initial Steady State (39.7% Capital Tax):", flush=True)
        print(f"Mean Capital: {analysis['initial_steady_state']['mean_state']:.2f}", flush=True)
        print(f"Capital Volatility: {analysis['initial_steady_state']['std_state']:.2f}", flush=True)
        print(f"Mean Savings Rate: {analysis['initial_steady_state']['mean_action']:.2f}", flush=True)
        print(f"Interest Rate: {transition.initial_equilibrium['interest_rate']:.4f}", flush=True)
        
        print("\nFinal Steady State (0% Capital Tax):", flush=True)
        print(f"Mean Capital: {analysis['final_steady_state']['mean_state']:.2f}", flush=True)
        print(f"Capital Volatility: {analysis['final_steady_state']['std_state']:.2f}", flush=True)
        print(f"Mean Savings Rate: {analysis['final_steady_state']['mean_action']:.2f}", flush=True)
        print(f"Interest Rate: {transition.final_equilibrium['interest_rate']:.4f}", flush=True)
        
    except Exception as e:
        print("\nError occurred:", flush=True)
        print(str(e), flush=True)
        print("\nTraceback:", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 