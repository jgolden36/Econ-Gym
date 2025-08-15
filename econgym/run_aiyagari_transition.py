from econgym.envs.aiyagari_transition import AiyagariTransition
from econgym.core.transition_wrapper import TransitionConfig
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    try:
        print("Starting Aiyagari transition analysis...", flush=True)
        
        # Define initial parameters (high capital tax state)
        initial_params = {
            'beta': 0.96,  # Discount factor
            'risk_aversion': 2.0,  # CRRA utility parameter
            'income_persistence': 0.9,  # Income shock persistence
            'labor_tax': 0.2,  # Initial labor tax rate
            'capital_tax': 0.397,  # Initial capital tax rate (39.7%)
            'alpha': 0.36,  # Capital share in production
            'delta': 0.08,  # Depreciation rate
            'num_agent_types': 2,  # Number of agent types
            'X': 1.0  # Productivity factor
        }

        print("Initial parameters:", initial_params, flush=True)

        # Define final parameters (zero capital tax state)
        final_params = {
            'beta': 0.96,
            'risk_aversion': 2.0,
            'income_persistence': 0.9,
            'labor_tax': 0.2,  # Initial labor tax (will be adjusted endogenously)
            'capital_tax': 0.0,  # Final capital tax rate (0%)
            'alpha': 0.36,
            'delta': 0.08,
            'num_agent_types': 2,
            'X': 1.0
        }

        print("Final parameters:", final_params, flush=True)

        # Create transition configuration
        config = TransitionConfig(
            num_periods=200,  # Total number of periods to simulate
            num_agents=1000,  # Number of agents to simulate
            burn_in=50,  # Burn-in periods
            transition_period=100,  # When to switch to final parameters
            compute_moments=True,
            save_path="aiyagari_transition_results.png"
        )

        print("Transition configuration:", config, flush=True)

        # Create transition analysis object
        print("\nCreating Aiyagari transition analysis...", flush=True)
        transition = AiyagariTransition(initial_params, final_params, config)

        # Find initial equilibrium
        print("\nFinding initial equilibrium...", flush=True)
        V_initial, g_initial = transition.find_initial_equilibrium()
        print("Initial equilibrium found. Value function shape:", V_initial.shape if V_initial is not None else None, flush=True)
        print("Initial policy shape:", g_initial.shape if g_initial is not None else None, flush=True)

        # Find final equilibrium
        print("\nFinding final equilibrium...", flush=True)
        V_final, g_final = transition.find_final_equilibrium()
        print("Final equilibrium found. Value function shape:", V_final.shape if V_final is not None else None, flush=True)
        print("Final policy shape:", g_final.shape if g_final is not None else None, flush=True)

        # Simulate transition
        print("\nSimulating transition...", flush=True)
        analysis = transition.simulate_transition()
        
        # Print some statistics about the transition
        if analysis and 'transition_path' in analysis:
            print("\nTransition Statistics:", flush=True)
            print("Mean assets over time:", np.mean(analysis['transition_path']['mean_state']), flush=True)
            print("Max assets:", np.max(analysis['transition_path']['mean_state']), flush=True)
            print("Min assets:", np.min(analysis['transition_path']['mean_state']), flush=True)
            print("\nLabor Tax Statistics:", flush=True)
            print("Initial labor tax:", analysis['initial_steady_state']['labor_tax'], flush=True)
            print("Final labor tax:", analysis['final_steady_state']['labor_tax'], flush=True)
            print("Max labor tax:", np.max(analysis['transition_path']['labor_tax']), flush=True)
            print("Min labor tax:", np.min(analysis['transition_path']['labor_tax']), flush=True)

        # Plot results
        print("\nPlotting results...", flush=True)
        transition.plot_transition(analysis)

        print("\nAnalysis complete!", flush=True)

    except Exception as e:
        print(f"Error occurred: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 