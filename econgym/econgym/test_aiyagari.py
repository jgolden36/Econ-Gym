import numpy as np
import matplotlib.pyplot as plt
from econgym.envs.aiyagari_env import AiyagariEnv
from econgym.envs.aiyagari_transition import AiyagariTransition
from econgym.core.transition_wrapper import TransitionConfig

def test_aiyagari_transition():
    print("Initializing Aiyagari transition analysis...")
    
    # Define initial parameters
    initial_params = {
        'beta': 0.96,
        'risk_aversion': 2.0,
        'income_persistence': 0.9,
        'labor_tax': 0.2,
        'capital_tax': 0.397,  # Initial capital tax
        'alpha': 0.36,
        'delta': 0.08,
        'num_agent_types': 2,
        'X': 1.0
    }
    
    # Define final parameters (with different capital tax)
    final_params = initial_params.copy()
    final_params['capital_tax'] = 0.25  # Lower capital tax
    
    # Create transition configuration
    config = TransitionConfig(
        num_periods=200,
        num_agents=1000,
        burn_in=50,
        transition_period=100,
        compute_moments=True,
        save_path="aiyagari_transition.png"
    )
    
    # Initialize transition analysis
    transition = AiyagariTransition(initial_params, final_params, config)
    
    print("\nFinding initial equilibrium...")
    V_initial, g_initial = transition.find_initial_equilibrium()
    
    print("\nFinding final equilibrium...")
    V_final, g_final = transition.find_final_equilibrium()
    
    print("\nSimulating transition...")
    analysis = transition.simulate_transition()
    
    print("\nPlotting transition...")
    transition.plot_transition(analysis)
    
    print("\nTransition analysis complete!")
    print("\nInitial steady state:")
    print(f"Mean assets: {analysis['initial_steady_state']['mean_state']:.4f}")
    print(f"Asset volatility: {analysis['initial_steady_state']['std_state']:.4f}")
    print(f"Mean savings rate: {analysis['initial_steady_state']['mean_action']:.4f}")
    
    print("\nFinal steady state:")
    print(f"Mean assets: {analysis['final_steady_state']['mean_state']:.4f}")
    print(f"Asset volatility: {analysis['final_steady_state']['std_state']:.4f}")
    print(f"Mean savings rate: {analysis['final_steady_state']['mean_action']:.4f}")

if __name__ == "__main__":
    test_aiyagari_transition() 