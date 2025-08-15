from econgym.envs.TANK_env import TANKEnv
import numpy as np

# Create TANK environment
env = TANKEnv(
    beta=0.99,
    alpha=0.33,
    delta=0.025,
    phi_pi=1.5,
    phi_y=0.5,
    rho_a=0.9,
    rho_r=0.8,
    num_agent_types=2
)

# Find equilibrium
print("Finding equilibrium...")
policy, value = env.find_equilibrium(tol=1e-4, max_iter=1000)

# Simulate for a few periods
print("\nSimulating economy...")
state, _ = env.reset()
for t in range(5):
    action = env.get_policy(state)
    state, reward, done, truncated, info = env.step(action)
    print(f"\nPeriod {t+1}:")
    print(f"State: {state}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")

# Plot stationary distribution
print("\nPlotting stationary distribution...")
env.plot_stationary_distribution() 