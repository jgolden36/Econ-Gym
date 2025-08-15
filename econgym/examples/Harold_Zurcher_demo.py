import numpy as np
from econgym.envs.zurcher_env import ZurcherEnv
from econgym.solvers.zurcher_solver import ZurcherSolver

# Initialize the environment
env = ZurcherEnv(max_mileage=100, replace_cost=500, beta=0.9)

# Solve the model with the dedicated solver (tabular, consistent)
solver = ZurcherSolver(env)
policy, V = solver.solve_equilibrium()

# Print the results
print("Optimal Value Function:", V)
print("Optimal Policy:", policy)

print("Solved via ZurcherSolver.")

obs, _ = env.reset()
for _ in range(10):
    action = env.action_space.sample()  # Randomly choose actions
    next_state, reward, done, _, _ = env.step(action)
    print(f"Next State: {next_state}, Reward: {reward}")
    if done:
        obs, _ = env.reset()