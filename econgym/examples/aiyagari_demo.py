import numpy as np
from econgym.envs.aiyagari_env import AiyagariEnv

# 1. Instantiate environment
env = AiyagariEnv(r=0.03, a_max=30.0)

# 2. Load or generate synthetic data
data = ...

# 3. Placeholder for calibration/estimation (requires proper data/moment functions)
# from econgym.core.estimation import calibrate, gmm_estimate
# calibrate(env, data, target_moments={'mean_assets': 10.0})
# gmm_estimate(env, data, moment_function=my_moment_fn, weight_matrix=W)

# 5. Use env
obs, _ = env.reset()
for t in range(10):
    action = env.action_space.sample()  # or a policy
    next_obs, reward, done, truncated, info = env.step(action)
    if done:
        break
