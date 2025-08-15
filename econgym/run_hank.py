import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
import os

from econgym.envs.HANK_env import HANKEnv


def main():
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    env = HANKEnv(num_agents=50)
    state, _ = env.reset()

    # Optional: compute equilibrium (can be slow). Commented by default.
    # policy, _ = env.find_equilibrium()

    T = 200
    states = np.zeros((T, env.parameters['num_agents'], 3))
    rewards = np.zeros((T, env.parameters['num_agents']))
    infos = []

    states[0] = state
    for t in range(1, T):
        actions = env.get_policy(states[t-1]) if hasattr(env, 'equilibrium_policy') else env.action_space.sample()
        s, r, done, trunc, info = env.step(actions)
        states[t] = s
        rewards[t-1] = r
        infos.append(info)
        if done or trunc:
            break

    # Build simple aggregates over time
    K_agg = np.sum(states[:, :, 0], axis=1)
    A_agg = np.exp(np.mean(states[:, :, 1], axis=1))
    # Approximate L as sum of exp(log A_i)
    L_agg = np.sum(np.exp(states[:, :, 1]), axis=1)

    # Plot aggregates
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(K_agg)
    axes[0, 0].set_title('Aggregate Capital')
    axes[0, 0].set_xlabel('Time')

    axes[0, 1].plot(A_agg)
    axes[0, 1].set_title('Aggregate Productivity (A)')
    axes[0, 1].set_xlabel('Time')

    axes[1, 0].plot(L_agg)
    axes[1, 0].set_title('Aggregate Effective Labor (L)')
    axes[1, 0].set_xlabel('Time')

    # Cross-sectional distribution (last period)
    axes[1, 1].hist(states[-1, :, 0], bins=30, alpha=0.8)
    axes[1, 1].set_title('Cross-sectional Capital (final period)')
    axes[1, 1].set_xlabel('K')

    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'hank_simulation.png')
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == '__main__':
    main()


