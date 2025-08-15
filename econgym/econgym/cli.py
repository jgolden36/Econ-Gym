import argparse
import sys
from typing import Optional

import gymnasium as gym


def _run_random(env_id: str, steps: int, seed: Optional[int]) -> None:
    import econgym  # registers envs

    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    episode_returns = []

    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward) if not isinstance(reward, (list, tuple)) else float(sum(reward))
        if terminated or truncated:
            episode_returns.append(total_reward)
            total_reward = 0.0
            obs, info = env.reset()

    if total_reward != 0.0:
        episode_returns.append(total_reward)

    if episode_returns:
        avg = sum(episode_returns) / len(episode_returns)
        print(f"Completed {len(episode_returns)} episodes. Average return: {avg:.4f}")
    else:
        print("No completed episodes; printing cumulative return over steps.")
        print(f"Cumulative return: {total_reward:.4f}")


def _run_sb3_ppo(env_id: str, timesteps: int, seed: Optional[int]) -> None:
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        print(
            "stable-baselines3 is not installed. Install with `pip install econgym[sb3]`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    import econgym  # registers envs

    # SB3 expects a vectorized env
    def make_env():
        return gym.make(env_id)

    vec_env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)
    model.learn(total_timesteps=int(timesteps))

    # Simple rollout after training
    env = make_env()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 1000
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward) if not isinstance(reward, (list, tuple)) else float(sum(reward))
        if terminated or truncated:
            break
    print(f"Evaluation return over {steps} steps (or until done): {total_reward:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="econgym",
        description="Run EconGym environments with simple rollouts or SB3 PPO.",
    )
    parser.add_argument(
        "--env",
        "-e",
        default="EconGym/Aiyagari-v0",
        help="Gymnasium environment ID (default: EconGym/Aiyagari-v0)",
    )
    parser.add_argument(
        "--algo",
        "-a",
        choices=["random", "sb3-ppo"],
        default="random",
        help="Algorithm to use (default: random)",
    )
    parser.add_argument(
        "--steps",
        "-n",
        type=int,
        default=1000,
        help="Number of steps for random rollout (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--sb3-timesteps",
        type=int,
        default=10000,
        help="Total timesteps for SB3 PPO training (default: 10000)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.algo == "random":
        _run_random(env_id=args.env, steps=args.steps, seed=args.seed)
    elif args.algo == "sb3-ppo":
        _run_sb3_ppo(env_id=args.env, timesteps=args.sb3_timesteps, seed=args.seed)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown algo: {args.algo}")


if __name__ == "__main__":  # pragma: no cover
    main()


