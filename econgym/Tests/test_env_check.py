import gymnasium as gym
from gymnasium.utils.env_checker import check_env


def test_registered_envs_pass_checker():
    # Import registers envs via econgym.__init__
    import econgym  # noqa: F401

    env_ids = [
        "EconGym/Aiyagari-v0",
        "EconGym/Zurcher-v0",
        "EconGym/BBL-v0",
        "EconGym/RepeatedPrisoners-v0",
    ]

    for env_id in env_ids:
        env = gym.make(env_id)
        check_env(env, warn=True)
        env.close()


