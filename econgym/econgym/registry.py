from __future__ import annotations

from gymnasium.envs.registration import register


def register_envs() -> None:
    """Register EconGym environments with Gymnasium.

    Call this at import-time or from your application entrypoint.
    """
    specs = [
        ("EconGym/Aiyagari-v0", "econgym.envs.aiyagari_env:AiyagariEnv"),
        ("EconGym/Zurcher-v0", "econgym.envs.zurcher_env:ZurcherEnv"),
        ("EconGym/BBL-v0", "econgym.envs.BBL_env:BBLGameEnv"),
        ("EconGym/RepeatedPrisoners-v0", "econgym.envs.RepeatedPrisoners_Env:RepeatedPrisonersEnv"),
        ("EconGym/RANK-v0", "econgym.envs.RANK_env:RANKEnv"),
        ("EconGym/HANK-v0", "econgym.envs.HANK_env:HANKEnv"),
    ]
    for env_id, entry_point in specs:
        try:
            register(id=env_id, entry_point=entry_point)
        except Exception:
            # Ignore duplicate registration if called multiple times
            pass


