import numpy as np


def test_aiyagari_smoke():
    # Import lazily to avoid optional heavy deps during collection
    from econgym.envs.aiyagari_env import AiyagariEnv

    env = AiyagariEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape[0] == env.observation_space.shape[0]

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, (float, np.floating)) or np.isscalar(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


def _smoke(env_cls):
    env = env_cls()
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5


def test_zurcher_smoke():
    from econgym.envs.zurcher_env import ZurcherEnv
    _smoke(ZurcherEnv)


def test_bbl_smoke():
    from econgym.envs.BBL_env import BBLGameEnv
    _smoke(BBLGameEnv)


