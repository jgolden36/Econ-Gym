## EconGym

Economic environments and tooling compatible with Gymnasium for RL, calibration, and estimation.

### Install

```bash
pip install -e .
# optional extras
pip install -e .[sb3]   # stable-baselines3
pip install -e .[jax]   # jax, jaxlib, optax
pip install -e .[plot]  # matplotlib, scipy
pip install -e .[dev]   # pytest, ruff, mypy, black
```

### Quick start

```python
import gymnasium as gym
import econgym  # registers envs

env = gym.make("EconGym/Aiyagari-v0")
obs, info = env.reset(seed=0)
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Environments
- Aiyagari (heterogeneous agents, savings)
- Zurcher (bus replacement)
- BBL (dynamic competition)
- Repeated Prisoners’ Dilemma

### Development
- Run tests: `pytest -q`
- Lint: `ruff check econgym examples tests`
- Type-check: `mypy econgym`

### Notes
- Heavy dependencies (Torch, SB3, SciPy, Matplotlib, JAX) are optional extras.
- Legacy imports like `from core.base_env import EconEnv` are still supported with a deprecation warning; prefer `from econgym.core.base_env import EconEnv`.


