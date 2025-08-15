# EconGym Documentation

Welcome to EconGym. This site provides:

- Quickstart for installation and running environments
- API overview for `EconEnv` and wrappers
- Model guides (Aiyagari, Zurcher, BBL, RANK, HANK, TANK)

## Install

```bash
pip install -e .
# optional extras
pip install -e .[sb3]         # stable-baselines3
pip install -e .[jax]         # jax, jaxlib, optax
pip install -e .[plot]        # matplotlib, scipy
pip install -e .[pettingzoo]  # PettingZoo multi-agent wrappers
```

## CLI

```bash
# Random rollout
python -m econgym --env EconGym/Zurcher-v0 --steps 1000 --algo random

# SB3 PPO (requires [sb3])
econgym --env EconGym/BBL-v0 --algo sb3-ppo --sb3-timesteps 50000
```

## PettingZoo wrappers

- `econgym.wrappers.pettingzoo.BBLParallelEnv`
- `econgym.wrappers.pettingzoo.RyanParallelEnv`

Usage:
```python
from econgym.wrappers.pettingzoo import BBLParallelEnv

penv = BBLParallelEnv()  # defaults to 3 firms
obs, infos = penv.reset(seed=0)
actions = {agent: penv.action_spaces[agent].sample() for agent in penv.agents}
next_obs, rewards, terms, truncs, infos = penv.step(actions)
```


