from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pettingzoo.utils.env import ParallelEnv  # type: ignore
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - optional dep
    ParallelEnv = None  # type: ignore
    spaces = None  # type: ignore

from econgym.envs.BBL_env import BBLGameEnv
from econgym.envs.Ryan_env import RyanEnv


def _require_pettingzoo() -> None:
    if ParallelEnv is None or spaces is None:
        raise NotImplementedError(
            "PettingZoo is not installed. Install with `pip install econgym[pettingzoo]`."
        )


class BBLParallelEnv(ParallelEnv):  # type: ignore[misc]
    """
    PettingZoo ParallelEnv wrapper for `BBLGameEnv`.
    Agents are named "firm_0", ..., "firm_{n-1}".
    """

    metadata = {"name": "EconGym/BBLParallel-v0", "is_parallelizable": True}

    def __init__(self, env: Optional[BBLGameEnv] = None, *, n_firms: Optional[int] = None):
        _require_pettingzoo()
        self.env = env or BBLGameEnv(n_firms=n_firms or 3)
        self.num_agents = int(self.env.parameters["n_firms"])  # type: ignore[index]
        self.agents: List[str] = [f"firm_{i}" for i in range(self.num_agents)]
        self.possible_agents: List[str] = list(self.agents)

        # Per-agent spaces
        state_dim = int(self.env.parameters["state_dim"])  # type: ignore[index]
        state_low = float(self.env.parameters["state_min"])  # type: ignore[index]
        state_high = float(self.env.parameters["state_max"])  # type: ignore[index]
        price_levels = int(self.env.parameters["price_levels"])  # type: ignore[index]

        box = spaces.Box(low=state_low, high=state_high, shape=(state_dim,), dtype=np.float32)  # type: ignore[attr-defined]
        disc = spaces.Discrete(price_levels)  # type: ignore[attr-defined]

        self.observation_spaces: Dict[str, spaces.Space] = {a: box for a in self.agents}  # type: ignore[name-defined]
        self.action_spaces: Dict[str, spaces.Space] = {a: disc for a in self.agents}  # type: ignore[name-defined]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        observations = {agent: obs[i] for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        # Map dict to array in the agent index order
        action_array = np.array([int(actions[agent]) for agent in self.agents], dtype=np.int64)

        next_obs, rewards, done, truncated, info = self.env.step(action_array)

        observations = {agent: next_obs[i] for i, agent in enumerate(self.agents)}
        rewards_dict = {agent: float(rewards[i]) for i, agent in enumerate(self.agents)}
        terminations = {agent: bool(done) for agent in self.agents}
        truncations = {agent: bool(truncated) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards_dict, terminations, truncations, infos

    def render(self):  # pragma: no cover - passthrough
        return self.env.render()

    def close(self):  # pragma: no cover - passthrough
        return self.env.close()


class RyanParallelEnv(ParallelEnv):  # type: ignore[misc]
    """
    PettingZoo ParallelEnv wrapper for `RyanEnv`.
    Agents are named "firm_0", ..., "firm_{n-1}".
    """

    metadata = {"name": "EconGym/RyanParallel-v0", "is_parallelizable": True}

    def __init__(self, env: Optional[RyanEnv] = None, *, n_firms: Optional[int] = None):
        _require_pettingzoo()
        self.env = env or RyanEnv(n_firms=n_firms or 3)
        self.num_agents = int(self.env.parameters["n_firms"])  # type: ignore[index]
        self.agents: List[str] = [f"firm_{i}" for i in range(self.num_agents)]
        self.possible_agents: List[str] = list(self.agents)

        # Per-agent spaces
        low = float(np.min(self.env.parameters["cost_states"]))  # type: ignore[index]
        high = float(np.max(self.env.parameters["cost_states"]))  # type: ignore[index]
        box = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)  # type: ignore[attr-defined]
        disc = spaces.Discrete(3)  # {0,1,2} production levels  # type: ignore[attr-defined]

        self.observation_spaces: Dict[str, spaces.Space] = {a: box for a in self.agents}  # type: ignore[name-defined]
        self.action_spaces: Dict[str, spaces.Space] = {a: disc for a in self.agents}  # type: ignore[name-defined]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # env.reset returns shape (n_firms,) -> wrap to (1,) per agent
        observations = {agent: np.array([obs[i]], dtype=np.float32) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        action_array = np.array([int(actions[agent]) for agent in self.agents], dtype=np.int64)
        next_obs, rewards, done, truncated, info = self.env.step(action_array)

        observations = {agent: np.array([next_obs[i]], dtype=np.float32) for i, agent in enumerate(self.agents)}
        rewards_dict = {agent: float(rewards[i]) for i, agent in enumerate(self.agents)}
        terminations = {agent: bool(done) for agent in self.agents}
        truncations = {agent: bool(truncated) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards_dict, terminations, truncations, infos

    def render(self):  # pragma: no cover
        return self.env.render()

    def close(self):  # pragma: no cover
        return self.env.close()


