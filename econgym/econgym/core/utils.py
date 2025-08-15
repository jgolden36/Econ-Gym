import os
import random
from typing import Optional

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """
    Seed all major RNGs for reproducibility across NumPy, Python, Torch, and hashing.

    Args:
        seed: The random seed to use
        deterministic_torch: If True and torch is available, enforce deterministic algorithms
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch (optional)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
                # Some ops require this env for reproducibility
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            except Exception:
                # Fallback silently if not supported
                pass


class NumpyRNGMixin:
    """
    Mixin to provide a per-instance NumPy Generator (`self.rng`) and a helper to seed it.
    Intended for use in environments to avoid global RNG mutation.
    """

    rng: np.random.Generator

    def _ensure_rng(self) -> None:
        if not hasattr(self, "rng") or self.rng is None:
            self.rng = np.random.default_rng()

    def reseed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self._ensure_rng()


