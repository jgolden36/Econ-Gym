"""
Deprecated shim: re-export EconEnv from the canonical package module.

Use `from econgym.core.base_env import EconEnv` instead of `from core.base_env import EconEnv`.
"""

from __future__ import annotations

import warnings

from econgym.core.base_env import EconEnv, EconEnvWrapper

__all__ = ["EconEnv", "EconEnvWrapper"]

warnings.warn(
    (
        "Importing from 'core.base_env' is deprecated and will be removed in a future release. "
        "Please import 'EconEnv' from 'econgym.core.base_env' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)


