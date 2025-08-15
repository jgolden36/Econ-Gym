"""
Deprecated shim: import wrappers from `econgym.wrappers`.

Use `from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig` instead of `from wrappers.mbrl import ...`.
"""

from __future__ import annotations

import warnings

from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

__all__ = ["MBRLWrapper", "MBRLConfig"]

warnings.warn(
    (
        "Importing from 'wrappers.mbrl' is deprecated and will be removed in a future release. "
        "Please import from 'econgym.wrappers.mbrl' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)