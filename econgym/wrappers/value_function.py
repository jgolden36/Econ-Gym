"""
Deprecated shim: import wrappers from `econgym.wrappers`.

Use `from econgym.wrappers.value_function import ValueFunctionWrapper` instead of `from wrappers.value_function import ...`.
"""

from __future__ import annotations

import warnings

from econgym.wrappers.value_function import ValueFunctionWrapper

__all__ = ["ValueFunctionWrapper"]

warnings.warn(
    (
        "Importing from 'wrappers.value_function' is deprecated and will be removed in a future release. "
        "Please import from 'econgym.wrappers.value_function' instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)