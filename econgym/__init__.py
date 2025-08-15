"""
EconGym: A Gymnasium-based framework for economic models.
"""

from econgym.core.base_env import EconEnv, EconEnvWrapper
from econgym.wrappers.value_function import ValueFunctionWrapper
from econgym.wrappers.mbrl import MBRLWrapper, MBRLConfig

# Import environments
from econgym.envs.zurcher_env import ZurcherEnv
from econgym.envs.aiyagari_env import AiyagariEnv
from econgym.envs.RANK_env import RANKEnv
from econgym.envs.HANK_env import HANKEnv
from econgym.envs.TANK_env import TANKEnv

# Import transfer learning utilities
from econgym.demos.transfer_learning_demo import (
    ValueFunctionNetwork,
    train_value_function,
    calibrate_with_network,
    estimate_with_network
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "EconEnv",
    "EconEnvWrapper",
    "ValueFunctionWrapper",
    "MBRLWrapper",
    "MBRLConfig",
    
    # Environments
    "ZurcherEnv",
    "AiyagariEnv",
    "RANKEnv",
    "HANKEnv",
    "TANKEnv",
    
    # Transfer learning utilities
    "ValueFunctionNetwork",
    "train_value_function",
    "calibrate_with_network",
    "estimate_with_network"
]