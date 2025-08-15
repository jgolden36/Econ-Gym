#!/usr/bin/env python3

import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("✓ NumPy version:", np.__version__)
except ImportError as e:
    print("✗ NumPy import failed:", e)

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print("✗ Matplotlib import failed:", e)

try:
    import scipy
    print("✓ SciPy version:", scipy.__version__)
except ImportError as e:
    print("✗ SciPy import failed:", e)

try:
    from econgym.envs.aiyagari_env import AiyagariEnv
    print("✓ Successfully imported AiyagariEnv from econgym.envs.aiyagari_env")
except ImportError as e:
    print("✗ ImportError (econgym.envs):", e)
    try:
        from envs.aiyagari_env import AiyagariEnv
        print("✓ Successfully imported AiyagariEnv from envs.aiyagari_env")
    except ImportError as e2:
        print("✗ Both imports failed:", e2)

# Test basic functionality
try:
    env = AiyagariEnv()
    print("✓ AiyagariEnv instantiated successfully")
    print("  - Has solve_egm method:", hasattr(env, 'solve_egm'))
    print("  - Asset grid shape:", env.asset_grid.shape)
    print("  - Income grid shape:", env.income_grid.shape)
except Exception as e:
    print("✗ Error creating AiyagariEnv:", e)
