import numpy as np
from gymnasium import Wrapper
from typing import Dict, Any, Optional, Tuple

class ValueFunctionWrapper(Wrapper):
    """
    A wrapper that adds value function estimation to an environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self.value_function = None
        
    def get_value_function(self, state):
        """
        Get the value function for a given state.
        """
        if hasattr(self.env, 'get_value_function'):
            return self.env.get_value_function(state)
        return None
        
    def get_policy(self, state):
        """
        Get the policy for a given state.
        """
        if hasattr(self.env, 'get_policy'):
            return self.env.get_policy(state)
        return None 