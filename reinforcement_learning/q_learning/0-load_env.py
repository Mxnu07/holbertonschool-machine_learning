#!/usr/bin/env python3
"""This module loads the FrozenLakeEnv environment"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """This function loas a pre-made Frozenlake enviroment
    Args:
        desc is either None or a list of lists containing a custom
            description of the map to load for the environment
        map_name is either None or a string containing the pre-made map to load
        is_slippery is a boolean to determine if the ice is slippery
    Returns:
        the environment
    """
    # Load the environment with the custom map
    # FrozenLake-v0 is deprecated so instead use FrozenLake-v1
    ENV = gym.make(
        'FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery,
          render_mode="ansi")

    return ENV
