#!/usr/bin/env python3
"""This module contines the q_init function"""
import numpy as np


def q_init(env):
    """This functions initializes a Q-table
    Args:
        env: the FrozenLakeEnv instance
    Returns:
        the Q-table as a numpy.ndarray of zeros
    """
    # Q-table is a 2D array of zeros with dimensions (states, actions)
    # states = env.observation_space.n
    # actions = env.action_space.n
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    return q_table
