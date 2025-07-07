#!/usr/bin/env python3
"""This module hoes the epsilo_greedy funtion"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """This function uses epsilon_greedy to determine the next action
    Args:
        Q: a numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation
    Returns:
        the next action index
    """
    # Epsilon-greedy is a simple method to balance
    # exploration and exploitation
    # If the random number is less than epsilon, we should explore
    # Otherwise, we should exploit
    if np.random.uniform(0, 1) < epsilon:
        # Explore
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit
        action = np.argmax(Q[state])

    return action
