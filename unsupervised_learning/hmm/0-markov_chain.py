#!/usr/bin/env python3
"""This modlue contains the function markov_chain that
determines the probability"""
import numpy as np


def markov_chain(P, s, t=1):
    """This function determines the probability of a markov chain being in a
    particular state after a specified number of iterations
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
        n: number of states in the markov chain
        s: 2D numpy.ndarray of shape (1, n) representing the probability of
        starting in each state
        t: number of iterations that the markov chain has been through
    Returns:
        numpy.ndarray of shape (1, n) representing the probability of being in
        a specific state after t iterations, or None on failure
    """
    # Step 1: veryify inputs
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    # Check if P is a square matrix
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    # Check if s is a row vector
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t <= 0:
        return None

    # Step 2: ensure that the transition matrix is a valid probability matrix
    probs = P.shape[0]
    if not np.all(P >= 0):
        return None

    # Step 3: ensure that the sum of each row of P is equal to 1
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    # Step 4: calculate the probability of being in a specific state after t
    # iterations
    for i in range(t):
        s = np.matmul(s, P)
    return s
