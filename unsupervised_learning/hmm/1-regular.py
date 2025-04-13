#!/usr/bin/env python3
"""This modlue contains the function regular that determines
the steady state"""
import numpy as np


def regular(P):
    """This function determines the steady state probabilities of a regular
    markov chain
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
            n: number of states in the markov chain
            P[i, j]: probability of transitioning from state i to state j
            P[i, j] = 0 if the state i cannot transition to state j
            P[i, j] + P[i, k] + ... + P[i, n-1] = 1
    Returns:
        numpy.ndarray of shape (1, n) containing the steady state
        probabilities,
        or None on failure
    """
    # Step 1: verify input
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    # Step 2: ensure that the transition matrix is a valid probability matrix
    probs = P.shape[0]
    if not np.all(P >= 0):
        return None

    # Step 3: ensure that the sum of each row of P is equal to 1
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    # Step 4: initialice a 1D  for the starting probabilities
    starting_probs = np.full(probs, 1 / probs)[np.newaxis, ...]

    # Step 5: make a deep copy of P
    P_copy = np.copy(P)
    # initialize the steady state probabilities prev
    steady_state = np.copy(starting_probs)

    # Step 6: calculate the steady state probabilities
    while True:
        # multiply the starting probabilities by P
        P_copy = np.matmul(P_copy, P)
        # check if any value of P_copy is less than or equal to 0
        if np.any(P_copy <= 0):
            return None
        # Reinitialize starting probs to p
        starting_probs = np.matmul(starting_probs, P)
        # check if the steady state probabilities have been reached
        if np.all(starting_probs == steady_state):
            return starting_probs
        # update the steady state probabilities
        steady_state = np.copy(starting_probs)
