#!/usr/bin/env python3
"""This modlue contains the function absorbing that determines if a
markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """This function determines if a markov chain is absorbing
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the standard
        transition matrix
            - P[i, j] is the probability of transitioning from state i
                to state j
            - n is the number of states in the markov chain
            - P will be a square matrix
    Returns:
        True if it is absorbing, or False on failure
    """
    # Step 1: Verify input
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    # Step 2: Ensure the sum of all probabilities is equal to
    # 1 when summing along the rows
    num_states = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(num_states))[0]:
        return None

    # Step 3: Check if the matrix is absorbing
    # Markov chain is absorbing if it has at least one absorbing state
    if np.all(np.diag(P) != 1):
        return False
    # Step 4: If all states are absorbing (P == identity matrix)
    if np.all(np.diag(P) == 1):
        return True

    # Step 5: Check if every state can reach an absorbing state
    for i in range(num_states):
        if np.any(P[i, :] == 1):
            continue
        break

    sub_matrix_I = P[:i, :i]
    identity_matrix = np.identity(num_states - i)
    sub_matrix_R = P[i:, :i]
    sub_matrix_Q = P[i:, i:]

    # Step 6: Calculate the fundamental matrix
    try:
        fundamental_matrix = np.linalg.inv(identity_matrix - sub_matrix_Q)
    except Exception:
        return False

    # Step 7: Calculate the product of the fundamental matrix and sub_matrix_R
    FR_product = np.matmul(fundamental_matrix, sub_matrix_R)
    limiting_matrix = np.zeros((num_states, num_states))
    limiting_matrix[:i, :i] = sub_matrix_I
    limiting_matrix[i:, :i] = FR_product

    sub_matrix_Qbar = limiting_matrix[i:, i:]
    # Step 8: Check if the sub_matrix_Qbar is zero
    if np.all(sub_matrix_Qbar == 0):
        return True

    return False
