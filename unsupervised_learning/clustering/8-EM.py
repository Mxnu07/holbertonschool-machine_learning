#!/usr/bin/env python3
"""This module contains the function expectation_maximization that performs
the expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """This function performs the expectation maximization for a GMM
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of iterations
        tol: non-negative float containing tolerance of the log likelihood
        verbose: boolean that determines if you should print information about
                 the algorithm
    
    Returns:
        pi, m, S, g, l or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means for each
            cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
            for each cluster
        g: numpy.ndarray of shape (k, n) containing the probabilities for each
            data point in each cluster
        l: log likelihood of the model
    """
    # Step 1: verify inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Step 2: Initialize the cluster parameters
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Step 3: Perform the EM algorithm with one loop
    l_prev = None
    i = 0
    
    while i < iterations:
        # E-step
        g, likelihood = expectation(X, pi, m, S)
        if g is None or likelihood is None:
            return None, None, None, None, None

        # Print log likelihood if verbose
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {round(likelihood, 5)}")

        # Check for convergence
        if l_prev is not None and abs(likelihood - l_prev) <= tol:
            break

        # M-step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = likelihood
        i += 1

    # Final E-step for return values
    g, likelihood = expectation(X, pi, m, S)
    if g is None or likelihood is None:
        return None, None, None, None, None

    return pi, m, S, g, likelihood
