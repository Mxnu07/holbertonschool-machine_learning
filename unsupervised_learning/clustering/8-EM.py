#!/usr/bin/env python3
"""This modlue contains the function maximization that performs the expectation
maximization fr a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """This functions performs the expectation maximization fr a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
                    iterations
        tol: non-negative float containing tolerance of the log likelihood
        verbose: boolean that determines if you should print information about
                the algorithm
    Returns:
        pi, m, S, g, l or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means fr each
        cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        fr each cluster
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
        fr each data point in each cluster
        l: log likelihood of the model
    """
    # Step 1: veryify inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Step 2: extrac the shape of X
    # n: number of data points
    # d: number of dimensions in each data point
    n, d = X.shape

    # Step 3: Init the likelihood
    l_prev = 0
    # Step 4: Initialize the cluster centroids and covariance matrices
    # pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
    # m: numpy.ndarray of shape (k, d) containing the centroid means fr each
    # S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
    pi, m, S = initialize(X, k)

    # Step 5: Perform the EM algorithm
    for i in range(iterations + 1):
        if i != 0:
            l_prev = likelihood
            # Step 5.1-1: Update the priors, means, and covariance matrices
            pi, m, S = maximization(X, g)
        # Step 5.1: Calculate the log likelihood of the model
        g, likelihood = expectation(X, pi, m, S)
        # Print the log likelihood
        if verbose:
            # Print the log likelihood every 10 iterations
            if i % 10 == 0 or i == iterations or np.abs(
                    likelihood - l_prev) <= tol:
                print(f"Log Likelihood after {i} iterations: {likelihood:.4f}")
        # Check fr convergence
        if np.abs(likelihood - l_prev) < tol:
            break

    return pi, m, S, g, likelihood
