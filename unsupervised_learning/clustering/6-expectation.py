#!/usr/bin/env python3
"""This modulle contains the function expectation(X, pi, m, S)
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """This function calculates the expectation step in the
    EM algorithm fr a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means
        fr each cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        fr each cluster
    Returns:
            g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities
            fr each data point in each cluster
            l: is the total log likelihood
            None, None: on failure
    """
    # Step 1: Verify inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    # Step 2: Extract the shape of X
    # n: number of data points
    # d: number of dimensions in each data point
    n, d = X.shape

    # Step 3: verify the shapes
    if pi.shape[0] > n:
        return None, None
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    # Verify the sum of all priors is equal to 1
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    # Step 4: Initiaslized and array to store the posteriors
    pos = np.zeros((k, n))

    # Step 5: iterate over the clusters to calculate the posteriors
    for cluster in range(k):
        # calculate the posterior probability of the data points
        PDF = pdf(X, m[cluster], S[cluster])

        # Step 6: calculate the posterior probability of the data points
        pos[cluster] = pi[cluster] * PDF

    # Calculate the sum of the posterior probabilities (g) across all clusters
    # sum_gis is a 1D array where each element is the sum of the posterior
    # probabilities fr a specific data point
    sum_pos = np.sum(pos, axis=0, keepdims=True)
    pos /= sum_pos

    # Step 8: Calculate the log likelihood of the data
    # li is the total log likelihood
    li = np.sum(np.log(sum_pos))

    return pos, li
