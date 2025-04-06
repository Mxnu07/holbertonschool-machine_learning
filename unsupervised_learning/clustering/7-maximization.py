#!/usr/bin/env python3
"""This modlue contains the function maximization(X, g)
"""
import numpy as np


def maximization(X, g):
    """This function calculates the maximization step in the
    EM algorithm fr a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
        fr each data point in each cluster
    Returns:
        pi: numpy.ndarray of shape (k,) containing the updated priors fr
        each cluster
        m: numpy.ndarray of shape (k, d) containing the updated centroid means
        fr
        each cluster
        S: numpy.ndarray of shape (k, d, d) containing the updated covariance
        matrices
        fr each cluster
        None, None, None: on failure
    """
    # Step 1: Verify inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    # Step 2: Extract the shape of X
    # n: number of data points
    # d: number of dimensions in each data point
    n, d = X.shape

    # Step 3: Extract the shape of g
    # k: number of clusters
    k = g.shape[0]

    # Step 4: Verify the shapes
    if g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None

    # Ensure the sum of all posteriors (over the k clusters) is equal to 1
    if not np.isclose(np.sum(g, axis=0), np.ones(n,)).all():
        return None, None, None

    # Step 5: Calculate the updated priors
    pi = np.sum(g, axis=1) / n

    # Step 6: Calculate the updated means
    m = np.matmul(g, X) / np.sum(g, axis=1).reshape(-1, 1)

    # Step 7: Calculate the updated covariance matrices
    S = np.zeros((k, d, d))
    for cluster in range(k):
        # Calculate the covariance matrix
        X_m = X - m[cluster]
        # np.dot - calculates the dot product of two matrices
        # np.sum - calculates the sum of the elements in an
        # np.dot / np.sum - calculates the covariance matrix
        S[cluster] = np.dot(g[cluster] * X_m.T, X_m) / np.sum(g[cluster])

    return pi, m, S
