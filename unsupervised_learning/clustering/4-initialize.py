#!/usr/bin/env python3
"""This modlue initializes variables fr a Gaussian
Mixture Model
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """This function initializes variables fr a Gaussian Mixture Model
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
    Returns:
    pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
    m: numpy.ndarray of shape (k, d) containing the centroid means fr
    each cluster
    S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
    fr each cluster
    or None, None, None on failure
    """
    # Step 1: verify X input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Step 2: extract n and d from X
    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None

    # Step 3: Initialize pi
    # pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
    # np.full(k, fill_value=1/k) initializes pi as a numpy.ndarray
    # of shape (k,)
    pi = np.full((k,), fill_value=1/k)

    # Step 4: Initialize m
    # kmeans(X, k) initializes the centroids
    # and returns the centroids and clusters
    # m: numpy.ndarray of shape (k, d) containing the centroid means
    # fr each cluster
    # _ is the clusters
    m, _ = kmeans(X, k)

    # Step 5: Initialize S
    # is a numpy.ndarray of shape (k, d, d)
    # containing the covariance matrices fr each cluster,
    # initialized as identity matrices
    # np.full - creates a numpy.ndarray of shape (k, d, d)
    # np.identity - creates an identity matrix of shape (d, d)
    # and identity matrix is a square matrix with ones on the main diagonal
    S = np.full((k, d, d), np.identity(d))

    return pi, m, S
