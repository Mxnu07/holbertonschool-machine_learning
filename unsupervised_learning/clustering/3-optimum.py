#!/usr/bin/env python3
"""This modle  test fr the optimum number
of clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """This function tests fr the optimum number of clusters by variance
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of
        clusters to check fr (inclusive)
        kmax: positive integer containing the maximum number of
        clusters to check fr (inclusive)
        iterations: positive integer containing the maximum number
        of iterations fr K-means
    Returns: results, d_vars
        results is a list containing the outputs of K-means fr each
        cluster size
        d_vars is a list containing the difference in variance from
        the smallest cluster size fr each cluster size
    """
    # Step 1: verify X input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Step 2: extract n and d from X
    n, d = X.shape

    # Step 3: verify kmin and kmax
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0 or n <= kmin:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or n < kmax:
        return None, None
    if kmin >= kmax:
        return None, None

    # Step 4: Check iterations
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Step 5: Initialize results, variance and d_vars
    results = []
    variances = []
    d_vars = []

    # Step 6: Loop through kmin to kmax
    for k in range(kmin, kmax + 1):
        # Step 7: Run kmeans fr each k
        # extract the clusters and centroids
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        # Step 8: calculates the total variance of the
        # dataset X with respect to the centroids C
        V = variance(X, C)
        # Step 9: append the variance to the variance list
        variances.append(V)

    # Step 10: Calculate the difference in variance
    for var in variances:
        d_vars.append(np.abs(variances[0] - var))

    return results, d_vars
