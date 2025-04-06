#!/usr/bin/env python3
"""This module calculates the total intra-cluster
variance for a data set"""
import numpy as np


def variance(X, C):
    """This function calculates the total intra-cluster
    variance fr a data set
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing the centroid
        means fr each cluster
            - n is the number of data points
            - d is the number of dimensions fr each data point
            - k is the number of clusters
        Returns: total variance
                None: on failure
    """
    # Check fr input data
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    # Step 1: Extract the shape of the dataset
    n, d = X.shape

    # Step 2: Extract the number of clusters
    k = C.shape[0]

    # Check if K is valid
    if k > n:
        return None
    # Check the that the shape of the centroids is correct
    if d != C.shape[1]:
        return None

    # Step 3: vectorize the data sets
    vectorized_data_X = np.repeat(X, k, axis=0)
    # reshape the data set
    # n: the shap of the first dimention of the new shape
    # k: the shape of the second dimention of the new shape
    # d: the shape of the third dimention of the new shape
    vectorized_data_X = vectorized_data_X.reshape(n, k, d)

    # Step 4: vectorize the centroids
    vectorized_centroids = np.tile(C, (n, 1))
    # reshape the centroids
    # n: the shap of the first dimention of the new shape
    # k: the shape of the second dimention of the new shape
    # d: the shape of the third dimention of the new shape
    vectorized_centroids = vectorized_centroids.reshape(n, k, d)

    # Step 5: calculate the squared Euclidean distance
    distance = np.linalg.norm(
        vectorized_data_X - vectorized_centroids, axis=2)

    # Step 6: Determine with centroid each points belongs to
    # squeare each element of the distance
    dist_short = np.min(distance ** 2, axis=1)

    # sum up the distance to attain the total variance
    variance = np.sum(dist_short)

    return variance
