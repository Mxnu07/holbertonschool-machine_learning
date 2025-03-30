#!/usr/bin/env python3
"""This modlue contains the function pca(X, ndim)"""
import numpy as np


def pca(X, ndim):
    """This function performs PCA on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
        ndim: the new dimensionality of the transformed X
        Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed version
        of X
        """
    # Ensure all dimetions have a mean between all data points
    X = X - np.mean(X, axis=0)

    # Compute the SVD:
    U, S, Vt = np.linalg.svd(X)

    # Compute the weights matrix
    W = Vt[:ndim].T

    # Transform the data
    T = np.matmul(X, W)

    return T
