#!/usr/bin/env python3
"""This module contains the function for task 0
that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """This function performs PCA on a dataset
    Args:
        X: numpy.ndarray - shape (n, d) containing the dataset
        var: float - the fraction of the variance that the PCA
        transformation should maintain
        Returns:
        W: numpy.ndarray - shape (d, nd) containing the weights matrix
        that maintains var fraction of X's original variance
        """
    # Compute the SVD:
    U, S, Vt = np.linalg.svd(X)

    # Compute the cumulative sum of the explained variance ratio
    sum_s = np.cumsum(S)

    # Infer 'r' (number of principal components to extract from W/V)
    # based on the 'var' treshold passed as argument to the method
    # Normalize sum_s:
    sum_s = sum_s / sum_s[-1]

    r = np.min(np.where(sum_s >= var))

    # Compute Vr(= Wr):
    V = Vt.T
    Vr = V[..., :r + 1]

    return Vr
