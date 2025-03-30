#!/usr/bin/env python3
"""This module contains the function for task 0
that calculates the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """This function calculates the mean and covariance of a data set
    Args:
        X: numpy.ndarray - shape (n, d) that contains the data set
    Returns:
        mean: numpy.ndarray - shape (1, d) containing the mean of the data set
        covariance: numpy.ndarray - shape (d, d) containing the
        covariance matrix
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean and covariance
    n = X.shape[0]
    mean = np.mean(X, axis=0).reshape(1, X.shape[1])
    covariance = np.dot((X - mean).T, (X - mean)) / (n - 1)

    return mean, covariance
