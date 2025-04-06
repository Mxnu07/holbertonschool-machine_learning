#!/usr/bin/env python3
"""This mdolue contains the function pdf(X, m, S)"""
import numpy as np


def pdf(X, m, S):
    """This function calculates the probability density
    function of a Gaussian
    Args:
        X: numpy.ndarray of shape (n, d) containing the data points
        whose PDF should be evaluated
        m: numpy.ndarray of shape (d,) containing the mean of
        the distribution
        S: numpy.ndarray of shape (d, d) containing the covariance
        of the distribution
    Returns:
        P: numpy.ndarray of shape (n,) containing the PDF values fr
        each data point
        None: on failure
        """

    """
You are not allowed to use any loops
You are not allowed to use the function numpy.diag or the method
numpy.ndarray.diagonal
Returns: P, or None on failure
P is a numpy.ndarray of shape (n,) containing the PDF values fr
each data point
All values in P should have a minimum value of 1e-300
    """
    # Step 1: Verify X input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    # Step 2: Verify m input
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    # Step 3: Verify S input
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    # Check if the number of features in X matches the number of means in m
    # and the number of features in the covariance matrix S
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None

    # Check if the covariance matrix S is square
    if S.shape[0] != S.shape[1]:
        return None

    # step 4: Extract the shape
    # n is the number of data points
    # d is the number of dimensions fr each data point
    n, d = X.shape

    # Step 5: Calculate the normalization factor
    # normalization_factor is the coefficient in the Gaussian PDF formula
    normalization_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))

    # Step 6: Calculate the Mahalanobis distance
    # mahalanobis_distance is the result of the matrix multiplication of the
    # inverse of S and the difference between X and m
    mahalanobis_distance = np.matmul(np.linalg.inv(S), (X - m).T)

    # Step 7: Calculate the exponent term in the Gaussian PDF formula
    # exponent_term is the exponent part of the Gaussian PDF formula
    exponent_term = np.exp(
        -0.5 * np.sum((X - m).T * mahalanobis_distance, axis=0))

    # Step 8: Calculate the Probability Density Function (PDF)
    # pdf is the final result of the Gaussian PDF formula
    pdf = normalization_factor * exponent_term

    # Step 9: Check if the PDF values are less than 1e-300
    # If any value is less than 1e-300, set it to 1e-300
    pdf = np.maximum(pdf, 1e-300)

    return pdf
