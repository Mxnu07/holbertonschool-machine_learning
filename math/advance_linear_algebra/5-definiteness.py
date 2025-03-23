#!/usr/bin/env python3
"""This module calculates the definitness of a
matrix
"""
import numpy as np


def definiteness(matrix):
    """Args:
        matrix: numpy.ndarray of shape (n, n) whose
        definiteness should be calculated
    Returns:
        the string: Positive definite, Positive semi-definite,
        Negative definite,
        Negative semi-definite, Indefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Validate the matrix is square and non-empty
    height = len(matrix)
    if height == 0:
        return None
    width = len(matrix[0])

    if height != width or (height == 1 and width == 0):
        return None
    if not all(len(row) == len(matrix) for row in matrix):
        return None
    if not np.array_equal(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvals(matrix)
    if all(eigenvalues > 0):
        return "Positive definite"
    elif all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif all(eigenvalues < 0):
        return "Negative definite"
    elif all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
