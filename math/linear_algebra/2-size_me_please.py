#!/usr/bin/env python3
"""Function that returns size of the shape of a matrix."""
import numpy as np


def matrix_shape(matrix):
    """Calculate the shape of a matrix using NumPy."""
    np_matrix = np.array(matrix)
    return list(np_matrix.shape)
