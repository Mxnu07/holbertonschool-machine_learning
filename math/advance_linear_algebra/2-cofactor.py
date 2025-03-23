#!/usr/bin/env python3
"""This module calculates the cofactor matrix
of a square matrix"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    Args:
        matrix: list of lists whose cofactor matrix should be calculated
    Returns:
        the cofactor matrix of a matrix
        """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    # Validate the matrix is square and non-empty
    height = len(matrix)
    width = len(matrix[0])

    if height != width or (height == 1 and width == 0):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            minor_matrix = minor(matrix)
            cofactor_row.append(((-1) ** (i + j)) * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix
