#!/usr/bin/env python3
"""This module calculate the determinant
of a given matrix"""


def determinant(matrix):
    """A function that calculates the determinant of a matrix
    Args:
        matrix (list): list of lists whose determinant should
        be calculated
    Returns:
        int: the determinant of the matrix
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if its and empty list
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    # Determinant of a 0x0 matrix represented as [[]]
    elif len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    elif len(matrix) == 1:
        return matrix[0][0]  # Determinant of a 1x1 matrix

    # Check if matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for a 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    # Recursive case for matrix larger than 2x2
    det = 0
    for c in range(len(matrix)):
        # Remove column c from the matrix
        minor = [row[:c] + row[c + 1:] for row in matrix[1:]]
        cofactor = (-1) ** c * matrix[0][c] * determinant(minor)
        det += cofactor
    return det
