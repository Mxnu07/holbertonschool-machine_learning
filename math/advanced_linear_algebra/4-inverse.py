#!/usr/bin/env python3
"""This module calculates the inverse
of a square matrix
"""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Args:
        matrix: list of lists whose inverse should be calculated
    Returns:
        the inverse of a matrix
    """
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

    detA = determinant(matrix)
    # Chck is its a singular matrix
    if detA == 0:
        return None

    adjugateA = adjugate(matrix)
    # Iterate over the rowns and columns
    # of the adjugate matrix
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            # Divide each element of the adjugate matrix
            # by the determinant of the matrix
            adjugateA[row][col] /= detA
    return adjugateA
