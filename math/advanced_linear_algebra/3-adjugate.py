#!/usr/bin/env python3
"""This module calculates the adjugate matrix
of a square matrix
"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Args:
        matrix: list of lists whose adjugate matrix should be calculated
    Returns:
        the adjugate matrix of a matrix
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    # calculate the cofactor matrix
    cofactor_matrix = cofactor(matrix)
    # calculate the adjugate matrix
    adjugate_matrix = [list(row) for row in zip(*cofactor_matrix)]
    return adjugate_matrix
