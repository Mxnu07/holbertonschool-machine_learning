#!/usr/bin/env python3
"""Function that flips a 2D matrix over its main diagonal."""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix."""
    transposed = []
    for i in range(len(matrix[0])):
        new_row = []
        for row in matrix:
            new_row.append(row[i])
        transposed.append(new_row)
    return transposed
