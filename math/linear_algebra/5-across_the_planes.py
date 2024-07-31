#!/usr/bin/env python3
"""Function that adds two matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Add two matrices element-wise."""
    # Check if both matrices have the same shape
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    # Add corresponding elements of both matrices
    return [[a + b for a, b in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]
