#!/usr/bin/env python3
"""Function that performs matrix multiplication."""


def mat_mul(mat1, mat2):
    """Perform matrix multiplication."""
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(row1, col2))
             for col2 in zip(*mat2)]
            for row1 in mat1]
