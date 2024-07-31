#!/usr/bin/env python3
"""Function that adds two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Add two arrays element-wise."""
    # Check if both arrays have the same length
    if len(arr1) != len(arr2):
        return None
    # Add corresponding elements of both arrays
    return [a + b for a, b in zip(arr1, arr2)]
