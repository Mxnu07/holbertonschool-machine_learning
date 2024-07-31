#!/usr/bin/env python3
"""Function that adds two 2D matrices along a specific axis."""


def add_arrays(arr1, arr2):
    """Add two arrays along a specific axis."""
    if len(arr1) != len(arr2):
        return None
    if len(arr1[0]) != len(arr2[0]):
        return None
    new_arr = []
    for i in range(len(arr1)):
        new_row = []
        for j in range(len(arr1[0])):
            new_row.append(arr1[i][j] + arr2[i][j])
        new_arr.append(new_row)
    return new_arr
