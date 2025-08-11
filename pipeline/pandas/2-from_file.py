#!/usr/bin/env python3
"""This module loads data from a file into a dataframe"""
import pandas as pd


def from_file(filename, delimiter):
    """This function loads data from a file into a dataframe
    Args:
        filename: the file to load from
        delimiter: the column separator
        Returns: the loaded dataframe
    """
    return pd.read_csv(filename, delimiter=delimiter)
