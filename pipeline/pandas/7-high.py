#!/usr/bin/env python3
"""This module sorts a dataframe in descending order"""


def high(df):
    """This function sorts a dataframe in descending order

    Args:
        df: the dataframe to sort

    Returns:
        The sorted dataframe
    """
    # Sort the dataframe in descending order
    return df.sort_values(by='High', ascending=False)
