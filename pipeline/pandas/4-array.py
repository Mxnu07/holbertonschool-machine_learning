#!/usr/bin/env python3
"""This module converts a dataframe to a numpy array"""


def array(df):
    """This function converts a dataframe to a numpy array.

    Args:
        df: the dataframe to convert

    Returns:
        The numpy array
    """
    # Return the last 10 rows of the columns 'High' and 'Close'
    # as a numpy array
    return df[['High', 'Close']].tail(10).to_numpy()
