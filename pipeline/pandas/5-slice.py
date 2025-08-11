#!/usr/bin/env python3
"""This module slices a dataframe along the
columns High and Close"""


def slice(df):
    """This function slices a dataframe along the
    columns High and Close.

    Args:
        df: the dataframe to slice

    Returns:
        The sliced dataframe
    """
    # Return the columns 'High' and 'Close'
    # iloc[::60] is used to slice the dataframe every 60 rows
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
