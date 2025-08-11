#!/usr/bin/env python3
"""This module sorts data in reverse chronological order"""


def flip_switch(df):
    """This function sorts data in reverse chronological
    order

    Args:
        df: the dataframe to flip

    Returns:
        The flipped dataframe
    """
    # Sort the dataframe in reverse chronological order
    # trasnpose is used to flip the dataframe
    return df.sort_index(ascending=False).transpose()
