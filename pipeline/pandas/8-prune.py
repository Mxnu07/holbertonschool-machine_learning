#!/usr/bin/env python3
"""This modlue prunes a dataframe to only contain the columns"""


def prune(df):
    """This function prunes nan fron close in a dataframe

        Args:
            df: the dataframe to prune

        Returns:
            The pruned dataframe
        """
    # Prune the dataframe to only contain the columns 'Close'
    return df.dropna(subset=['Close'])
