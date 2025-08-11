#!/usr/bin/env python3
"""This modlue sets a colum as an index for the dataframe"""


def index(df):
    """This function sets the column Timestamp as the index
    of the dataframe

    Args:
        df: the dataframe to set the index

    Returns:
        The dataframe with the index set
    """
    # Set the column 'Timestamp' as the index of the dataframe
    if 'Timestamp' in df.columns:
        df = df.set_index('Timestamp')
    return df
