#!/usr/bin/env python3
"""This module concatenates two dataframes"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """This function concatenates two dataframes

    Args:
        df: the dataframe to concatenate

    Returns:
        The concatenated dataframe
    """
    # Index both DataFrames on their Timestamp column
    df1 = index(df1)
    df2 = index(df2)

    # Filter the second DataFrame up to the specified Timestamp
    df2 = df2.loc[:1417411920]

    # NOTE concatenated dataframes, using keys to differentiate data origin
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
