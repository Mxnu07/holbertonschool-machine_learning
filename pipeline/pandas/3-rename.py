#!/usr/bin/env python3
"""This module renames a column of a dataframe"""
import pandas as pd


def rename(df):
    """This function renames the column Timestamp to Datetime,
    converts the timestamp values to datetime values,
    and displays only the Datetime and Close columns.

    Args:
        df: the dataframe to modify

    Returns:
        The modified dataframe
    """
    # Rename the column 'Timestamp' to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the 'Datetime' column to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Display only the 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
