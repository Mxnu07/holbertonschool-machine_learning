#!/usr/bin/env python3
"""
This module analyzes data in a DataFrame
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp
    column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze

    Returns:
        pd.DataFrame: A DataFrame containing the descriptive statistics.
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df.describe()
