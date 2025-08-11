#!/usr/bin/env python3
"""This modlue create a pd.DataFrame from a dictionary"""
import pandas as pd


# Create the dictionary
data = {'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']}
# Create the index
index = ['A', 'B', 'C', 'D']
# Create the pd.DataFrame
df = pd.DataFrame(data, index=index)
