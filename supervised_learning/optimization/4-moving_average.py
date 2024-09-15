#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """data = list of data to calculate the moving average of
    beta = weight used for the moving average
    Returns: list containing the moving averages of data"""
    m_avg = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        m_avg.append(v / (1 - beta ** (i + 1)))
    return m_avg
