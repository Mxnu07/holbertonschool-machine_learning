#!/usr/bin/env python3
"""converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """one-hot = one-hot encoded numpy.ndarray with shape
    (classes, m)
    classes = number of classes
    m = number of examples"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    # returns numpy array of the indices of the maximum values
    # along the first axis (axis=0)
    return np.argmax(one_hot, axis=0)
