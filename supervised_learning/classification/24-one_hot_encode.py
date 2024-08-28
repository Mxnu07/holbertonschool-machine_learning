#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Y = numpy.ndarray of shape (m,) containing
    numeric class labels
    m = number of examples
    classes = maximum number of classes found in Y"""
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.amax(Y):
        return None
    # creates new numpy array of zeros with shape (classes, m)
    one_hot = np.zeros((classes, Y.shape[0]))
    # sets value of matrix to 1 at the index of the class
    # labels in Y
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot
