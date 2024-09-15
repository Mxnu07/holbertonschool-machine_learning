#!/usr/bin/env python3
"""creates mini-batches to be used for training a neural network
using mini-batch gradient descent"""
import numpy as np


def create_mini_batches(X, Y, batch_size):
    """X = numpy.ndarray of shape (m, nx) input data
    m = number of data points
    nx = number of features in X
    Y = numpy.ndarray of shape (m, ny) the labels
    m = same number of data points as in X
    ny = number of classes for classification tasks
    batch_size = data points in a batch
    Returns: a list mini-batches of tuples (X_batch, Y_batch)"""
    m = X.shape[0]
    mini_batches = []

    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)

    num_complete_minibatches = m // batch_size

    for k in range(0, num_complete_minibatches):
        X_mini_batch = X[k * batch_size: k * batch_size + batch_size, :]
        Y_mini_batch = Y[k * batch_size: k * batch_size + batch_size, :]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append((mini_batch))

    if m % batch_size != 0:
        X_mini_batch = X[num_complete_minibatches * batch_size: m, :]
        Y_mini_batch = Y[num_complete_minibatches * batch_size: m, :]
        mini_batch = (X_mini_batch, Y_mini_batch)
        mini_batches.append((mini_batch))

    return mini_batches
