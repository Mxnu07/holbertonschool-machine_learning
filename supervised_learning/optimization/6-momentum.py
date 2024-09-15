#!/usr/bin/env python3
"""Script of momentum in Tensorflow"""

import tensorflow as tf


import tensorflow as tf

def create_momentum_op(alpha, beta1):
    """
    Creates a momentum optimization operation.

    Parameters:
    - alpha: learning rate (float).
    - beta1: momentum weight (float).

    Returns:
    - optimizer: a TensorFlow momentum optimizer.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
