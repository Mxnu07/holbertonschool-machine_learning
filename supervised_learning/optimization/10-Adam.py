#!/usr/bin/env python3
"""Script to optimize DNN using Adam with Tensorflow"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates an Adam optimization operation.

    Parameters:
    - alpha: learning rate (float).
    - beta1: weight for the first moment (float).
    - beta2: weight for the second moment (float).
    - epsilon: small number to avoid division by zero (float).

    Returns:
    - optimizer: a TensorFlow Adam optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1,
                                         beta_2=beta2, epsilon=epsilon)
    return optimizer
