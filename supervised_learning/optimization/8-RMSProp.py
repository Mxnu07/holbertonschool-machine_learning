#!/usr/bin/env python3
"""Script to optimize DNN using RMSprop with Tensorflow"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates an RMSProp optimization operation.

    Parameters:
    - alpha: learning rate (float).
    - beta2: RMSProp weight or discounting factor (float).
    - epsilon: small number to avoid division by zero (float).

    Returns:
    - optimizer: a TensorFlow RMSProp optimizer.
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return optimizer
