#!/usr/bin/env python3
''' L2 regularization'''

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow with L2 regularization.

    Args:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function for the new layer
    lambtha -- L2 regularization parameter

    Returns:
    output -- the output of the new layer
    """
    # L2 regularization
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    # Layer creation with L2 regularization
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=l2_regularizer)(prev)

    return layer
