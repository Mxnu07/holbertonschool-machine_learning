#!/usr/bin/env python3
'''Create a Layer with Dropout '''

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function for the new layer
    keep_prob -- probability that a node will be kept
    training -- boolean indicating whether the model is in training mode

    Returns:
    output -- the output of the new layer with dropout applied
    """
    # L2 regularization using variance scaling initializer
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Dense layer creation
    layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=initializer)(prev)

    # Apply dropout only during training
    if training:
        output = tf.keras.layers.Dropout(rate=1 - keep_prob)(layer)
    else:
        output = layer

    return output
