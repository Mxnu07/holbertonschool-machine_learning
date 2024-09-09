#!/usr/bin/env python3
"""creates a layer, returns tensor output of the layer"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """prev = tensor output of the previous layer
    n = number of nodes in the layer to create
    activation = activation function that the layer should use"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer(prev)
