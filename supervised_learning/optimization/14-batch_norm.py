#!/usr/bin/env python3
"""Script to create a batch normalization layer in a DNN
    using tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    - prev: activated output of the previous layer.
    - n: number of nodes in the layer to be created.
    - activation: activation function to be applied after batch normalization

    Returns:
    - A tensor of the activated output for the layer.
    """
    # Define kernel initializer
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    # Create Dense layer
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)

    # Create Batch Normalization layer
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,  # Apply normalization along the last axis(the feature axis)
        momentum=0.99,
        epsilon=1e-7,
        beta_initializer=tf.keras.initializers.Zeros(),  # Init beta to 0
        gamma_initializer=tf.keras.initializers.Ones(),  # Init gamma to 1
    )(dense)

    # Apply the activation function
    if activation:
        output = activation(batch_norm)
    else:
        output = batch_norm

    return output
