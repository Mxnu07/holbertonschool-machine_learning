#!/usr/bin/env python3
"""This module contains the function
transition_layer."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely
    Connected Convolutional Networks

    Args:
    X -- output frm the previous layer
    nb_filters -- integer representing the number of filters in X
    compression -- the compression factor for the transition layer

    Returns:
    The output of the transition layer and the number of
    filters within the output, respectively
    """
    # Batch Normalization
    X = K.layers.BatchNormalization()(X)
    # Rectified Linear Activation (ReLU)
    X = K.layers.Activation('relu')(X)
    # Compression: using 1x1 convolutions
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(nb_filters, (1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(
                            seed=0))(X)
    # Average pooling layer
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, nb_filters
