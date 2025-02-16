#!/usr/bin/env python3
"""This module builds a projection block as
described in Deep Residual Learning for Image Recognition (2015):
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015)

    Args:
        A_prev: output frm the previous layer
        filters: tuple or list containing F11, F3, F12
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution as
            well as the 1x1 convolution in the shortcut connection
        s: stride of the first convolution in both the main path and
        the shortcut connection

    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters

    # Save the input value for the shortcut
    X_shortcut = A_prev

    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                        padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                        padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1), strides=(s, s),
                                 padding='valid',
                                 kernel_initializer=K.initializers.he_normal(
                                     seed=0))(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
