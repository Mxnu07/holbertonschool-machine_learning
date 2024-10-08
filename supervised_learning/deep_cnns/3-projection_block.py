#!/usr/bin/env python3
""" Deep CNNs """

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for Image
        Recognition (2015):
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection.
    s is the stride of the first convolution in both the main path and the
        shortcut connection
    All convolutions inside the block be followed by batch normalization along
      the channels axis and a rectified linear activation (ReLU), respectively.
    All weights use he normal initialization
    Returns: the activated output of the projection block
    """

    F11, F3, F12 = filters

    out = K.layers.Conv2D(F11, 1, s, kernel_initializer='he_normal')(A_prev)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Activation('relu')(out)

    out = K.layers.Conv2D(F3, 3, padding='same',
                          kernel_initializer='he_normal')(out)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Activation('relu')(out)

    out = K.layers.Conv2D(F12, 1, kernel_initializer='he_normal')(out)
    out = K.layers.BatchNormalization()(out)
    out2 = K.layers.Conv2D(F12, 1, s, kernel_initializer='he_normal')(A_prev)
    out2 = K.layers.BatchNormalization()(out2)
    out = K.layers.add([out, out2])
    out = K.layers.Activation('relu')(out)

    return out
