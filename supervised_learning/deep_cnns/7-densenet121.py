#!/usr/bin/env python3
"""This module contains the function for
DenseNet-121 architecture"""
from tensorflow import keras as K

# Import the required functions
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks

    Args:
    growth_rate -- the growth rate
    compression -- the compression factor

    Returns:
    the keras model
    """
    # Define the input shape
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution
    X = K.layers.BatchNormalization()(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(2 * growth_rate, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer=K.initializers.he_normal(
                            seed=0))(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Dense block (1)
    X, nb_filters = dense_block(X, 2 * growth_rate, growth_rate, 6)

    # Transition layer (1)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block (2)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition layer (2)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block (3)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition layer (3)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block (4)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.AveragePooling2D((7, 7))(X)
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=K.initializers.he_normal(
                           seed=0))(X)

    # Create Keras model instance
    model = K.Model(inputs=inputs, outputs=X)

    return model
