#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using Keras
"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of LeNet-5 architecture using Keras

    parameters:
        X [K.Input of shape (m, 28, 28, 1)]:
            contains the input images for the network

    model layers:
    C1: convolutional layer with 6 kernels of shape (5, 5) with same padding
    P2: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    C3: convolutional layer with 16 kernels of shape (5, 5) with valid padding
    P4: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    F5: fully connected layer with 120 nodes
    F6: fully connected layer with 84 nodes
    F7: fully connected softmax output layer with 10 nodes

    All layers requiring init should initialize kernels with he_normal method
    All hidden layers requiring activation should use relu activation function

    returns:
        K.Model compiled to use Adam optimization (default hyperparameters)
            and accuracy metrics
    """

    # C1: Convolutional layer (6 filters of 5x5, same padding)
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=K.initializers.he_normal(),
    )(X)

    # P2: Max pooling layer (2x2, stride 2)
    P2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C1)

    # C3: Convolutional layer (16 filters of 5x5, valid padding)
    C3 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=K.initializers.he_normal(),
    )(P2)

    # P4: Max pooling layer (2x2, stride 2)
    P4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C3)

    # Flatten the output for fully connected layers
    flatten = K.layers.Flatten()(P4)

    # F5: Fully connected layer (120 nodes)
    F5 = K.layers.Dense(
        120,
        activation="relu",
        kernel_initializer=K.initializers.he_normal(),
    )(flatten)

    # F6: Fully connected layer (84 nodes)
    F6 = K.layers.Dense(
        84,
        activation="relu",
        kernel_initializer=K.initializers.he_normal(),
    )(F5)

    # F7: Fully connected layer (10 nodes with softmax activation)
    F7 = K.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer=K.initializers.he_normal(),
    )(F6)

    # Build the model
    model = K.Model(inputs=X, outputs=F7)

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
