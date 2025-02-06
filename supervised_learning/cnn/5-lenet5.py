#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using Keras
"""


from tensorflow import keras as K
from tensorflow.keras import layers


def lenet5(X):
    """ Function that builds a modified version of the LeNet-5 architecture"""
    initializer = K.initializers.he_normal(seed=0)

    # 1. First Convolutional Layer
    conv1 = layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # 2. First Max Pooling Layer
    pool1 = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv1)

    # 3. Second Convolutional Layer
    conv2 = layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # 4. Second Max Pooling Layer
    pool2 = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv2)

    # Flatten the output for Fully Connected Layers
    flatten = layers.Flatten()(pool2)

    # 5. First Fully Connected Layer
    fc1 = layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # 6. Second Fully Connected Layer
    fc2 = layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # 7. Output Layer (Softmax)
    output = layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Create the model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
