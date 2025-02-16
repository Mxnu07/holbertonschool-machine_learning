#!/usr/bin/env python3
"""This module builds the ResNet-50 architecture as
described in Deep Residual Learning for Image Recognition (2015):
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep
    Residual Learning for Image Recognition (2015)

    Returns: the keras model
    """
    # Define the input shape
    inputs = K.Input(shape=(224, 224, 3))

    # Apply convolutions and max pooling before the blocks
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='linear',
                        kernel_initializer=K.initializers.he_normal(
                            seed=0))(inputs)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Apply the blocks
    x = projection_block(x, [64, 64, 256], s=1)
    for _ in range(2):
        x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512])
    for _ in range(3):
        x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048])
    for _ in range(2):
        x = identity_block(x, [512, 512, 2048])

    # Apply average pooling and a softmax layer for the final output
    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(x)
    outputs = K.layers.Dense(1000, activation='softmax')(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=outputs)

    return model
