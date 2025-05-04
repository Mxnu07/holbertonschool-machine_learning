#!/usr/bin/env python3
""" Function that builds a GAN

Requires:
    - tensorflow
"""
from tensorflow import keras

layers, models = keras.layers, keras.models


def convolutional_GenDiscr():
    """ Builds a Convolutional GAN """

    def get_generator():
        """ Builds the generator model """
        input_layer = layers.Input(shape=(16,), name='input_1')

        x = layers.Dense(2048, name='dense')(input_layer)
        x = layers.Reshape((2, 2, 512), name='reshape')(x)

        x = layers.UpSampling2D(name='up_sampling2d')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', name='conv2d')(x)
        x = layers.BatchNormalization(name='batch_normalization')(x)
        x = layers.Activation('relu', name='activation_1')(x)

        x = layers.UpSampling2D(name='up_sampling2d_1')(x)
        x = layers.Conv2D(16, (3, 3), padding='same', name='conv2d_1')(x)
        x = layers.BatchNormalization(name='batch_normalization_1')(x)
        x = layers.Activation('relu', name='activation_2')(x)

        x = layers.UpSampling2D(name='up_sampling2d_2')(x)
        x = layers.Conv2D(1, (3, 3), padding='same', name='conv2d_2')(x)
        x = layers.BatchNormalization(name='batch_normalization_2')(x)
        output_layer = layers.Activation('tanh', name='activation_3')(x)

        return models.Model(input_layer, output_layer, name='generator')

    def get_discriminator():
        """ Builds the discriminator model """
        inpt = layers.Input(shape=(16, 16, 1), name='input_2')

        x = layers.Conv2D(32, (3, 3), padding='same', name='conv2d_3')(inpt)
        x = layers.MaxPooling2D(name='max_pooling2d')(x)
        x = layers.Activation('relu', name='activation_4')(x)

        x = layers.Conv2D(64, (3, 3), padding='same', name='conv2d_4')(x)
        x = layers.MaxPooling2D(name='max_pooling2d_1')(x)
        x = layers.Activation('relu', name='activation_5')(x)

        x = layers.Conv2D(128, (3, 3), padding='same', name='conv2d_5')(x)
        x = layers.MaxPooling2D(name='max_pooling2d_2')(x)
        x = layers.Activation('relu', name='activation_6')(x)

        x = layers.Conv2D(256, (3, 3), padding='same', name='conv2d_6')(x)
        x = layers.MaxPooling2D(name='max_pooling2d_3')(x)
        x = layers.Activation('relu', name='activation_7')(x)

        x = layers.Flatten(name='flatten')(x)
        output_layer = layers.Dense(1, name='dense_1')(x)

        return models.Model(inpt, output_layer, name='discriminator')

    return get_generator(), get_discriminator()
