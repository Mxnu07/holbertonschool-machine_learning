#!/usr/bin/env python3
"""This modlue contains a vinilla autoencoder function"""
import tensorflow.keras as keras


def autoencoder(input_dim, hidden_layers, laten_dims):
    """This function creates a vanilla autoencoder
    Args:
        input_dim: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
                        hidden layer
        latent_dims: integer containing the dimensions of the latent space
                        representation
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Step 1: Create the encoder model
    # input_encoder is the input to the encoder
    input_encoder = keras.layers.Input(shape=(input_dim,))
    # input_encoded is the input to the next layer
    input_encoded = input_encoder
    # iterate through the hidden layers
    for layer in hidden_layers:
        # create a dense layer with relu activation
        input_encoded = keras.layers.Dense(
            layer, activation='relu')(input_encoded)
    # create the latent space representation with relu activation
    latent = keras.layers.Dense(laten_dims, activation='relu')(input_encoded)
    # define the encoder model with input_encoder as input and latent as output
    encoder = keras.models.Model(input_encoder, latent)

    # Step 2: Create the decoder model
    # input_decoder is the input to the decoder
    input_decoder = keras.layers.Input(shape=(laten_dims,))
    # input_decoded is the input to the next layer
    input_decoded = input_decoder
    # iterate through the hidden layers in reverse order
    for layer in hidden_layers[::-1]:
        # create a dense layer with relu activation
        input_decoded = keras.layers.Dense(
            layer, activation='relu')(input_decoded)
    # create the output layer with sigmoid activation to reconstruct the input
    decoded = keras.layers.Dense(
        input_dim, activation='sigmoid')(input_decoded)
    # define the decoder model with input_decoder as
    # input and decoded as output
    decoder = keras.models.Model(input_decoder, decoded)

    # Step 3: Create the full autoencoder model
    # input_auto is the input to the autoencoder
    input_auto = keras.layers.Input(shape=(input_dim,))
    # pass the input through the encoder to get the latent representation
    encoder_out = encoder(input_auto)
    # pass the latent representation through the decoder to
    # reconstruct the input
    decoder_out = decoder(encoder_out)
    # define the autoencoder model with input_auto as input and
    # decoder_out as output
    auto = keras.models.Model(input_auto, decoder_out)

    # compile the autoencoder model with adam optimizer and binary
    # crossentropy loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    # return the encoder, decoder, and autoencoder models
    return encoder, decoder, auto
