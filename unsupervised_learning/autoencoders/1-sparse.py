#!/usr/bin/env python3
"""This modlue contains a sparse autoencoder function"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """This function creates a sparse autoencoder
    Args:
        input_dim: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
                        hidden layer
        latent_dim: integer containing the dimensions of the latent space
                        representation
        lambtha: the regularization parameter used for L1 regularization
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Step 1: Define the encoder model
    # create the input layer for the encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    # print for data visualization
    # print(f"---" * 20)
    # print(f"input_dims: {input_dims}\n\nencodes_inputs: {encoder_inputs}")
    # print(f"---" * 20)
    # print hidden layers for data visualization
    # print(f"---" * 20)
    # print(f"hidden_layers: {hidden_layers}")
    # print(f"---" * 20)
    for idx, units in enumerate(hidden_layers):
        # Add dense layers with the relu activation function
        layer = keras.layers.Dense(units=units, activation="relu")
        # print(f"---" * 20)
        # print(f"layer: {layer}")
        # print(f"---" * 20)
        if idx == 0:
            # if it is the first layer, set the input
            outputs = layer(encoder_inputs)
            # print(f"---" * 20)
            # print(f"outputs: {outputs}")
            # print(f"---" * 20)
        else:
            # if it is not the first layer, set the output of the
            # previous layer
            outputs = layer(outputs)
        # print(f"---" * 20)
        # print(f"output: {outputs}")
        # print(f"---" * 20)
    # Add the latent layer with L1 regularization to enforce sparsity
    latent = keras.layers.Dense(
        units=latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha),
    )
    # print(f"---" * 20)
    # print(f"latent: {latent}")
    # print(f"---" * 20)
    # make the latent layer the output layer for the encoder
    latent = latent(outputs)
    # print(f"---" * 20)
    # print(f"latent after making it teh output for the encoder: {latent}")
    # print(f"---" * 20)
    # create the encoder model with impust from the input layer and output
    # from the latent layer
    encoder = keras.models.Model(inputs=encoder_inputs, outputs=latent)
    # print(f"---" * 20)
    # print(f"encoder: {encoder}")
    # print(f"---" * 20)

    # Step 2: Define the decoder model
    # create the input layer for the decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    # prints for data visualization
    # print(f"---" * 20)
    # print(f"decoder_inputs: {decoder_inputs}\n\nlatent_dims: {latent_dims}")
    # print(f"---" * 20)
    for idx, units in enumerate(reversed(hidden_layers)):
        # Add dense layers with the relu activation function
        layer = keras.layers.Dense(units=units, activation="relu")
        # print(f"---" * 20)
        # print(f"layer in the decoder: {layer}")
        # print(f"---" * 20)
        if idx == 0:
            # if it is the first layer, set the input
            outputs = layer(decoder_inputs)
            # print(f"---" * 20)
            # print(f"outputs in the decoder: {outputs}")
            # print(f"---" * 20)
        else:
            # if it is not the first layer, set the output of
            # the previous layer
            outputs = layer(outputs)
            # print(f"---" * 20)
            # print(f"output in the decoder: {outputs}")
            # print(f"---" * 20)
    # create the output layer with the sigmoid activation function
    # this is the decoded output
    layer = keras.layers.Dense(units=input_dims, activation="sigmoid")
    # print(f"---" * 20)
    # print(f"layer in the decoder: {layer}")
    # print(f"---" * 20)
    # make the output layer the output of the decoder
    outputs = layer(outputs)
    # print(f"---" * 20)
    # print(f"outputs in the decoder: {outputs}")
    # print(f"---" * 20)

    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)
    # print(f"---" * 20)
    # print(f"decoder: {decoder}")
    # print(f"---" * 20)

    # Step 3: Create the full autoencoder model
    # call the encoder with the input layer to get the latent representation
    outputs = encoder(encoder_inputs)
    # print(f"---" * 20)
    # print(f"outputs in the autoencoder: {outputs}")
    # print(f"---" * 20)
    # call the decoder with the latent representation to get the decoded output
    decoded = decoder(outputs)
    # print(f"---" * 20)
    # print(f"decoded in the autoencoder: {decoded}")
    # print(f"---" * 20)
    # create the autoencoder model with the input layer as input and the
    auto = keras.models.Model(inputs=encoder_inputs, outputs=decoded)

    # Compile the autoencoder
    auto.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, auto
