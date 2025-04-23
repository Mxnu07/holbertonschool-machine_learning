#!/usr/bin/env python3
"""This modlue contains a function that creates a variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """This function creates a variational autoencoder
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for
        each hidden layer
        latent_dims: integer containing the dimensions of the latent space
                     representation
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Step 1: Define the encoder model
    # Create the input layer for the encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"input_dims: {input_dims}\n\nencoder_inputs: {encoder_inputs}")
    # print(f"---" * 20)

    # Create the encoder layers
    for idx, units in enumerate(hidden_layers):
        # Add dense layers with the relu activation function
        layer = keras.layers.Dense(units=units, activation="relu")
        # Print for data visualization
        # print(f"---" * 20)
        # print(f"layer: {layer}")
        # print(f"---" * 20)
        if idx == 0:
            # If it is the first layer, set the input
            outputs = layer(encoder_inputs)
            # Print for data visualization
            # print(f"---" * 20)
            # print(f"outputs encoder: {outputs.shape}")
            # print(f"---" * 20)
        else:
            # If it is not the first layer, set the
            # output of the previous layer
            outputs = layer(outputs)
        # Print for data visualization
        # print(f"---" * 20)
        # print(f"outputs encoder: {outputs.shape}")
        # print(f"---" * 20)
    # crerate a mean layer
    layer = keras.layers.Dense(units=latent_dims)
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"layer encoder: {layer}")
    # print(f"---" * 20)
    # Create the mean layer
    mean = layer(outputs)
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"mean encoder: {mean.shape}")
    # print(f"---" * 20)
    layer = keras.layers.Dense(units=latent_dims)
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"layer encoder 2: {layer}")
    # print(f"---" * 20)
    log_variation = layer(outputs)
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"log_variation encoder: {log_variation.shape}")
    # print(f"---" * 20)

    # Create a sampling function to sample from the mean and log variation
    def sampling(args):
        """This function samples from the mean and log variation
        Args:
            args: list containing the mean and log variation
        Returns: sampled tensor
        """
        # Get the mean and log variation from the arguments
        mean, log_variation = args
        # print(f"---" * 20)
        # print(f"mean: {mean.shape}")
        # print(f"log_variation sampling: {log_variation.shape}")
        # print(f"---" * 20)
        # Generate a random tensor
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        # print(f"---" * 20)
        # print(f"epsilon sampling: {epsilon.shape}")
        # print(f"---" * 20)
        # Return the sampled tensor
        return mean + keras.backend.exp(log_variation * 0.5) * epsilon

    # Use a keras layer to wrap the sampling function
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mean, log_variation]
    )
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"layer after sampling: {layer}")
    # print(f"---" * 20)
    # Create the encoder model
    encoder = keras.models.Model(
        inputs=encoder_inputs, outputs=[z, mean, log_variation]
    )
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"encoder summary: {encoder.summary()}")
    # print(f"---" * 20)

    # Define the decoder model
    decoder_inputs = keras.Input(shape=(latent_dims,))
    for idx, units in enumerate(reversed(hidden_layers)):
        # Create a Dense layer with relu activation
        layer = keras.layers.Dense(units=units, activation="relu")
        # Print for data visualization
        # print(f"---" * 20)
        # print(f"layer decoder: {layer}")
        # print(f"---" * 20)
        if idx == 0:
            # if it is the first layer, set the input
            outputs = layer(decoder_inputs)
            # Print for data visualization
            # print(f"---" * 20)
            # print(f"outputs decoder: {outputs}")
            # print(f"---" * 20)
        else:
            outputs = layer(outputs)
            # Print for data visualization
            # print(f"---" * 20)
            # print(f"outputs decoder: {outputs}")
            # print(f"---" * 20)
    # Create the output layer for the decoder modle
    # using sigmoid activation function
    layer = keras.layers.Dense(units=input_dims, activation="sigmoid")
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"layer decoder: {layer}")
    # print(f"---" * 20)
    outputs = layer(outputs)
    # Print for data visualization
    # print(f"---" * 20)
    # print(f"outputs decoder: {outputs}")
    # print(f"---" * 20)
    # Create the decoder model
    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Create the full autoencoder model
    outputs = encoder(encoder_inputs)
    # print(f"---" * 20)
    # print(f"outputs encoder: {outputs}")
    # print(f"---" * 20)
    outputs = decoder(outputs[0])
    # print(f"---" * 20)
    # print(f"outputs decoder: {outputs}")
    # print(f"---" * 20)
    auto = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    # print the modle summaries
    # print(f"---" * 20)
    # print(f"encoder summary: {encoder.summary()}")
    # print(f"---" * 20)
    # print(f"decoder summary: {decoder.summary()}")
    # print(f"---" * 20)
    # print(f"auto summary: {auto.summary()}")
    # print(f"---" * 20)

    # compute the loss of the modle using the binary crossentropy
    # def vae_loss(inputs, outputs):
    #     """This function returns the loss of the model
    #     Args:
    #         x: the input tensor
    #         x_decoded_mean: the output tensor
    #     Returns: loss
    #     """
    #     xent_loss = keras.backend.binary_crossentropy(inputs, outputs)
    #     xent_loss = keras.backend.sum(xent_loss, axis=1)
    #     kl_loss = -0.5 * keras.backend.sum(
    #         1
    #         + log_variation
    #         - keras.backend.square(mean)
    #         - keras.backend.exp(log_variation),
    #         axis=-1,
    #     )
    #     return xent_loss + kl_loss

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
