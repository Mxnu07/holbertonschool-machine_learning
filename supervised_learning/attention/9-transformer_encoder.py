#!/usr/bin/env python3
"""This module conatins the Encoder class
that inherits from tensorflow.keras.layers.Layer"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """This class creates an encoder for a transformer"""
    def __init__(
            self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """Class constructor
            Args:
                N (int): the number of blocks in the encoder
                dm (int): the dimensionality of the model
                h (int): the number of heads
                hidden (int): the number of hidden units in the fully
                connected layer
                input_vocab (int): the size of the input vocabulary
                max_seq_len (int): the maximum sequence length possible
                drop_rate (float): the dropout rate
        """
        # call the parent class constructor
        super(Encoder, self).__init__()

        # set the number of blocks
        self.N = N
        # Set the dimensionality of the model
        self.dm = dm
        # set the embedding layer
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        # set the positional encoding layer
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # set the list of encoder blocks
        self.blocks = [EncoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        # set the dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """This method calls the encoder
            Args:
                x (tf.Tensor): contains the input to the encoder
                training (bool): determines if the model is in training
                mask (tf.Tensor): contains the mask to be applied for
                multihead attention
            Returns:
                (tf.Tensor): contains the encoder's output
        """
        seq_len = x.shape[1]

        # apply the embedding layer
        x = self.embedding(x)
        # scale the embedding by the square root of the dimension
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # add the positional encoding
        x += self.positional_encoding[:seq_len]

        # apply the dropout layer
        x = self.dropout(x, training=training)

        # call each block in the encoder
        for block in self.blocks:
            x = block(x, training, mask)

        return x
