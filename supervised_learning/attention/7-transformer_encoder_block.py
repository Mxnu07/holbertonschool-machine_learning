#!/usr/bin/env python3
"""This module contains the Encoder block class
    that inherits from tensorflow.keras.layers.Layer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """This class creates an encoder block for a transformer"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor
            Args:
                dm (int): the dimensionality of the model
                h (int): the number of heads
                hidden (int): the number of hidden units in the fully
                connected layer
                drop_rate (float): the dropout rate
        """
        super(EncoderBlock, self).__init__()
        # set the multi head layer
        self.mha = MultiHeadAttention(dm, h)
        # set the dense hidden layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # set the dense output layer with dm units
        self.dense_output = tf.keras.layers.Dense(dm)
        # set the layer normalization layer
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # set the layer normalization layer
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # set the dropout layer
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # set the dropout layer
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """This method calls the encoder block
            Args:
                x (tf.Tensor): contains the input to the encoder block
                training (bool): determines if the model is in training
                mask (tf.Tensor): contains the mask to be applied for
                multihead attention
            Returns:
                (tf.Tensor): contains the block's output
        """
        # call the multi head attention layer
        attn_output, _ = self.mha(x, x, x, mask)
        # apply dropout layer
        attn_output = self.dropout1(attn_output, training=training)
        # apply layer normalization
        out1 = self.layernorm1(x + attn_output)
        # call the dense hidden layer
        ffn_output = self.dense_hidden(out1)
        # call the dense output layer
        ffn_output = self.dense_output(ffn_output)
        # apply dropout layer
        ffn_output = self.dropout2(ffn_output, training=training)
        # apply layer normalization
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
