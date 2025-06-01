#!/usr/bin/env python3
"""This module contains the Decoder block class
    that inherits from tensorflow.keras.layers.Layer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """This class creates a decoder block for a transformer"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor
            Args:
                dm (int): the dimensionality of the model
                h (int): the number of heads
                hidden (int): the number of hidden units in the fully
                connected layer
                drop_rate (float): the dropout rate
        """
        # call the parent class constructor
        super(DecoderBlock, self).__init__()

        # set the first multi head attention layer
        self.mha1 = MultiHeadAttention(dm, h)
        # set the second multi head attention layer
        self.mha2 = MultiHeadAttention(dm, h)
        # set the dense hidden layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # set the dense output layer with dm units
        self.dense_output = tf.keras.layers.Dense(dm)
        # set the layer normalization layer
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # set the layer normalization layer
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # set the layer normalization layer
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # set the dropout layer
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # set the dropout layer
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        # set the dropout layer
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    # def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
    #     """This method calls the decoder block
    #         Args:
    #             x (tf.Tensor): contains the input to the decoder block
    #             encoder_output (tf.Tensor): contains the output of the encoder
    #             training (bool): determines if the model is in training
    #             look_ahead_mask (tf.Tensor): contains the mask to be applied to
    #             the first multi head attention layer
    #             padding_mask (tf.Tensor): contains the mask to be applied to
    #             the second multi head attention layer
    #         Returns:
    #             (tf.Tensor): contains the block's output
    #     """
    #     # call the first multi head attention layer
    #     attn1, _ = self.mha1(x, x, x, look_ahead_mask)
    #     # apply dropout layer
    #     attn1 = self.dropout1(attn1, training=training)
    #     # apply layer normalization
    #     out1 = self.layernorm1(x + attn1)
    #     # call the second multi head attention layer
    #     attn2, _ = self.mha2(
    #         out1, encoder_output, encoder_output, padding_mask)
    #     # apply dropout layer
    #     attn2 = self.dropout2(attn2, training=training)
    #     # apply layer normalization
    #     out2 = self.layernorm2(out1 + attn2)
    #     # call the dense hidden layer
    #     ffn_output = self.dense_hidden(out2)
    #     # call the dense output layer
    #     ffn_output = self.dense_output(ffn_output)
    #     # apply dropout layer
    #     ffn_output = self.dropout3(ffn_output, training=training)
    #     # apply layer normalization
    #     out3 = self.layernorm3(out2 + ffn_output)
    #     return out3

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the transformer's decoder block.

        :param x: a tensor of shape `(batch, target_seq_len, dm)` containing
        the input to the decoder block
        :param encoder_output: a tensor of shape `(batch, input_seq_len, dm)`
        containing the output of the encoder
        :param training: a boolean to determine if the model is training
        :param look_ahead_mask: the mask to be applied to the first multi head
        attention layer
        :param padding_mask: the mask to be applied to the second multi head
        attention layer

        Returns:
        A tensor of shape `(batch, target_seq_len, dm)` containing the block's
        output
        """
        # Masked Multi-head attention
        masked_mha_output, _ = self.mha1(x, x, x, look_ahead_mask)
        # 1st dropout
        masked_mha_output = self.dropout1(masked_mha_output, training=training)
        # 1st residual connection + layer normalization
        output1 = self.layernorm1(x + masked_mha_output)

        # Second multi-head attention
        mha2_output, _ = self.mha2(output1, encoder_output, encoder_output,
                                   padding_mask)
        mha2_output = self.dropout2(mha2_output)

        # 2nd residual connection + layer normalization
        output2 = self.layernorm2(mha2_output + output1)

        # Feed-forward neural network: 1st dense layer with ReLU activation
        ff_output = self.dense_hidden(output2)
        # Second dense layer
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout3(ff_output, training=training)

        # 2nd Residual connection + layer normalization
        output2 = self.layernorm3(ff_output + output2)

        return output2
