#!/usr/bin/env python3
"""This module contains the sel;f atttention class"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """This class creates a Self Attention layer
        and inherits from tf.keras.layers.Layer"""
    def __init__(self, units):
        """"Self Attention constructor
            Args:
                units (int): represents the number of hidden units in the
                    fully connected layer
        """
        super(SelfAttention, self).__init__()
        # A dense layer to be apliyed to the prev decoder
        # hidden states
        self.W = tf.keras.layers.Dense(units)

        # a Dense l;ayer to be apliyed to the encoder hidden states
        self.U = tf.keras.layers.Dense(units)

        # A Dense layers with ! unit to be apliyed to the tanh of the sum
        # of the outputs of W and U
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """This method builds the Self Attention layer
            Args:
                s_prev (tf.Tensor): a tensor of shape (batch, units)
                    containing the previous decoder hidden state
                hidden_states (tf.Tensor): a tensor of shape (batch,
                input_seq_len, units)
                    containing the outputs of the encoder
            Returns:
                context (tf.Tensor): a tensor of shape (batch, units)
                    that contains the context vector for the decoder
                weights (tf.Tensor): a tensor of shape (batch,
                input_seq_len, 1)
                    that contains the attention weights
        """
        # Expand the dimensions of s_prev to shape (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Apply the dense layer to the hidden states to create
        # a tensor of shape (batch, input_seq_len, units)
        W_hidden_states = self.W(s_prev_expanded)

        # Apply the dense layer to s_prev_expanded to create
        # a tensor of shape (batch, 1, units)
        U_s_prev = self.U(hidden_states)

        # Add the two tensors above to create a tensor of shape
        # (batch, input_seq_len, units)
        tanh = tf.nn.tanh(W_hidden_states + U_s_prev)

        # Apply the dense layer to tanh to create a tensor of shape
        # (batch, input_seq_len, 1)
        scores = self.V(tanh)

        # Calculate the attention weights
        weights = tf.nn.softmax(scores, axis=1)

        # Calculate the context vector
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
