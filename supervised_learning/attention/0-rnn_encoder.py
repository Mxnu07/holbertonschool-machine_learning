#!/usr/bin/env python3
"""This modlue contains the RNnENcoder class"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """This class represents an encoder for a transformer
        and inherits from tf.keras.layers.Layer"""

    def __init__(self, vocab, embedding, units, batch):
        """RNNEncoder constructor

        Args:
            vocab (int): represents the size of the input vocabulary
            embedding (int): represents the dimensionality of the embedding
                vector
            units (int): represents the number of hidden units in the RNN cell
            batch (int): represents the batch size
        """
        super(RNNEncoder, self).__init__()
        # Embedding layer to convert input tokens to dense vectors
        # of fixed size
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # GRU layer to process the embedded input sequence
        # return_sequences=True: return the full sequence of outputs for
        # each input sequence
        # return_state=True: return the last state in addition to the output
        # recurrent_initializer='glorot_uniform': initializer for the
        # recurrent kernel weights
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.batch = batch
        self.units = units

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns:
            tf.Tensor: a tensor of shape (self.batch, self.units) containing
                the initialized hidden states
        """
        return tf.zeros((self.batch, self.gru.units))

    def call(self, x, initial):
        """This method builds the encoder

        Args:
            x (tf.Tensor): a tensor of shape (batch, input_seq_len) containing
                the input to the encoder
            initial (tf.Tensor): a tensor of shape (batch, units) containing
                the initial hidden state

        Returns:
            tf.Tensor, tf.Tensor: the outputs of the encoder and the last
                hidden state
        """
        # Pass the input through the embedding layer to
        # convert tokens to dense vectors
        x = self.embedding(x)

        # Pass the embedded input through the GRU layer
        # 'outputs' contains the full sequence of outputs
        # for each input sequence
        # 'hidden' contains the last hidden state of the GRU
        outputs, hidden = self.gru(x, initial)
        return outputs, hidden
