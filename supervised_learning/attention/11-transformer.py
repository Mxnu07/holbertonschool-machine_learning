#!/usr/bin/env python3
"""
Full Transformer Network
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    This class represents a complete transformer network.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initializes the Transformer model.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Forward pass through the Transformer network.
        """
        # Pass inputs through the encoder
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass the target and encoder output through the decoder
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # Pass the decoder output through the final linear layer
        final_output = self.linear(dec_output)

        return final_output
