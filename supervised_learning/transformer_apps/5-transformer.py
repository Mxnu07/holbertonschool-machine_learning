#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer using TensorFlow.
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    angle_rads = pos * angle_rates
    # Apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale the matmul_qk by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax on the last axis (seq_len_k) to get attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights with the values (V)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer.
    """

    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.dense = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth).
        Transpose the result to shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        K = self.Wk(K)  # (batch_size, seq_len_k, dm)
        V = self.Wv(V)  # (batch_size, seq_len_v, dm)

        Q = self.split_heads(Q, batch_size)  # (batch_size, h, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch_size, h, seq_len_k, depth)
        V = self.split_heads(V, batch_size)  # (batch_size, h, seq_len_v, depth)

        # Apply scaled dot product attention
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, h, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))  # (batch_size, seq_len_q, dm)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, attention_weights

class EncoderBlock(tf.keras.layers.Layer):
    """
    Transformer encoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),  # (batch_size, input_seq_len, hidden)
            tf.keras.layers.Dense(dm)  # (batch_size, input_seq_len, dm)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, dm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, dm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection

        return out2

class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),  # (batch_size, target_seq_len, hidden)
            tf.keras.layers.Dense(dm)  # (batch_size, target_seq_len, dm)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # Masked mha
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)  # (batch_size, target_seq_len, dm)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, dm)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3

class Transformer(tf.keras.Model):
    """
    Transformer model combining the encoder and decoder.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        enc_output = self.encoder(inputs, training, encoder_mask)  # (batch_size, input_seq_len, dm)

        dec_output = self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)  # (batch_size, target_seq_len, dm)

        final_output = self.final_layer(dec_output)  # (batch_size, target_seq_len, target_vocab)

        return final_output

class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder consisting of multiple Encoder Blocks.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.encoder_blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        # Apply dropout to the positional encoding
        x = self.dropout(x, training=training)

        # Pass through the stack of Encoder Blocks
        for i in range(self.N):
            x = self.encoder_blocks[i](x, training, mask)

        return x  # (batch_size, input_seq_len, dm)


class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder consisting of multiple Decoder Blocks.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.decoder_blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch_size, target_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        # Apply dropout to the positional encoding
        x = self.dropout(x, training=training)

        # Pass through the stack of Decoder Blocks
        for i in range(self.N):
            x = self.decoder_blocks[i](x, enc_output, training, look_ahead_mask, padding_mask)

        return x  # (batch_size, target_seq_len, dm)
