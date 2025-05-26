#!/usr/bin/env python3
"""This module has the method gensim_to_keras"""
import tensorflow as tf


def gensim_to_keras(model):
    """This method converts a gensim word2vec model to a
    keras Embedding layer
    Args:
        model is a trained gensim word2vec models
    Returns:
        the trainable keras Embedding
    """
    # Extract the word vectors from the model
    weights = model.wv.vectors
    # crerate the keras embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
    return embedding_layer
