#!/usr/bin/env python3
"""Thios module contains a function that creates a TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """This function creates a TF-IDF embedding
    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
    Return:
        embeddings: is a numpy.ndarray of shape (s, f) containing the
        embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features: is a listr of the features used for embeddings
    """
    # Initialize the TfidfVectorizer
    # - TfidfVectorizer: Converts a collection of raw documents
    # to a matrix of TF-IDF features.
    # - vocabulary: If provided, it will use this list of words as the
    # vocabulary; otherwise, it will determine the vocabulary from the
    # input sentences.
    vectorize = TfidfVectorizer(vocabulary=vocab)

    # Create the matrix of TF-IDF features
    # - fit_transform: Learns the vocabulary and idf (inverse document
    # frequency)
    # from the input sentences and returns the term-document matrix.
    # - sentences: The input sentences to be transformed into the
    # TF-IDF matrix.
    matrix = vectorize.fit_transform(sentences)

    # Transform the matrix into an array
    # - toarray: Converts the sparse matrix to a dense numpy array.
    # - embeddings: The resulting dense array where each row
    # represents a sentence
    # and each column represents a word from the vocabulary, with the values
    # being the TF-IDF scores.
    embeddings = matrix.toarray()

    # Get the feature names
    # - get_feature_names_out: Retrieves the feature names (unique words)
    # from the vectorizer.
    # - features: The list of feature names corresponding to the columns
    # in the TF-IDF matrix.
    features = vectorize.get_feature_names_out()

    # Return the embeddings and the feature names
    # - embeddings: The dense numpy array of TF-IDF scores.
    # - features: The list of feature names (unique words).
    return embeddings, features
