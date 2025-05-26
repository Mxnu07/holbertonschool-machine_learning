#!/usr/bin/env python3
"""This module contain the BOW funtion"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """This function creates a bag of words embedding matrix
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
    # Initialize the CountVectorizer
    # If vocab is provided, it will use that vocabulary; otherwise,
    # it will determine the vocabulary from the input sentences
    vectorizer = CountVectorizer(vocabulary=vocab)

    # Fit the model and transform the sentences into a BOW representation
    # X is a sparse matrix of shape (n_samples, n_features)
    X = vectorizer.fit_transform(sentences)

    # Convert the sparse matrix to a dense numpy array
    embeddings = X.toarray()

    # Get the feature names (unique words) from the vectorizer
    features = vectorizer.get_feature_names_out()

    # Return the embeddings and the feature names
    return embeddings, features
