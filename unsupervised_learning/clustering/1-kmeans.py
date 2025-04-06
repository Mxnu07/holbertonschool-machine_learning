#!/usr/bin/env python3
"""This module initializes a cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """This function initializes cluster centroids fr K-means
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
        used for K-means clustering
            - n is the number of data points
            - d is the number of dimensions fr each data point
        k: positive integer containing the number of clusters
        Returns:
            numpy.ndarray of shape (k, d) containing the initialized
            centroids fr each cluster, or None on failure
    """
    # Check fr input data
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Extract the shape of the dataset
    n, d = X.shape
    # XCheck fr inpuit data k
    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    # Generate k initial centroids by sampling from a uniform
    # distribution
    # The uniform distribution is defined by the minimum and
    # maximum values
    # of the dataset X along each dimension

    # np.min(X, axis=0) computes the minimum value fr each
    # feature (column) in X
    # np.max(X, axis=0) computes the maximum value fr each
    # feature (column) in X
    # These min and max values define the range fr the uniform
    # distribution

    # np.random.uniform generates random numbers from a uniform
    # distribution
    # with the specified low and high bounds fr each feature
    # size=(k, d) specifies that we want to generate k samples,
    # each with d features
    centroids = np.random.uniform(
        low=np.min(X, axis=0), high=np.max(X, axis=0), size=(k, d)
    )
    return centroids


def kmeans(X, k, iterations=1000):
    """This function performs K-means on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
        used for K-means clustering
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
        iterations that should be performed
        Returns:
            C: numpy.ndarray of shape (k, d) containing the final centroids for
            each cluster
            clss: numpy.ndarray of shape (n,) containing the index of the
            cluster in
            C that each data point belongs to
    """
    # Step 1: Initialize the centroids
    centroids = initialize(X, k)

    # Step 1.5: Check if centroids is None and verify iterations
    if centroids is None:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Step 2: Extract the shapes
    n, d = X.shape

    # Step 3: Iterate over the number of iterations
    for iteration in range(iterations):
        # Keep a copy of the prev centroids
        prev_centroids = np.copy(centroids)

        # Step 4: apply vectorization to calculate
        # the squared Euclidean distance
        # between each data point and each centroid

        # convert X into a vector
        X_vector = np.repeat(X, k, axis=0)

        # reshape X_vector
        # n: the shap of the first dimention of the new shape
        # k: the shape of the second dimention of the new shape
        # d: the shape of the third dimention of the new shape
        X_vector = X_vector.reshape(n, k, d)

        # convert centroids into a vector
        # We use the np.tile function to repeat the centroids
        centroids_vector = np.tile(centroids, (n, 1))

        # reshape centroids_vector
        # n: the shap of the first dimention of the new shape
        # k: the shape of the second dimention of the new shape
        # d: the shape of the third dimention of the new shape
        centroids_vector = centroids_vector.reshape(n, k, d)

        # Step 5: Compute the Euclidean distance
        # between each data point and each centroid
        # X_vector - centroids_vector computes the difference
        # between each data point and each centroid
        # teh resulting shape is (n, k, d)
        # np.linalg.norm computes the Euclidean distance
        # axis=2 especifies the third dimention
        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)

        # Step 6: Assign each data point to the closest centroid
        # based on the computed distance
        # np.argmin returns the index of the minimum value along an axis
        clss = np.argmin(distance ** 2, axis=1)

        # Step 7: Update the centroids
        # Move the centroids to the center of their respective cluster
        for i in range(k):

            indixes = np.where(clss == i)[0]
            if len(indixes) == 0:
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = np.mean(X[indixes], axis=0)

        # Step 8: Check if the centroids have converged
        if np.all(prev_centroids == centroids):
            return centroids, clss

        # Step 9: If the centroids have not converged, repeat
        # the process
        centroids_vector = np.tile(centroids, (n, 1))
        centroids_vector = centroids_vector.reshape(n, k, d)
        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)
        clss = np.argmin(distance ** 2, axis=1)

    return centroids, clss
