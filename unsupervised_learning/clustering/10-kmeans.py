#!/usr/bin/env python3
"""This modlue preforms the k means on a data set
"""
import sklearn.cluster


def kmeans(X, k):
    """This functions preforms the k means on a data set
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
    Returns:
        C, clss or None, None on failure
        C: numpy.ndarray of shape (k, d) containing the centroid means fr each
        cluster
        clss: numpy.ndarray of shape (n,) containing the index of the
            cluster in
        C that each data point belongs to
    """
    # Step 1: Perform the kmeans algorithm
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    # Step 2: Return the centroids and the clusters
    return C, clss
