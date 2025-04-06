#!/usr/bin/env python3
"""This modlue contains the function agglomerative
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """This function performs agglomerative clustering on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters
    Returns:
        clss, a numpy.ndarray of shape (n,) containing the cluster indices
    """
    # Step 1: Perform the agglomerative clustering
    hierarchy = scipy.cluster.hierarchy
    # Step 2: Perform the linkage
    links = hierarchy.linkage(X, method='ward')
    # Step 3: Perform the clustering
    clss = hierarchy.fcluster(links, t=dist, criterion='distance')

    # Step 4: Plot the dendrogram
    plt.figure()
    # use dendrogram to plot the tree
    hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()

    # Step 5: Return the cluster indices
    return clss
