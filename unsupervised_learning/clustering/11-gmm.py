#!/usr/bin/env python3
"""This modlue contains the GMM function"""
import sklearn.mixture


def gmm(X, k):
    """This function calculates the expectation maximization fr a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
    Returns:
        pi, m, S, clss, bic or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the priors fr each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means fr each
        cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        fr each cluster
        clss: numpy.ndarray of shape (n, ) containing the clusters
        fr each data point in each cluster
        bic: BIC value fr each cluster size tested
    """

    # Step 1: Perform the GMM algorithm
    Gmm = sklearn.mixture.GaussianMixture(n_components=k)
    # Step 2: Fit the model to the data
    parameters = Gmm.fit(X)
    # Step 3: Extract the weigth
    pi = parameters.weights_
    # Step 4: Extract the mean
    m = parameters.means_
    # Step 5: Extract the covariance matrix
    S = parameters.covariances_
    # Step 6: Extract the prior probabilities
    clss = Gmm.predict(X)
    # Step 7: get the beyesian information criterion
    bic = Gmm.bic(X)

    # Step 8: Return the priors, means, covariances, classes, and BIC
    return pi, m, S, clss, bic
