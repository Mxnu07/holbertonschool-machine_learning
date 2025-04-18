#!/usr/bin/env python3
"""This modlue contains the class GaussianProcess"""
import numpy as np


class GaussianProcess:
    """This class represents a noisless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """This method initialized an oject of the class GaussianProcess
        Args:
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
            t: the number of initial samples
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                    black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """This method calculates the covariance kernel matrix between two
        matrices
        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)
        Returns: the covariance kernel matrix as a numpy.ndarray of shape
                (m, n)
        """
        # Step 1: Calculate the squared Euclidean distance between the
        #         rows of X1 and X2
        # np.sum(X1**2, 1) returns a column vector of the sum of the squares
        #                   of the elements of X1
        # np.sum(X2**2, 1) returns a column vector of the sum of the squares
        #                   of the elements of X2
        # np.dot(X1, X2.T) returns a matrix of the dot product of X1 and the
        #                   transpose of X2
        # .reshape(-1, 1) reshapes the column vector to a column vector of
        #                 the same length
        sqdist = np.sum(X1**2, 1).reshape(
            -1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

        # Step 2: Calculate the covariance kernel matrix
        # self.sigma_f**2 is the variance of the black-box function
        # self.l**2 is the squared length parameter of the kernel
        # np.exp(-0.5 / self.l**2 * sqdist) is
        #               the exponential term of the kernel
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
