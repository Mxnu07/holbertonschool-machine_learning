#!/usr/bin/env python3
"""This modlue contains the class GaussianProcess"""
import numpy as np


# class GaussianProcess:
#     """This class represents a noisless 1D Gaussian process"""
#     def __init__(self, X_init, Y_init, l=1, sigma_f=1):
#         """This method initialized an oject of the class GaussianProcess
#         Args:
#             X_init: numpy.ndarray of shape (t, 1) representing the inputs
#                     already sampled with the black-box function
#             Y_init: numpy.ndarray of shape (t, 1) representing the outputs
#                     of the black-box function for each input in X_init
#             t: the number of initial samples
#             l: the length parameter for the kernel
#             sigma_f: the standard deviation given to the output of the
#                     black-box function
#         """
#         self.X = X_init
#         self.Y = Y_init
#         self.l = l
#         self.sigma_f = sigma_f
#         self.K = self.kernel(X_init, X_init)

#     def kernel(self, X1, X2):
#         """This method calculates the covariance kernel matrix between two
#         matrices
#         Args:
#             X1: numpy.ndarray of shape (m, 1)
#             X2: numpy.ndarray of shape (n, 1)
#         Returns: the covariance kernel matrix as a numpy.ndarray of shape
#                 (m, n)
#         """
#         # Step 1: Calculate the squared Euclidean distance between the
#         #         rows of X1 and X2
#         # np.sum(X1**2, 1) returns a column vector of the sum of the squares
#         #                   of the elements of X1
#         # np.sum(X2**2, 1) returns a column vector of the sum of the squares
#         #                   of the elements of X2
#         # np.dot(X1, X2.T) returns a matrix of the dot product of X1 and the
#         #                   transpose of X2
#         # .reshape(-1, 1) reshapes the column vector to a column vector of
#         #                 the same length
#         sqdist = np.sum(X1**2, 1).reshape(
#             -1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

#         # Step 2: Calculate the covariance kernel matrix
#         # self.sigma_f**2 is the variance of the black-box function
#         # self.l**2 is the squared length parameter of the kernel
#         # np.exp(-0.5 / self.l**2 * sqdist) is
#         #               the exponential term of the kernel
#         return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

#     def predict(self, X_s):
#         """This method preedicts the mean and standard deviation of points
#         in a Gaussian process
#         Args:
#             X_s: numpy.ndarray of shape (s, 1) containing all of the points
#                  whose mean and standard deviation should be calculated
#         Returns: mu, sigma
#                  mu: numpy.ndarray of shape (s,) containing the mean for each
#                      point in X_s, respectively
#                  sigma: numpy.ndarray of shape (s,) containing the variance
#                         for each point in X_s, respectively
#         """
#         # Step 1: Calculate the covariance kernel matrix between X_s and X
#         #         and between X_s and X_s
#         K_s = self.kernel(self.X, X_s)
#         # print(f"K_s: {K_s}")
#         K_ss = self.kernel(X_s, X_s)
#         # print(f"K_ss: {K_ss}")
#         # Calculate the inverse of the covariance kernel matrix
#         # np.linalg.inv() calculates the inverse of a matrix
#         K_inv = np.linalg.inv(self.K)

#         # Step 2: Calculate the mean and standard deviation
#         # .T.dot is the dot product of the transpose of K_s and K_inv
#         # reshape(-1) reshapes the column vector to a row vector
#         mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
#         # print(f"mu: {mu}")
#         # K_ss - K_s.T.dot(K_inv).dot(K_s) calculates the diagonal of the
#         #       covariance kernel matrix between X_s and X_s
#         # np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s)) returns the diagonal
#         sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

#         return mu, sigma

#     def update(self, X_new, Y_new):
#         """This method updates a Gaussian process
#         Args:
#             X_new: numpy.ndarray of shape (1,) that represents the new sample
#                    point
#             Y_new: numpy.ndarray of shape (1,) that represents the new sample
#                    function value
#         """
#         # Step 1: Update the Gaussian process by concatenating the new sample
#         #         point and the new sample function value to X and Y,
#         #         respectively
#         self.X = np.append(self.X, X_new).reshape(-1, 1)
#         self.Y = np.append(self.Y, Y_new).reshape(-1, 1)

#         # Step 2: Update the covariance kernel matrix
#         self.K = self.kernel(self.X, self.X)
class GaussianProcess:
    """Class that instantiates a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """define and initialize variables and methods"""

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        function that calculates the covariance kernel matrix
        between two matrices
        """

        # Composition of the constant kernel with the
        # radial basis function (RBF) kernel, which encodes
        # for smoothness of functions (i.e. similarity of
        # inputs in space corresponds to the similarity of outputs)

        # Two hyperparameters: signal variance (sigma_f**2) and lengthscale l
        # K: Constant * RBF kernel function

        # Compute "dist_sq" (helper to K)
        # X1: shape (m, 1), m points of 1 coordinate
        # X2: shape (n, 1), n points of 1 coordinate
        a = np.sum(X1 ** 2, axis=1, keepdims=True)
        b = np.sum(X2 ** 2, axis=1, keepdims=True)
        c = np.matmul(X1, X2.T)
        # Note: Ensure a and b are aligned with c: shape (m, n)
        # -> b should be a row vector for the subtraction with c
        dist_sq = a + b.reshape(1, -1) - 2 * c

        # K: covariance kernel matrix of shape (m, n)
        K = (self.sigma_f ** 2) * np.exp(-0.5 * (1 / (self.l ** 2)) * dist_sq)
        # print("K.shape:", K.shape)

        return K

    def predict(self, X_s):
        """
        function that predicts the mean and standard deviation of points
        in a Gaussian process
        """

        # Call K
        K = self.K
        # Compute K_s in a call to kernel()
        K_s = self.kernel(self.X, X_s)
        # Compute K_ss in a call to kernel()
        K_ss = self.kernel(X_s, X_s)
        # Call Y
        Y = self.Y

        # The prediction follows a normal distribution completely
        # described by the mean "mu" and the covariance "sigma**2"

        # Compute the mean "mu"
        K_inv = np.linalg.inv(K)
        mu_s = np.matmul(np.matmul(K_s.T, K_inv), Y).reshape(-1)
        # Compute the covariance matrix "cov_s"
        cov_s = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        # Infer the standard deviation "sigma"
        sigma = np.diag(cov_s)

        return mu_s, sigma

    def update(self, X_new, Y_new):
        """function that updates a Gaussian process"""

        # Add the new sample point

        self.X = np.concatenate((self.X, X_new[..., np.newaxis]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[..., np.newaxis]), axis=0)
        # Add the new function value
        self.K = self.kernel(self.X, self.X)
