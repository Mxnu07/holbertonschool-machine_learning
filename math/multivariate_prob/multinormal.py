#!/usr/bin/env python3
"""
multinormal.py
"""
import numpy as np


class MultiNormal:
    """
    Multivariate Normal distribution
    """

    def __init__(self, data):
        """constructor
        Args:
            data: numpy.ndarray - shape (d, n) that contains the data set
            d: number of dimensions
            n: number of data points
        """

        err_1 = "data must be a 2D numpy.ndarray"
        if not isinstance(data, np.ndarray):
            raise TypeError(err_1)
        if data.ndim != 2:
            raise TypeError(err_1)

        err_2 = "data must contain multiple data points"
        if data.shape[1] < 2:
            raise ValueError(err_2)

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """
        function that calculates the mean and covariance matrix of a data set
        Args:
            X: numpy.ndarray - shape (d, n) that contains the data set
            d: number of dimensions
            n: number of data points
        Returns:
            mean: numpy.ndarray - shape (d, 1) containing the mean of data set
            cov: numpy.ndarray - shape (d, d) containing the covariance matrix
        """

        d = X.shape[0]
        n = X.shape[1]

        mean = np.mean(X, axis=1)
        mean = mean[..., np.newaxis]

        X = X - mean
        cov = np.matmul(X, X.T) / (n - 1)

        return mean, cov

    def pdf(self, x):
        """
        function that calculates the PDF at a data point
        Args:
            x: numpy.ndarray - shape (d, 1) containing the data point
            d: number of dimensions
        Returns:
            PDF: float - value of the   PDF at x
        """

        # err1 = "x must be a numpy.ndarray"
        # if not isinstance(x, np.ndarray):
        #     raise TypeError(err1)

        # d = self.cov.shape[0]
        # err2 = "x must have the shape ({}, 1)".format(d)
        # if x.ndim != 2:
        #     raise ValueError(err2)
        # if x.shape[1] != 1 or x.shape[0] != d:
        #     raise ValueError(err2)

        # # A = 1.0 / ((2 * np.pi) ** (d / 2) * np.linalg.det(self.cov) ** 0.5)
        # A = 1.0 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        # B = np.exp(-0.5 * np.linalg.multi_dot([(x - self.mean).T,
        #                                        np.linalg.inv(self.cov),
        #                                        (x - self.mean)]))
        # PDF = A * B

        # return float(PDF)
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape
               ) != 2 or x.shape[1] != 1 or x.shape[0] != self.mean.shape[0]:
            raise ValueError(
                "x must have the shape ({}, 1)".format(self.mean.shape[0]))
        n = self.mean.shape[0]
        x_m = x - self.mean
        pdf = (
            1
            / np.sqrt(((2 * np.pi) ** n) * np.linalg.det(self.cov))
            * np.exp(
                -0.5 * np.dot(np.dot(x_m.T, np.linalg.inv(self.cov)), x_m))
        )
        return pdf.item()
