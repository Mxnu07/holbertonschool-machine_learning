#!/usr/bin/env python3
"""This modlue contain the ByseianOptimization class
    That erforms Byseian optimization on a noiseless
    1D Gaussian process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """This class performs Bayesian optimization on a
    noiseless 1D Gaussian process
    """

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True,
    ):
        """This method initializes the class BayesianOpt
        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
            t: the number of initial samples
            bounds: tuple of (min, max) representing the bounds of the space
                    in which to look for the optimal point
            ac_samples: the number of samples that should be analyzed during
                        acquisition
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                    black-box function
            xsi: the exploration-exploitation factor for acquisition
            minimize: a bool determining whether optimization should be
                    performed for minimization (True) or maximization (False)
        """
        self.f = f
        # gp is an instance of the GaussianProcess class
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], num=ac_samples).reshape(-1, 1)
        # print(f"-"*20)
        # print(f"X_s: {self.X_s}")
        # print(f"-"*20)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """This method calculates the next best sample location
            Using the Expected Improvement acquisition function
        Returns: X_next, EI
            X_next: numpy.ndarray of shape (1,) representing the next best
                    sample point
            EI: numpy.ndarray of shape (ac_samples,) containing the expected
                improvement of each potential sample
        """
        # Step 1: predict the mean and standard deviation at
        # all the potential sample
        mu, sigma = self.gp.predict(self.X_s)
        # print(f"mu: {mu}")
        # print(f"sigma: {sigma}")
        if self.minimize:
            # np.min returns the minimum value of an array
            Y_sample_opt = np.min(self.gp.Y)
            # imp is the improvement of the potential sample over the current
            # best sample
            # Y_sample_opt is the current best sample
            # mu is the mean of the potential sample
            # self.xsi is the exploration-exploitation factor for acquisition
            imp = Y_sample_opt - mu - self.xsi
            # print(f"imp: {imp}")
        else:
            # np.max returns the maximum value of an array
            Y_sample_opt = np.max(self.gp.Y)
            # imp is the improvement of the potential sample over the current
            # best sample
            # Y_sample_opt is the current best sample
            # mu is the mean of the potential sample
            # self.xsi is the exploration-exploitation factor for acquisition
            imp = mu - Y_sample_opt - self.xsi
        # Step 2: Calculate the Expected Improvement
        # Z is the number of standard deviations between the mean and the
        # current best sample
        Z = imp / sigma
        # EI is the Expected Improvement
        # norm.cdf returns the cumulative distribution function of the standard
        # norm.pdf returns the probability density function of the standard
        # # sigma is the standard deviation of the black-box function
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        # Step 3: Find the sample with the maximum Expected Improvement
        # X_next is the next best sample point
        # np.argmax returns the index of the maximum value of an array
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
