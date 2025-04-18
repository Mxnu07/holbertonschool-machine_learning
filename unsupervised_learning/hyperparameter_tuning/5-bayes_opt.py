#!/usr/bin/env python3
"""This modlue contain the ByseianOptimization class
    That erforms Byseian optimization on a noiseless
    1D Gaussian process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


# class BayesianOptimization:
#     """This class performs Bayesian optimization on a
#     noiseless 1D Gaussian process
#     """

#     def __init__(
#         self,
#         f,
#         X_init,
#         Y_init,
#         bounds,
#         ac_samples,
#         l=1,
#         sigma_f=1,
#         xsi=0.01,
#         minimize=True,
#     ):
#         """This method initializes the class BayesianOpt
#         Args:
#             f: the black-box function to be optimized
#             X_init: numpy.ndarray of shape (t, 1) representing the inputs
#                     already sampled with the black-box function
#             Y_init: numpy.ndarray of shape (t, 1) representing the outputs
#                     of the black-box function for each input in X_init
#             t: the number of initial samples
#             bounds: tuple of (min, max) representing the bounds of the space
#                     in which to look for the optimal point
#             ac_samples: the number of samples that should be analyzed during
#                         acquisition
#             l: the length parameter for the kernel
#             sigma_f: the standard deviation given to the output of the
#                     black-box function
#             xsi: the exploration-exploitation factor for acquisition
#             minimize: a bool determining whether optimization should be
#                     performed for minimization (True) or maximization (False)
#         """
#         self.f = f
#         # gp is an instance of the GaussianProcess class
#         self.gp = GP(X_init, Y_init, l, sigma_f)
#         self.X_s = np.linspace(
#             bounds[0], bounds[1], num=ac_samples).reshape(-1, 1)
#         # print(f"-"*20)
#         # print(f"X_s: {self.X_s}")
#         # print(f"-"*20)
#         self.xsi = xsi
#         self.minimize = minimize

#     def acquisition(self):
#         """This method calculates the next best sample location
#             Using the Expected Improvement acquisition function
#         Returns: X_next, EI
#             X_next: numpy.ndarray of shape (1,) representing the next best
#                     sample point
#             EI: numpy.ndarray of shape (ac_samples,) containing the expected
#                 improvement of each potential sample
#         """
#         # Step 1: predict the mean and standard deviation at
#         # all the potential sample
#         mu, sigma = self.gp.predict(self.X_s)
#         sigma = np.maximum(sigma, 1e-8)
#         # print(f"mu: {mu}")
#         # print(f"sigma: {sigma}")
#         if self.minimize:
#             # np.min returns the minimum value of an array
#             Y_sample_opt = np.min(self.gp.Y)
#             # imp is the improvement of the potential sample over the current
#             # best sample
#             # Y_sample_opt is the current best sample
#             # mu is the mean of the potential sample
#             # self.xsi is the exploration-exploitation factor for acquisition
#             imp = Y_sample_opt - mu - self.xsi
#             # print(f"imp: {imp}")
#         else:
#             # np.max returns the maximum value of an array
#             Y_sample_opt = np.max(self.gp.Y)
#             # imp is the improvement of the potential sample over the current
#             # best sample
#             # Y_sample_opt is the current best sample
#             # mu is the mean of the potential sample
#             # self.xsi is the exploration-exploitation factor for acquisition
#             imp = mu - Y_sample_opt - self.xsi
#         # Step 2: Calculate the Expected Improvement
#         # Z is the number of standard deviations between the mean and the
#         # current best sample
#         Z = imp / sigma
#         # EI is the Expected Improvement
#         # norm.cdf returns the cumulative distribution function of the
#         # standard
#         # norm.pdf returns the probability density function of the standard
#         # # sigma is the standard deviation of the black-box function
#         EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
#         # Step 3: Find the sample with the maximum Expected Improvement
#         # X_next is the next best sample point
#         # np.argmax returns the index of the maximum value of an array
#         X_next = self.X_s[np.argmax(EI)]
#         return X_next, EI

#     def optimize(self, iterations=100):
#         """This method optimazes the black-box fuction
#         Args:
#             iterations: the number of iterations to perform
#         Returns: X_opt, Y_opt
#             X_opt: numpy.ndarray of shape (1,) representing the optimal point
#             Y_opt: numpy.ndarray of shape (1,) representing the
#                     optimal function
#         """
#         for i in range(iterations):
#             # Step 1: Calculate the next best sample
#             X_next, _ = self.acquisition()
#             # Step 2: Evaluate the black-box function
#             Y_next = self.f(X_next)
#             # Step 3: Update the Gaussian process
#             self.gp.update(X_next, Y_next)
#         if self.minimize:
#             Y_opt = np.min(self.gp.Y)
#             X_opt = self.gp.X[np.argmin(self.gp.Y)]
#         else:
#             Y_opt = np.max(self.gp.Y)
#             X_opt = self.gp.X[np.argmax(self.gp.Y)]
#         return X_opt, Y_opt

class BayesianOptimization:
    """
    Class that instantiates a Bayesian optimization
    on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """define and initialize variables and methods"""

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """function that calculates the next best sample location"""

        # Compute mu and sigma in a call to predict() on gp
        mu, sigma = self.gp.predict(self.X_s)
        # print("mu:", mu, mu.shape)
        # print("sigma:", sigma, sigma.shape)

        # Note: sigma of shape (s,)
        Z = np.zeros(sigma.shape)
        if self.minimize is True:
            f_plus = np.min(self.gp.Y)
            Z_NUM = f_plus - mu - self.xsi
        else:
            f_plus = np.max(self.gp.Y)
            Z_NUM = mu - f_plus - self.xsi

        for i in range(sigma.shape[0]):
            if sigma[i] > 0:
                Z[i] = Z_NUM[i] / sigma[i]
            else:
                Z[i] = 0

        # Compute the Expected Improvement (EI)
        EI = np.zeros(sigma.shape)
        for i in range(sigma.shape[0]):
            if sigma[i] > 0:
                EI[i] = Z_NUM[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
            else:
                EI[i] = 0
        X_next = self.X_s[np.argmax(EI)]

        # print("EI:", EI)
        # print("self.X_s:", self.X_s)
        return X_next, EI

    def optimize(self, iterations=100):
        """function that optimizes the black-box function"""

        # Keep track of newly sampled points
        all_X_next = []

        for i in range(iterations):

            # Compute X_next in call to acquisition()
            X_next, _ = self.acquisition()
            # Exit condition on X_next (early stopping)
            if X_next in all_X_next:
                break
            # Evaluate Y_next from the black-box function f
            Y_next = self.f(X_next)
            # Update the Gaussian Process with the newly sampled
            # point (X_next, Y_next)
            self.gp.update(X_next, Y_next)
            # Save X_next
            all_X_next.append(X_next)

        if self.minimize is True:
            Y_opt = np.min(self.gp.Y)[np.newaxis]
            index = np.argmin(self.gp.Y)
        else:
            Y_opt = np.max(self.gp.Y)[np.newaxis]
            index = np.argmax(self.gp.Y)
        X_opt = self.gp.X[index]

        return X_opt, Y_opt
