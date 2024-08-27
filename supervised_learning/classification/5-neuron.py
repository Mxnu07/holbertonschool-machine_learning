#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """class Neuron"""
    def __init__(self, nx):
        """nx is the number of input features to the neuron
        Attributes:
        W: The weights vector for the neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b: The bias for the neuron. Upon instantiation, it should be
        initialized to 0.
        A: The activated output of the neuron (prediction). Upon
        instantiation, it should be initialized to 0."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """returns private instance weight"""
        return self.__W

    @property
    def b(self):
        """returns private instance bias"""
        return self.__b

    @property
    def A(self):
        """returns private instance output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X = a numpy.ndarray with shape (nx, m) that contains
        the input data
        nx = the number of input features to the neuron
        m = the number of examples
        It updates the private attribute __A
        neuron should use the sigmoid function"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y = a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A = a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        To avoid division by zero errors, it will be used 1.0000001 - A
        instead of 1 - A"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions
        X = numpy.ndarray with shape (nx, m) that contains the input data
        Y = numpy.ndarray with shape (1, m) that contains the correct
        labels
        nx = the number of input features to the neuron
        m = the number of examples
        It returns the neuron’s prediction and the cost of the network"""
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neuron
        X = numpy.ndarray with shape (nx, m) that contains the input data
        Y = numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A = numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        alpha = the learning rate
        nx = the number of input features to the neuron
        m = the number of examples
        Updates the private attributes __W and __b"""
        m = Y.shape[1]
        dz = A - Y
        dw = 1 / m * np.dot(dz, X.T)
        db = 1 / m * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db
        return self.__W, self.__b
