#!/usr/bin/env python3
""" Neuron class file """
import numpy as np


class Neuron:
    """ Neuron class for performing binary classification """
    def __init__(self, nx):
        """Constructor method for the Neuron class

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Initialize weights
        self.__b = 0  # Initialize bias to 0
        self.__A = 0  # Initialize activated output to 0

    @property
    def W(self):
        """Getter method for weights"""
        return self.__W

    @property
    def b(self):
        """Getter method for bias"""
        return self.__b

    @property
    def A(self):
        """Getter method for activated output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: Activated output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b  # Linear combination
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated output of shape (1, m).

        Returns:
            float: The cost of the model.
        """
        m = Y.shape[1]  # Number of examples
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: Tuple containing the prediction and the cost.
                   -prediction (numpy.ndarray):Predicted labels of shape (1, m)
                   -cost (float): Cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        # Prediction: Apply threshold of 0.5 to the activated output
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost
