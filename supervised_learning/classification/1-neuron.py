#!/usr/bin/env python3
""" Neuron class file """
import numpy as np


class Neuron:
    """ Neuron class """
    def __init__(self, nx):
        """Constructor method for the Neuron class

        Args:
            nx (int): Number of inputs.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Initialize the weight
        self.__b = 0  # Initialize the bias to 0
        self.__A = 0  # Initialize the activated output to 0

    @property
    def W(self):
        """Getter method for the weights"""
        return self.__W

    @property
    def b(self):
        """Getter method for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter method for the activated output"""
        return self.__A
