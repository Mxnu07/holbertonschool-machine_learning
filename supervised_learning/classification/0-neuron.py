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

        self.W = np.random.randn(1, nx)  # Initialize the weight
        self.b = 0  # Initialize the bias to 0
        self.A = 0  # Initialize the activated output to 0
