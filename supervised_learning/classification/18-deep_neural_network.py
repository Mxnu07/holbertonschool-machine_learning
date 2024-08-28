#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """nx is the number of input features
        layers is a list representing the number of nodes in each
        layer of the network
        L: The number of layers in the neural network.
        cache: A dictionary to hold all intermediary values of the network.
        weights: A dictionary to hold all weights and biased of the network.
        The weights of the network should be initialized using the He et al.
        method and the biases should be initialized to 0's"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        # attribute L = number of layers in the neural network
        self.__L = len(layers)
        # cache = empty dictionary to hold all
        # intermediary values of the network
        self.__cache = {}
        # weights = empty dictionary to hold all
        # weights and biased of the network
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            # if layer is the first layer
            if i == 0:
                # He et al. method to initialize weights
                self.weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                # bias initialized to 0
                self.weights['b1'] = np.zeros((layers[i], 1))
            else:
                # He et al. method to initialize weights if i > 0
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                # bias initialized to 0
                self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """returns private instance attribute L"""
        return self.__L

    @property
    def cache(self):
        """returns private instance attribute cache"""
        return self.__cache

    @property
    def weights(self):
        """returns private instance attribute weights"""
        return self.__weights

    def forward_prop(self, X):
        """calculates the forward propagation of the neural network
        X: a numpy.ndarray with shape (nx, m) that contains the
        input data
        nx: is the number of input features to the neuron
        m: is the number of examples
        Updates the private attribute __A1
        The neuron uses a sigmoid activation function"""
        # save the input data in the cache
        self.cache['A0'] = X
        for i in range(self.__L):
            W = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            A = 'A' + str(i)
            Z = np.dot(self.weights[W], self.__cache[A])
            Z = Z + self.weights[b]
            # sigmioid activation function
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
        # return output and the cache
        return self.cache['A' + str(self.__L)], self.cache
