#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""
import numpy as np
from matplotlib import pyplot as plt
import pickle


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

    def cost(self, Y, A):
        """calculate the cost of the model"""
        # Y = 2D array
        m = Y.shape[1]
        # Y * np.log(A) = log loss one class label of
        # one example, only class labels that are 1
        # (1 - Y) * np.log(1.0000001 - A) = log loss
        # one class label of one example, only class 0
        # np.sum adds up the log loss all class labels
        # -1 / m averages the log loss all examples
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) *
                               np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """evaluates the neural network's predictions
        X: numpy.ndarray with shape (nx, m) that contains the
        input data
        Y: numpy.ndarray with shape (1, m) contains the correct
        labels for the input data"""
        # forward propagation
        self.forward_prop(X)
        # A = the neuron's prediction
        A = self.cache['A' + str(self.__L)]
        # Y_prediction = the neuron's prediction
        Y_prediction = np.where(A >= 0.5, 1, 0)
        # cost of the network
        cost = self.cost(Y, A)
        return Y_prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculates one pass of gradient on the
        neural network
        Y = numpy.ndarray with shape (1, m) contains the
        correct labels for the input data
        cache = dictionary containing all intermediary
        values of the network
        alpha = the learning rate"""
        # m = number of examples
        m = Y.shape[1]
        # copy weights
        weights_copy = self.__weights.copy()
        # back propagation / loop over each layer
        # in reverse order (output -> input)
        for i in range(self.__L, 0, -1):
            # retrieve activation values of the
            # current and previous layer
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            # weights and bias of the current layer
            W = weights_copy['W' + str(i)]
            b = weights_copy['b' + str(i)]
            # if current layer is output layer
            # dz in respect to z
            if i == self.__L:
                dz = A - Y
            else:
                dz = da * (A * (1 - A))
            # derivative of cost with respect to
            # weights(dw) and
            # biases(db) of the current layer
            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            # da = deriviates of the cost with respect to
            # the activation of the previous layer
            da = np.dot(weights_copy['W' + str(i)].T, dz)
            # update weights and biases of the current layers
            # by substracing the product of the learning rate
            # and the derivatives dw and db
            self.__weights['W' + str(i)] = weights_copy[
                'W' + str(i)] - alpha * dw
            self.__weights['b' + str(i)] = weights_copy[
                'b' + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """trains the deep neural network
        X: numpy.ndarray with shape (nx, m) that contains the
        input data
        Y: numpy.ndarray with shape (1, m) contains the correct
        labels for the input data
        iterations: positive integer containing the number of
        iterations to train over
        alpha: positive integer containing the learning rate
        verbose: boolean to determine if output should be printed
        graph: boolean to indicate whether to graph information"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph is True:
            plt.plot(costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """save the instance object to a file in pickle format"""
        # file extension is not pkl
        # add pkl extension to the filename
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        # write the object to a file using pickle
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load(filename):
        """load a pickled DeepNeuralNetwork object"""
        if not filename.endswith('.pkl'):
            return None
        try:
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
            return obj
        except FileNotFoundError:
            return None
