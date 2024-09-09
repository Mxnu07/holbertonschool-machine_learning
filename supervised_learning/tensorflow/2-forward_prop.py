#!/usr/bin/env python3
"""create the forward propagation"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """x = placeholder for the input data
    layer_sizes = list containing the number of nodes in
    each layer of the network
    activations = list containing the activation functions
    for each layer of the network
    Returns: the prediction of the network in tensor form"""
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i == 0:
            prediction = create_layer(x, layer_sizes[i], activations[i])
        else:
            prediction = create_layer(prediction, layer_sizes[i],
                                      activations[i])
    return prediction
