#!/usr/bin/env python3
''' L2 regularization'''

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network with L2 regularization.

    Args:
    cost -- tensor containing the cost of the network without L2 regularization
    model -- Keras model that includes layers with L2 regularization

    Returns:
    total_cost -- tensor containing the total cost including L2 regularization
    """
    # Get the L2 regularization losses from the model as a list
    l2_losses = model.losses

    # Return the original cost combined with each individual L2 loss
    total_cost = tf.add(cost, l2_losses)

    return total_cost
