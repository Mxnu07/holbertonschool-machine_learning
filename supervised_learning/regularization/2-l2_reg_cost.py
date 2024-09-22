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
    # Get the L2 regularization losses from the model
    l2_loss = tf.add_n(model.losses)

    # Add the L2 regularization loss to the original cost
    total_cost = cost + l2_loss

    return total_cost
