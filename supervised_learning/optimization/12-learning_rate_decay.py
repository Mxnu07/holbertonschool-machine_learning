#!/usr/bin/env python3
"""Script to implement learning rate decay in DNN with Tensorflow"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.

    Parameters:
    - alpha: original learning rate (float).
    - decay_rate: rate at which the learning rate decays (float).
    - decay_step: number of steps before each decay (int).

    Returns:
    - learning rate decay operation: a TensorFlow learning rate schedule.
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return lr_schedule
