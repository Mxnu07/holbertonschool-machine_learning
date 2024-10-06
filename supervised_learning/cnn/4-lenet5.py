#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using TensorFlow
"""

import tensorflow.compat.v1 as tf


tf.disable_eager_execution()  # Disable eager execution for TensorFlow v1


def lenet5(x, y):
    """
    Builds a modified version of the `LeNet-5` architecture using Tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        kernel_initializer=init,
        activation=tf.nn.relu
    )(x)
    M1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C1)
    C2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        kernel_initializer=init,
        activation=tf.nn.relu
    )(M1)
    M2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C2)
    CF = tf.layers.Flatten()(M2)
    FC1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(CF)
    FC2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(FC1)
    y_pred = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(FC2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    truth_max = tf.argmax(y, 1)
    pred_max = tf.argmax(y_pred, 1)
    difference = tf.equal(truth_max, pred_max)
    accuracy = tf.reduce_mean(tf.cast(difference, "float"))
    train_op = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.nn.softmax(y_pred)
    return y_pred, train_op, loss, accuracy
