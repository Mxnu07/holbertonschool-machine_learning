#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using TensorFlow
"""

import tensorflow.compat.v1 as tf


tf.disable_eager_execution()  # Disable eager execution for TensorFlow v1


def lenet5(x, y):
    """
    Builds a modified version of LeNet-5 architecture using TensorFlow

    parameters:
        x [tf.placeholder of shape (m, 28, 28, 1)]:
            contains the input images for the network
            m: number of images
        y [tf.placeholder of shape (m, 10)]:
            contains the one-hot labels for the network

    returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (default hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """

    # C1: Convolutional layer (6 kernels, 5x5, same padding)
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu)

    # P2: Max pooling layer (2x2 kernels, 2x2 strides)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # C3: Convolutional layer (16 kernels, 5x5, valid padding)
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu)

    # P4: Max pooling layer (2x2 kernels, 2x2 strides)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten the output
    flatten = tf.layers.flatten(pool2)

    # F5: Fully connected layer (120 nodes)
    fc1 = tf.layers.dense(flatten, units=120, activation=tf.nn.relu)

    # F6: Fully connected layer (84 nodes)
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu)

    # F7: Fully connected softmax output layer (10 nodes)
    logits = tf.layers.dense(fc2, units=10)

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                     labels=y))
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, optimizer, loss, accuracy
