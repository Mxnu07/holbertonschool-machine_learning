#!/usr/bin/env python3
""" LeNet-5 architecture using tensorflow """

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ Function that builds a modified version of the LeNet-5 architecture"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # 1. First Convolutional Layer
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # 2. First Max Pooling Layer
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=(2, 2),
        strides=2
    )

    # 3. Second Convolutional Layer
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # 4. Second Max Pooling Layer
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=(2, 2),
        strides=2
    )

    # Flatten the output for Fully Connected Layers
    flatten = tf.layers.flatten(pool2)

    # 5. First Fully Connected Layer
    fc1 = tf.layers.dense(
        inputs=flatten,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # 6. Second Fully Connected Layer
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # 7. Output Layer (Softmax)
    output = tf.layers.dense(
        inputs=fc2,
        units=10,
        activation=None,
        kernel_initializer=initializer
    )

    # Softmax activation for output
    y_pred = tf.nn.softmax(output)

    # Loss function
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output
    )

    # Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, optimizer, loss, accuracy
