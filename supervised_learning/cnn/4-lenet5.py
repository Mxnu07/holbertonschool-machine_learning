#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using TensorFlow
"""

import tensorflow.compat.v1 as tf

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
    # Disable eager execution
    tf.disable_eager_execution()

    # Initialize weights with he_normal method
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # C1: Convolutional layer (6 kernels, 5x5, same padding)
    conv1 = tf.nn.conv2d(x, tf.Variable(initializer([5, 5, 1, 6])), strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, tf.Variable(tf.zeros([6])))
    conv1 = tf.nn.relu(conv1)

    # P2: Max pooling layer (2x2 kernels, 2x2 strides)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # C3: Convolutional layer (16 kernels, 5x5, valid padding)
    conv2 = tf.nn.conv2d(pool1, tf.Variable(initializer([5, 5, 6, 16])), strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, tf.Variable(tf.zeros([16])))
    conv2 = tf.nn.relu(conv2)

    # P4: Max pooling layer (2x2 kernels, 2x2 strides)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten the output
    flatten = tf.reshape(pool2, [-1, 4*4*16])

    # F5: Fully connected layer (120 nodes)
    fc1 = tf.matmul(flatten, tf.Variable(initializer([4*4*16, 120])))
    fc1 = tf.nn.bias_add(fc1, tf.Variable(tf.zeros([120])))
    fc1 = tf.nn.relu(fc1)

    # F6: Fully connected layer (84 nodes)
    fc2 = tf.matmul(fc1, tf.Variable(initializer([120, 84])))
    fc2 = tf.nn.bias_add(fc2, tf.Variable(tf.zeros([84])))
    fc2 = tf.nn.relu(fc2)

    # F7: Fully connected softmax output layer (10 nodes)
    out = tf.matmul(fc2, tf.Variable(initializer([84, 10])))
    out = tf.nn.bias_add(out, tf.Variable(tf.zeros([10])))

    # Softmax activation
    softmax = tf.nn.softmax(out)

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, optimizer, loss, accuracy
