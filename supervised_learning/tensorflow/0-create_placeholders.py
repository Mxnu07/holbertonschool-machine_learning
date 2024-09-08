#!/usr/bin/env python3
"""Create placeholders"""
import tensorflow as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """returns two placeholders, x and y, for a neural network
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y