#!/usr/bin/env python3
"""Script to create a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    # Convert one-hot encoded labels to class indices
    true_labels = np.argmax(labels, axis=1)  # shape (m,)
    pred_labels = np.argmax(logits, axis=1)  # shape (m,)

    # Number of classes
    num_classes = labels.shape[1]

    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for i in range(len(true_labels)):
        confusion_matrix[true_labels[i], pred_labels[i]] += 1

    return confusion_matrix
