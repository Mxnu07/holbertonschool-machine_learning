#!/usr/bin/env python3
""" Function that plots a histogram from a dataset."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram from a dataset."""

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Set Plot
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")

    # Set labels name
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Set axis limit
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Show plot
    plt.show()
