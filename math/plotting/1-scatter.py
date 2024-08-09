#!/usr/bin/env python3
"""Function that plots a scatter plot of"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Plot a scatter"""

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Scatter plot
    plt.scatter(x, y, c='magenta', s=10)

    # Set labels name
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title('Men\'s Height vs Weight')

    # Display the plot
    plt.show()
