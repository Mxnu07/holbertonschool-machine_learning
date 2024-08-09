#!/usr/bin/env python3
"""Function that changes the scale of a plot."""
import numpy as np
import matplotlib.pyplot as plt

def change_scale():
    """Changes the scale of plot."""

    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Set Plot
    plt.plot(x, y)

    # Set scale
    plt.yscale('log')

    # Set x-axis limit from 0 to 28650
    plt.xlim(0, 28650)

    # Set labels name
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    plt.show()
