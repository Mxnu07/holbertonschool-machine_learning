#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # Set Plot
    rows = ('apples', 'bananas', 'oranges', 'peaches')
    columns = ('Farrah', 'Fred', 'Felicia')
    index = columns
    colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
    n_rows = len(fruit)
    bar_width = 0.5
    y_offset = np.zeros(len(columns))

    # Plot
    for row in range(n_rows):
        plt.bar(index, fruit[row], bar_width, bottom=y_offset,
                color=colors[row], label=rows[row])
        y_offset = y_offset + fruit[row]

    # Set labels name
    plt.legend()
    plt.yticks(np.arange(0, 90, 10))
    plt.ylabel('Quantity of Fruit')
    plt.title("Number of Fruit per Person")
    plt.show()

bars()