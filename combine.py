import numpy as np
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

J = 0.4
L = 3
N = 150

epsilon = 1e-4
offset = 0

path1 = f"d:/results/bh/{L}/"
path2 = f"results/L={L} N={N} J={J} U=1.0"

def plot_lines(filename, color="black", interval=[0, 2.5]):
    data = np.loadtxt(filename, delimiter=",")
    
    if len(data) == 0 or len(data.shape) != 2 or data.shape[1] != 2:
        return

    js, es = data[:, 0], data[:, 1]

    # Find indices where x is within epsilon
    matching_indices = np.where(np.abs(js - J) <= epsilon)[0]

    for i in matching_indices:
        # Plot the line
        plt.plot([es[i] + offset, es[i] + offset], interval, color=color)

def plot_points(filename, type, color):
    es = np.loadtxt(f"{path2} {filename}.csv")
    data = np.loadtxt(f"{path2} {filename} {type}.csv")

    plt.scatter(es, data, color=color, label=f"{filename}", s=1, alpha=0.5)

def plot():
    color = ["red", "green", "black", "blue", "orange", "purple", "brown", "pink"]

    for i in range(4 * (L - 1) + 1):
        plot_lines(f"{path1}hunstable_{i}.txt", "red")
        plot_lines(f"{path1}hsaddle_{i}.txt", "green")
        plot_lines(f"{path1}hstable_{i}.txt", "black")

    for k in range(L):
        plot_points(f"k={k} parity=1", "entropy", color[k])

        if k == 0 or 2 * k == L:
            plot_points(f"k={k} parity=-1", "entropy", color[k])

plot()

plt.legend()
plt.show()