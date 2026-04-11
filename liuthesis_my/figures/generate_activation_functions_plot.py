import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 600)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200)

plots = [
    ("ReLU", relu, "#4C78A8", (-0.2, 5.2)),
    ("Sigmoid", sigmoid, "#F58518", (-0.05, 1.05)),
    ("Tanh", tanh, "#54A24B", (-1.1, 1.1)),
]

for ax, (title, y, color, ylim) in zip(axes, plots):
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures/activation_functions.png", bbox_inches="tight")
