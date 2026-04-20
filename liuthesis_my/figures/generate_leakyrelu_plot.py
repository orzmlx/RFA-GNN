import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-5, 5, 600)
    alpha = 0.2
    y = np.where(x >= 0, x, alpha * x)

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
    ax.plot(x, y, color="#4C78A8", linewidth=2.5, label=fr"$\alpha={alpha}$")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.2, 5.2)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("LeakyReLU(x)", fontsize=10)
    ax.set_title("LeakyReLU", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = "/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures/leakyrelu_function.png"
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
