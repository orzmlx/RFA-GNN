from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures")
PNG_PATH = OUT_DIR / "hybrid_context_weights_placeholder.png"
PDF_PATH = OUT_DIR / "hybrid_context_weights_placeholder.pdf"


def main():
    epochs = np.arange(1, 51)

    # Synthetic placeholder trends for layout only.
    cell_weight = 0.62 - 0.28 * (1 - np.exp(-epochs / 16.0))
    context_weight = 1.0 - cell_weight

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=220)

    ax.plot(
        epochs,
        cell_weight,
        color="#AFC8E8",
        linewidth=2.4,
        label="Cell identity weight",
    )
    ax.plot(
        epochs,
        context_weight,
        color="#BFDDB8",
        linewidth=2.4,
        label="Control context weight",
    )

    ax.fill_between(epochs, cell_weight - 0.015, cell_weight + 0.015, color="#DCEAF7", alpha=0.9)
    ax.fill_between(epochs, context_weight - 0.015, context_weight + 0.015, color="#E2F0DD", alpha=0.9)

    ax.set_xlim(1, 50)
    ax.set_ylim(0.25, 0.75)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Weight", fontsize=11)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="center right")

    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
