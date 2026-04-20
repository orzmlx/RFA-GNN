from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures")
PNG_PATH = OUT_DIR / "best_worst_drug_case_placeholder.png"
PDF_PATH = OUT_DIR / "best_worst_drug_case_placeholder.pdf"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.tick_params(labelsize=9)


def main():
    genes = ["EXT1", "TRIB3", "CTNND1", "IARS2", "NVL", "FIS1", "COQ8A", "MYLK", "DNAJB1", "CSRP1"]
    y = np.arange(len(genes))

    true_best = np.array([8.8, 8.5, 8.1, 8.0, 7.7, 7.6, -8.0, -8.3, -8.5, -8.7])
    pred_best = np.array([6.1, 6.5, 6.0, 5.7, 5.4, 5.8, -6.2, -5.8, -6.0, -6.4])

    true_worst = np.array([8.7, 8.2, 7.9, 7.4, 7.0, 6.8, -8.3, -8.6, -8.8, -9.0])
    pred_worst = np.array([3.1, 2.7, 3.4, 2.9, 2.5, 2.8, -3.6, -3.1, -2.7, -3.0])

    colors = {
        "true": "#AFC8E8",
        "pred": "#F6C89A",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=220, sharey=True)
    plt.subplots_adjust(wspace=0.08)

    for ax, truth, pred, title in [
        (axes[0], true_best, pred_best, "Best predicted drug"),
        (axes[1], true_worst, pred_worst, "Worst predicted drug"),
    ]:
        ax.barh(y - 0.18, truth, height=0.34, color=colors["true"], alpha=0.90, label="True")
        ax.barh(y + 0.18, pred, height=0.34, color=colors["pred"], alpha=0.85, label="Predicted")
        ax.axvline(0, color="#666666", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(genes)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Response value", fontsize=10)
        _style_axes(ax)

    axes[0].set_ylabel("Selected genes", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
