from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures")
PNG_PATH = OUT_DIR / "data_efficiency_placeholder.png"
PDF_PATH = OUT_DIR / "data_efficiency_placeholder.pdf"


def main():
    budgets = ["Zero shot", "One shot", "20%", "30%", "50%", "80%"]
    x = np.arange(len(budgets))

    # Synthetic placeholder values for visual drafting only.
    curves = {
        "Warm": {
            "DeepCOP": np.array([0.42, 0.50, 0.53, 0.55, 0.58, 0.60]),
            "GSNN": np.array([0.49, 0.57, 0.59, 0.60, 0.62, 0.64]),
            "CAGNN": np.array([0.54, 0.61, 0.63, 0.64, 0.66, 0.68]),
        },
        "Cold drug": {
            "DeepCOP": np.array([0.31, 0.39, 0.42, 0.44, 0.47, 0.49]),
            "GSNN": np.array([0.39, 0.48, 0.50, 0.52, 0.55, 0.57]),
            "CAGNN": np.array([0.45, 0.54, 0.56, 0.58, 0.61, 0.63]),
        },
        "Cold cell": {
            "DeepCOP": np.array([0.18, 0.25, 0.28, 0.30, 0.33, 0.35]),
            "GSNN": np.array([0.24, 0.33, 0.36, 0.38, 0.42, 0.45]),
            "CAGNN": np.array([0.29, 0.40, 0.43, 0.46, 0.50, 0.53]),
        },
    }

    spreads = {
        "DeepCOP": np.array([0.015, 0.018, 0.017, 0.017, 0.018, 0.018]),
        "GSNN": np.array([0.017, 0.020, 0.020, 0.019, 0.020, 0.021]),
        "CAGNN": np.array([0.018, 0.022, 0.021, 0.021, 0.022, 0.023]),
    }

    colors = {
        "DeepCOP": "#AFC8E8",
        "GSNN": "#F6C89A",
        "CAGNN": "#BFDDB8",
    }

    markers = {
        "DeepCOP": "o",
        "GSNN": "v",
        "CAGNN": "D",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), dpi=220, sharey=True)
    plt.subplots_adjust(wspace=0.18, right=0.84)

    for ax, (panel_title, panel_curves) in zip(axes, curves.items()):
        for model_name in ["DeepCOP", "GSNN", "CAGNN"]:
            y = panel_curves[model_name]
            sd = spreads[model_name]
            ax.plot(
                x,
                y,
                color=colors[model_name],
                marker=markers[model_name],
                linewidth=2.2,
                markersize=5.8,
                label=model_name,
            )
            ax.fill_between(
                x,
                y - sd,
                y + sd,
                color=colors[model_name],
                alpha=0.20,
                linewidth=0,
            )

        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(budgets, rotation=0)
        ax.set_ylim(0.1, 0.72)
        ax.set_xlim(-0.35, len(budgets) - 0.65)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

    axes[0].set_ylabel("PCC", fontsize=13)
    fig.supxlabel("Percentage of training data (per cell-drug pair)", fontsize=13, y=0.03)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=False, fontsize=11)

    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
