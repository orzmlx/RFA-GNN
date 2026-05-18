from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "liuthesis_my" / "figures"


# Figure 5.10 keeps the previous GSNN and CAGNN series.
# DeepCOP is updated from the latest warm budget run logs.
BUDGET_LABELS = ["One shot", "20%", "30%", "50%", "80%"]
X = np.arange(len(BUDGET_LABELS))

SERIES = {
    "DeepCOP": {
        "color": "#729ece",
        "marker": "o",
        "pcc": [0.3712, 0.4109, 0.4238, 0.4376, 0.4627],
        "mse": [0.9672, 0.9290, 0.9234, 0.9059, 0.8843],
    },
    "GSNN": {
        "color": "#f39c46",
        "marker": "s",
        "pcc": [0.4230, 0.4190, 0.4180, 0.4170, 0.4230],
        "mse": [1.0400, 1.0700, 1.0620, 1.0840, 1.0600],
    },
    "CAGNN": {
        "color": "#60ad5e",
        "marker": "D",
        "pcc": [0.4545, 0.5119, 0.5283, 0.5390, 0.5542],
        "mse": [0.9988, 0.9108, 0.8819, 0.8608, 0.8329],
    },
}


def annotate_series(ax, xs, ys, color, dy):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, f"{y:.3f}", color=color, fontsize=9, ha="center", va="bottom")


def main():
    plt.style.use("default")
    fig, (ax_pcc, ax_mse) = plt.subplots(1, 2, figsize=(12, 4.6))

    for ax in (ax_pcc, ax_mse):
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(X)
        ax.set_xticklabels(BUDGET_LABELS, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for name, spec in SERIES.items():
        ax_pcc.plot(
            X,
            spec["pcc"],
            color=spec["color"],
            marker=spec["marker"],
            linewidth=2.0,
            markersize=5,
            label=name,
        )
        annotate_series(ax_pcc, X, spec["pcc"], spec["color"], dy=0.006)

        ax_mse.plot(
            X,
            spec["mse"],
            color=spec["color"],
            marker=spec["marker"],
            linewidth=2.0,
            markersize=5,
            label=name,
        )
        annotate_series(ax_mse, X, spec["mse"], spec["color"], dy=0.010)

    ax_pcc.set_ylabel("Test PCC", fontsize=14)
    ax_pcc.set_xlabel("Training budget per drug-cell pair", fontsize=14)
    ax_pcc.set_ylim(0.30, 0.58)

    ax_mse.set_ylabel("Test MSE", fontsize=14)
    ax_mse.set_xlabel("Training budget per drug-cell pair", fontsize=14)
    ax_mse.set_ylim(0.80, 1.12)

    handles, labels = ax_pcc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_png = FIG_DIR / "data_efficiency_warm_real.png"
    out_pdf = FIG_DIR / "data_efficiency_warm_real.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
