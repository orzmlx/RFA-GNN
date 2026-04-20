from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures")
SUMMARY_PNG = OUT_DIR / "uncertainty_summary_placeholder.png"
SUMMARY_PDF = OUT_DIR / "uncertainty_summary_placeholder.pdf"
CASE_PNG = OUT_DIR / "uncertainty_case_study_placeholder.png"
CASE_PDF = OUT_DIR / "uncertainty_case_study_placeholder.pdf"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.tick_params(labelsize=9)


def build_summary():
    rng = np.random.default_rng(7)

    colors = {
        "warm": "#AFC8E8",
        "cold_drug": "#F6C89A",
        "cold_cell": "#BFDDB8",
        "line": "#6E7F99",
    }

    warm_sigma = np.clip(rng.normal(0.66, 0.05, 180), 0.48, 0.90)
    cold_drug_sigma = np.clip(rng.normal(0.73, 0.06, 180), 0.52, 1.00)
    cold_cell_sigma = np.clip(rng.normal(0.84, 0.07, 180), 0.58, 1.12)

    n_scatter = 260
    sigma_scatter = rng.uniform(0.55, 1.02, n_scatter)
    mse_scatter = 0.42 + 0.78 * (sigma_scatter - 0.55) + rng.normal(0.0, 0.05, n_scatter)
    pcc_scatter = 0.59 - 0.36 * (sigma_scatter - 0.55) + rng.normal(0.0, 0.03, n_scatter)

    pcc_bins = np.array([0.51, 0.43, 0.34], dtype=float)
    mse_bins = np.array([0.86, 1.00, 1.19], dtype=float)
    bin_labels = ["Low", "Medium", "High"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), dpi=220)
    plt.subplots_adjust(wspace=0.25, hspace=0.34)

    ax = axes[0, 0]
    box = ax.boxplot(
        [warm_sigma, cold_drug_sigma, cold_cell_sigma],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], [colors["warm"], colors["cold_drug"], colors["cold_cell"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#7F7F7F")
    for median in box["medians"]:
        median.set_color("#555555")
        median.set_linewidth(1.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Warm", "Cold drug", "Cold cell"])
    ax.set_ylabel("Mean predicted sigma", fontsize=10)
    ax.set_title("Uncertainty across splits", fontsize=11)
    _style_axes(ax)

    ax = axes[0, 1]
    ax.scatter(sigma_scatter, mse_scatter, s=20, color=colors["warm"], alpha=0.35, edgecolors="none")
    xs = np.linspace(float(np.min(sigma_scatter)), float(np.max(sigma_scatter)), 100)
    coef = np.polyfit(sigma_scatter, mse_scatter, 1)
    ax.plot(xs, np.polyval(coef, xs), color=colors["line"], linewidth=1.8)
    ax.set_xlabel("Mean predicted sigma", fontsize=10)
    ax.set_ylabel("Sample-wise MSE", fontsize=10)
    ax.set_title("Uncertainty and MSE", fontsize=11)
    _style_axes(ax)

    ax = axes[1, 0]
    ax.scatter(sigma_scatter, pcc_scatter, s=20, color=colors["cold_drug"], alpha=0.35, edgecolors="none")
    coef = np.polyfit(sigma_scatter, pcc_scatter, 1)
    ax.plot(xs, np.polyval(coef, xs), color=colors["line"], linewidth=1.8)
    ax.set_xlabel("Mean predicted sigma", fontsize=10)
    ax.set_ylabel("Sample-wise PCC", fontsize=10)
    ax.set_title("Uncertainty and PCC", fontsize=11)
    _style_axes(ax)

    ax = axes[1, 1]
    x = np.arange(len(bin_labels))
    width = 0.34
    ax.bar(x - width / 2, pcc_bins, width, color=colors["cold_cell"], alpha=0.82, label="PCC")
    ax.bar(x + width / 2, mse_bins, width, color=colors["cold_drug"], alpha=0.72, label="MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Average metric value", fontsize=10)
    ax.set_title("Stratified performance by uncertainty", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)

    fig.savefig(SUMMARY_PNG, bbox_inches="tight")
    fig.savefig(SUMMARY_PDF, bbox_inches="tight")
    plt.close(fig)


def build_case_study():
    rng = np.random.default_rng(11)
    genes = [f"G{i}" for i in range(1, 13)]
    x = np.arange(len(genes))

    truth_low = np.array([0.88, 0.75, 0.62, 0.54, 0.41, 0.30, -0.18, -0.28, -0.36, -0.45, -0.57, -0.70])
    mean_low = truth_low + rng.normal(0.0, 0.07, len(genes))
    sigma_low = np.array([0.10, 0.08, 0.09, 0.09, 0.08, 0.10, 0.09, 0.10, 0.08, 0.09, 0.10, 0.11])

    truth_high = np.array([0.92, 0.70, 0.66, 0.46, 0.36, 0.18, -0.10, -0.22, -0.34, -0.47, -0.63, -0.76])
    mean_high = truth_high + rng.normal(0.0, 0.16, len(genes))
    sigma_high = np.array([0.24, 0.22, 0.20, 0.23, 0.21, 0.24, 0.22, 0.25, 0.21, 0.23, 0.24, 0.26])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=220, sharey=True)
    plt.subplots_adjust(wspace=0.10)

    for ax, truth, mean_pred, sigma, title in [
        (axes[0], truth_low, mean_low, sigma_low, "Low uncertainty sample"),
        (axes[1], truth_high, mean_high, sigma_high, "High uncertainty sample"),
    ]:
        ax.plot(x, truth, color="#6E7F99", linewidth=2.0, marker="o", markersize=4.5, label="True response")
        ax.plot(x, mean_pred, color="#AFC8E8", linewidth=2.1, marker="o", markersize=4.5, label="Predicted mean")
        ax.fill_between(x, mean_pred - sigma, mean_pred + sigma, color="#DCEAF7", alpha=0.95, label=r"$\mu \pm \sigma$")
        ax.set_xticks(x)
        ax.set_xticklabels(genes, rotation=30, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Selected genes", fontsize=10)
        _style_axes(ax)

    axes[0].set_ylabel("Predicted response", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=9)
    fig.savefig(CASE_PNG, bbox_inches="tight")
    fig.savefig(CASE_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    build_summary()
    build_case_study()
    print(SUMMARY_PNG)
    print(SUMMARY_PDF)
    print(CASE_PNG)
    print(CASE_PDF)


if __name__ == "__main__":
    main()
