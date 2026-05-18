from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
FIG_DIR = ROOT / "liuthesis_my" / "figures"
NPZ_PATH = ROOT / "results" / "hybrid_context" / "hybrid_context_budget_eval.full.warm.npz"
LANDMARK_PATH = ROOT / "data" / "landmark_genes.json"


def load_symbol_map():
    with LANDMARK_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {str(item["entrez_id"]): str(item["gene_symbol"]) for item in items}


def load_npz():
    z = np.load(NPZ_PATH, allow_pickle=True)
    return {key: z[key] for key in z.files}


def fit_line(x, y):
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 120)
    ys = coef[0] * xs + coef[1]
    return xs, ys


def main():
    data = load_npz()
    symbol_map = load_symbol_map()

    y_true = np.asarray(data["y_true"], dtype=np.float32)
    y_pred = np.asarray(data["y_pred"], dtype=np.float32)
    y_logvar = np.asarray(data["y_logvar"], dtype=np.float32)
    genes = np.asarray(data["target_genes"], dtype=str)

    sigma = np.exp(0.5 * y_logvar)
    gene_mean_sigma = np.mean(sigma, axis=0)
    gene_mse = np.mean((y_true - y_pred) ** 2, axis=0)
    gene_mean_true = np.mean(y_true, axis=0)

    top_k = 20
    top_idx = np.argsort(-gene_mean_sigma)[:top_k]
    top_labels = [symbol_map.get(genes[i], genes[i]) for i in top_idx]
    top_sigma = gene_mean_sigma[top_idx]

    corr = float(np.corrcoef(gene_mean_sigma, gene_mse)[0, 1])
    xs, ys = fit_line(gene_mean_sigma, gene_mse)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), dpi=220)

    ax = axes[0]
    xpos = np.arange(top_k)
    ax.bar(
        xpos,
        top_sigma,
        color="#CFE8C7",
        edgecolor="#7A7A7A",
        linewidth=0.6,
        width=0.72,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(top_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean gene-level sigma")
    ax.set_title("Top genes by mean uncertainty")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    scatter = ax.scatter(
        gene_mean_sigma,
        gene_mse,
        c=gene_mean_true,
        cmap="YlGnBu",
        s=22,
        alpha=0.65,
        edgecolors="none",
    )
    ax.plot(xs, ys, color="#5C8FBD", linewidth=1.5)
    ax.text(0.03, 0.96, f"r = {corr:.2f}", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlabel("Mean gene-level sigma")
    ax.set_ylabel("Gene-level MSE")
    ax.set_title("Gene uncertainty and gene MSE")
    ax.grid(alpha=0.16, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean of true response")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gene_level_uncertainty_warm.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "gene_level_uncertainty_warm.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
