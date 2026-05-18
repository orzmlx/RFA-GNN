import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
FIG_DIR = ROOT / "liuthesis_my" / "figures"
RESULT_TEX = ROOT / "liuthesis_my" / "Result.tex"


def parse_top_gene_table():
    text = RESULT_TEX.read_text(encoding="utf-8")
    pattern = re.compile(
        r"Model & Split & Pos P@20 & Neg P@20 \\\\\n\\hline\n(.*?)\\hline",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find top gene precision table in Result.tex")

    rows = {}
    for line in match.group(1).strip().splitlines():
        parts = [x.strip() for x in line.replace("\\\\", "").split("&")]
        if len(parts) != 4:
            continue
        model, split, pos_val, neg_val = parts
        rows[(model, split)] = (float(pos_val), float(neg_val))
    return rows


def draw():
    rows = parse_top_gene_table()
    display_splits = ["Warm", "Cold drug", "Cold cell"]
    models = ["DeepCOP", "GSNN", "CAGNN"]
    colors = {
        "DeepCOP": "#C9DDF2",
        "GSNN": "#F7D8B5",
        "CAGNN": "#CFE8C7",
    }

    x = np.arange(len(display_splits))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=220, sharey=False)
    metric_info = [("Pos P@20", 0), ("Neg P@20", 1)]

    for ax, (title, metric_idx) in zip(axes, metric_info):
        for model_idx, model in enumerate(models):
            vals = [rows[(model, split)][metric_idx] for split in display_splits]
            offset = (model_idx - 1) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[model],
                label=model,
                edgecolor="#8A8A8A",
                linewidth=0.4,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_splits)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Precision")
    axes[1].legend(frameon=False, ncol=1, fontsize=9, loc="upper right")
    fig.tight_layout()

    png_path = FIG_DIR / "top_gene_precision_results.png"
    pdf_path = FIG_DIR / "top_gene_precision_results.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
