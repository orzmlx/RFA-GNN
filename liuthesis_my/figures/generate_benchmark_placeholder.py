from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


out_dir = Path("/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/figures")
png_path = out_dir / "benchmark_main_placeholder.png"
pdf_path = out_dir / "benchmark_main_placeholder.pdf"

metrics = ["PCC", "MSE", "Pos P@20", "Neg P@20"]
splits = ["Warm", "Cold-drug", "Cold-cell"]
models = ["DeepCOP", "GSNN", "GAT", "GAT + CF"]

means = {
    "PCC": {
        "Warm": [0.61, 0.64, 0.70, 0.74],
        "Cold-drug": [0.52, 0.56, 0.62, 0.66],
        "Cold-cell": [0.33, 0.39, 0.47, 0.52],
    },
    "MSE": {
        "Warm": [0.47, 0.43, 0.39, 0.36],
        "Cold-drug": [0.59, 0.55, 0.49, 0.46],
        "Cold-cell": [0.82, 0.75, 0.68, 0.63],
    },
    "Pos P@20": {
        "Warm": [0.39, 0.42, 0.46, 0.49],
        "Cold-drug": [0.28, 0.31, 0.35, 0.38],
        "Cold-cell": [0.17, 0.20, 0.24, 0.26],
    },
    "Neg P@20": {
        "Warm": [0.37, 0.40, 0.45, 0.48],
        "Cold-drug": [0.27, 0.30, 0.34, 0.37],
        "Cold-cell": [0.16, 0.19, 0.23, 0.25],
    },
}

palette = {
    "DeepCOP": {"face": "#DDEAF7", "edge": "#7FA6CC"},
    "GSNN": {"face": "#E5E1F8", "edge": "#9B8BC8"},
    "GAT": {"face": "#DDF3E6", "edge": "#69A97C"},
    "GAT + CF": {"face": "#FCE5C7", "edge": "#D89A3D"},
}

rng = np.random.default_rng(7)
replicates = {}
for metric in metrics:
    replicates[metric] = {}
    for split in splits:
        replicates[metric][split] = {}
        for model, mean in zip(models, means[metric][split]):
            if metric == "MSE":
                vals = mean + rng.normal(0, 0.015, size=5)
                vals = np.clip(vals, 0.05, None)
            else:
                vals = mean + rng.normal(0, 0.015, size=5)
                vals = np.clip(vals, 0.05, 0.90)
            replicates[metric][split][model] = vals

fig, axes = plt.subplots(1, 4, figsize=(14.8, 7.8))
fig.patch.set_facecolor("#FFFFFF")

for ax in axes:
    ax.set_facecolor("#FFFFFF")

group_gap = 1.3
bar_h = 0.18
inner_gap = 0.08
group_height = len(models) * (bar_h + inner_gap) + 0.18
group_starts = [i * group_gap for i in range(len(splits))]
split_centers = [start + group_height / 2 - 0.02 for start in group_starts]
model_offsets = np.arange(len(models)) * (bar_h + inner_gap)

for ax, metric in zip(axes, metrics):
    is_error = metric == "MSE"
    for split_idx, split in enumerate(splits):
        start = group_starts[split_idx]
        for model_idx, model in enumerate(models):
            y = start + model_offsets[model_idx]
            vals = replicates[metric][split][model]
            mean = float(np.mean(vals))
            face = palette[model]["face"]
            edge = palette[model]["edge"]
            ax.barh(
                y,
                mean,
                height=bar_h,
                color=face,
                edgecolor=edge,
                linewidth=1.2,
                zorder=2,
            )
            jitter = rng.normal(0, 0.016, size=len(vals))
            ax.scatter(
                vals,
                np.full_like(vals, y) + jitter,
                s=26,
                marker="s",
                facecolors="#FFFFFF",
                edgecolors=edge,
                linewidths=0.9,
                zorder=3,
            )

        top_y = start + model_offsets[-1] + bar_h / 2
        bottom_y = start - bar_h / 2
        x_bracket = 0.915 if not is_error else 0.865
        ax.plot([x_bracket, x_bracket], [bottom_y, top_y], color="#222222", lw=1.0, clip_on=False)
        ax.plot([x_bracket - 0.02, x_bracket], [bottom_y, bottom_y], color="#222222", lw=1.0, clip_on=False)
        ax.plot([x_bracket - 0.02, x_bracket], [top_y, top_y], color="#222222", lw=1.0, clip_on=False)
        ax.text(
            x_bracket + 0.018,
            (bottom_y + top_y) / 2,
            "***",
            va="center",
            ha="left",
            rotation=90,
            fontsize=12.5,
            color="#111111",
        )

    ax.set_title(metric, fontsize=17, pad=16, color="#1C1C1C")
    ax.set_ylim(group_starts[-1] + group_height + 0.08, -0.35)
    ax.set_yticks(split_centers)
    ax.set_yticklabels(splits, fontsize=12, color="#222222")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=11, colors="#333333", pad=6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#D9D9D3", linestyle="-", linewidth=0.8, alpha=0.75, zorder=1)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_color("#666666")
    ax.spines["left"].set_color("#666666")
    if is_error:
        ax.set_xlim(0, 0.92)
        ax.set_xticks(np.arange(0, 0.91, 0.2))
    else:
        ax.set_xlim(0, 0.94)
        ax.set_xticks(np.arange(0, 0.91, 0.2))

handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor=palette[m]["face"],
        markeredgecolor=palette[m]["edge"],
        markeredgewidth=1.2,
        markersize=10,
        label=m,
    )
    for m in models
]

fig.legend(
    handles=handles,
    labels=models,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
    fontsize=12,
)

fig.suptitle(
    "Main Benchmark Across Warm, Cold-drug, and Cold-cell Splits",
    fontsize=20,
    y=0.98,
    color="#1C1C1C",
)

fig.text(
    0.5,
    0.935,
    "Placeholder values for thesis figure layout",
    ha="center",
    va="center",
    fontsize=11.5,
    color="#666666",
)

plt.subplots_adjust(left=0.12, right=0.985, top=0.83, bottom=0.12, wspace=0.24)
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
print(png_path)
print(pdf_path)
