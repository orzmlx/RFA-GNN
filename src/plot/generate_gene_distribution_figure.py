from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import project_paths


PATHS = project_paths()
ROOT = PATHS["root"]
FIG_DIR = PATHS["fig_dir"]
DATA_DIR = PATHS["data_dir"]

RUN_PATHS = {
    "Warm": {
        "DeepCOP": ROOT / "outputs" / "deepcop_uncertainty_0524" / "deepcop_uncertainty.pred.warm.npz",
        "GSNN": ROOT / "outputs" / "gsnn_0524" / "gsnn_results.pred.warm.npz",
        "no-CF": ROOT / "outputs" / "no_cf_0523" / "ugat_no_cf_uncertainty_sparse.eval.warm.npz",
        "CF": ROOT / "outputs" / "with_cf_gat_0524" / "cagnn_control_context.eval.warm.npz",
    },
    "Cold drug target": {
        "DeepCOP": ROOT / "outputs" / "deepcop_uncertainty_0524" / "deepcop_uncertainty.pred.cold_target_pattern.npz",
        "GSNN": ROOT / "outputs" / "gsnn_0524" / "gsnn_results.pred.cold_target_pattern.npz",
        "no-CF": ROOT / "outputs" / "no_cf_0523" / "ugat_no_cf_uncertainty_sparse.eval.cold_target_pattern.npz",
        "CF": ROOT / "outputs" / "with_cf_gat_0524" / "cagnn_control_context.eval.cold_target_pattern.npz",
    },
    "Cold cell": {
        "DeepCOP": ROOT / "outputs" / "deepcop_uncertainty_0524" / "deepcop_uncertainty.pred.cold_cell.npz",
        "GSNN": ROOT / "outputs" / "gsnn_0524" / "gsnn_results.pred.cold_cell.npz",
        "no-CF": ROOT / "outputs" / "no_cf_0523" / "ugat_no_cf_uncertainty_sparse.eval.cold_cell.npz",
        "CF": ROOT / "outputs" / "with_cf_gat_0524" / "cagnn_control_context.eval.cold_cell.npz",
    },
}

COLORS = {
    "Truth": "#D9D9D9",
    "DeepCOP": "#C9DDF2",
    "GSNN": "#F7D8B5",
    "no-CF": "#CFE8C7",
    "CF": "#F4C7CF",
}


def load_symbol_map():
    with (DATA_DIR / "landmark_genes.json").open("r", encoding="utf-8") as f:
        items = json.load(f)
    mapping = {str(item["entrez_id"]): str(item["gene_symbol"]) for item in items}
    if len(mapping) != 978:
        gene_info = pd.read_csv(DATA_DIR / "GSE92742_Broad_LINCS_gene_info.txt", sep="\t")
        extra = {
            str(row["pr_gene_id"]): str(row["pr_gene_symbol"])
            for _, row in gene_info.iterrows()
        }
        mapping.update(extra)
    return mapping


def load_run(path):
    data = np.load(path, allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    return payload


def gene_ids_for_split(split_runs):
    no_cf = split_runs["no-CF"]
    if "target_genes" in no_cf:
        return np.asarray(no_cf["target_genes"], dtype=str)
    return np.asarray([str(i) for i in range(no_cf["y_true"].shape[1])], dtype=str)


def top_gene_indices(y_true, top_k=12):
    mean_abs = np.mean(np.abs(y_true), axis=0)
    return np.argsort(-mean_abs)[:top_k]


def add_box_group(ax, base_x, arrays, width=0.16):
    offsets = np.array([-0.36, -0.18, 0.0, 0.18, 0.36])
    labels = ["Truth", "DeepCOP", "GSNN", "no-CF", "CF"]
    for offset, label, values in zip(offsets, labels, arrays):
        bp = ax.boxplot(
            values,
            positions=[base_x + offset],
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#666666", "linewidth": 1.0},
            whiskerprops={"color": "#7A7A7A", "linewidth": 0.8},
            capprops={"color": "#7A7A7A", "linewidth": 0.8},
            boxprops={"edgecolor": "#7A7A7A", "linewidth": 0.8},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[label])


def generate():
    symbol_map = load_symbol_map()
    loaded = {
        split: {name: load_run(path) for name, path in runs.items()}
        for split, runs in RUN_PATHS.items()
    }

    fig, axes = plt.subplots(3, 1, figsize=(18, 13.5), dpi=220, sharey=False)

    for ax, (split_name, split_runs) in zip(axes, loaded.items()):
        y_true = np.asarray(split_runs["no-CF"]["y_true"], dtype=np.float32)
        gene_ids = gene_ids_for_split(split_runs)
        idx = top_gene_indices(y_true, top_k=12)
        gene_labels = [symbol_map.get(gene_ids[i], gene_ids[i]) for i in idx]

        for xpos, gene_idx in enumerate(idx):
            arrays = [
                np.asarray(split_runs["no-CF"]["y_true"][:, gene_idx], dtype=np.float32),
                np.asarray(split_runs["DeepCOP"]["y_pred"][:, gene_idx], dtype=np.float32),
                np.asarray(split_runs["GSNN"]["y_pred"][:, gene_idx], dtype=np.float32),
                np.asarray(split_runs["no-CF"]["y_pred"][:, gene_idx], dtype=np.float32),
                np.asarray(split_runs["CF"]["y_pred"][:, gene_idx], dtype=np.float32),
            ]
            add_box_group(ax, xpos, arrays)

        ax.set_title(split_name)
        ax.set_xticks(np.arange(len(idx)))
        ax.set_xticklabels(gene_labels, rotation=30, ha="right", fontsize=13)
        ax.set_ylabel(r"$x_{\mathrm{deg}}$")
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[name], edgecolor="#7A7A7A")
        for name in ["Truth", "DeepCOP", "GSNN", "no-CF", "CF"]
    ]
    fig.legend(handles, ["Truth", "DeepCOP", "GSNN", "no-CF", "CF"], ncol=5, frameon=False, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    png_path = FIG_DIR / "xpert_style_gene_distribution.png"
    pdf_path = FIG_DIR / "xpert_style_gene_distribution.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
