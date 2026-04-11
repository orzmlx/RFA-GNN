import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def load_gene_symbol_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        genes = json.load(f)
    return {str(item["entrez_id"]): item["gene_symbol"] for item in genes}


def load_split_payloads(root, split_mode):
    path_map = {
        "cold_cell": {
            "DeepCOP": os.path.join(root, "deepcop_res", "deepcop_978_allcells_pred.cold_cell.npz"),
            "GSNN": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.cold_cell.npz"),
            "CAGNN": os.path.join(root, "gat_res", "gat_cf_drug_loss_unique_trt_reuse_ctl.eval.cold_cell.npz"),
        },
        "cold_drug": {
            "DeepCOP": os.path.join(root, "deepcop_res", "deepcop_978_allcells_pred.cold_drug.npz"),
            "GSNN": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.cold_drug.npz"),
            "CAGNN": os.path.join(root, "gat_res", "gat_cf_drug_loss_unique_trt_reuse_ctl.eval.cold_drug.npz"),
        },
    }
    payloads = {}
    for model_name, npz_path in path_map[split_mode].items():
        payloads[model_name] = np.load(npz_path, allow_pickle=True)
    return payloads


def resolve_gene_order(payloads):
    cag = payloads["CAGNN"]
    if "target_genes" in cag.files:
        return [str(x) for x in cag["target_genes"]]
    raise ValueError("CAGNN npz does not contain target_genes; gene order cannot be resolved safely.")


def select_top_gene_indices(y_true, top_k=10):
    mean_abs = np.mean(np.abs(y_true), axis=0)
    return np.argsort(-mean_abs)[:top_k]


def make_boxplot(ax, payloads, split_title, gene_symbols):
    gene_order = resolve_gene_order(payloads)
    ref_true = np.asarray(payloads["CAGNN"]["y_true"], dtype=np.float32)
    top_idx = select_top_gene_indices(ref_true, top_k=10)
    labels = [gene_symbols.get(gene_order[idx], gene_order[idx]) for idx in top_idx]
    series_order = ["Ground truth", "DeepCOP", "GSNN", "CAGNN"]
    colors = {
        "Ground truth": "#d9d55b",
        "DeepCOP": "#d8ccb1",
        "DeepCOP_edge": "#a79a7a",
        "GSNN": "#b8c3a0",
        "GSNN_edge": "#87906f",
        "CAGNN": "#a98bc6",
        "CAGNN_edge": "#80639e",
    }
    model_fill = {
        "Ground truth": colors["Ground truth"],
        "DeepCOP": colors["DeepCOP"],
        "GSNN": colors["GSNN"],
        "CAGNN": colors["CAGNN"],
    }
    model_edge = {
        "Ground truth": "#9f9a37",
        "DeepCOP": colors["DeepCOP_edge"],
        "GSNN": colors["GSNN_edge"],
        "CAGNN": colors["CAGNN_edge"],
    }
    base_positions = np.arange(len(top_idx)) * 1.4
    offsets = [-0.33, -0.11, 0.11, 0.33]
    width = 0.18
    flierprops = dict(
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgecolor="#666666",
        markeredgewidth=0.5,
        alpha=0.9,
    )
    medianprops = dict(color="#444444", linewidth=0.9)
    whiskerprops = dict(color="#888888", linewidth=0.8)
    capprops = dict(color="#888888", linewidth=0.8)
    boxprops = dict(linewidth=0.8)
    for gene_pos, gene_idx in zip(base_positions, top_idx):
        data_map = {
            "Ground truth": ref_true[:, gene_idx],
            "DeepCOP": np.asarray(payloads["DeepCOP"]["y_pred"], dtype=np.float32)[:, gene_idx],
            "GSNN": np.asarray(payloads["GSNN"]["y_pred"], dtype=np.float32)[:, gene_idx],
            "CAGNN": np.asarray(payloads["CAGNN"]["y_pred"], dtype=np.float32)[:, gene_idx],
        }
        for offset, series_name in zip(offsets, series_order):
            bp = ax.boxplot(
                [data_map[series_name]],
                positions=[gene_pos + offset],
                widths=width,
                patch_artist=True,
                showfliers=True,
                flierprops=flierprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                boxprops=boxprops,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(model_fill[series_name])
                patch.set_edgecolor(model_edge[series_name])
                patch.set_alpha(0.95)
    ax.set_title(split_title, pad=8)
    ax.set_ylabel(r"$x_{\mathrm{deg}}$")
    ax.set_xticks(base_positions)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlim(base_positions[0] - 0.6, base_positions[-1] + 0.6)
    y_all = [ref_true[:, idx] for idx in top_idx]
    for model_name in ["DeepCOP", "GSNN", "CAGNN"]:
        pred = np.asarray(payloads[model_name]["y_pred"], dtype=np.float32)
        y_all.extend([pred[:, idx] for idx in top_idx])
    y_min = min(float(np.min(x)) for x in y_all)
    y_max = max(float(np.max(x)) for x in y_all)
    pad = 0.05 * max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - pad, y_max + pad)


def main():
    root = "/Users/liuxi/Desktop/RFA_GNN"
    gene_symbols = load_gene_symbol_map(os.path.join(root, "data", "landmark_genes.json"))
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.6), dpi=220, sharex=False)
    split_specs = [
        ("cold_cell", r"$x_{\mathrm{deg}}$ distribution of top 10 responsive genes in cold-cell scenario"),
        ("cold_drug", r"$x_{\mathrm{deg}}$ distribution of top 10 responsive genes in cold-drug scenario"),
    ]
    for ax, (split_mode, title) in zip(axes, split_specs):
        payloads = load_split_payloads(root, split_mode)
        make_boxplot(ax, payloads, title, gene_symbols)
    legend_handles = [
        Patch(facecolor="#d9d55b", edgecolor="#9f9a37", label="Ground truth"),
        Patch(facecolor="#d8ccb1", edgecolor="#a79a7a", label="DeepCOP"),
        Patch(facecolor="#b8c3a0", edgecolor="#87906f", label="GSNN"),
        Patch(facecolor="#a98bc6", edgecolor="#80639e", label="CAGNN"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.4,
        handlelength=1.0,
    )
    fig.subplots_adjust(top=0.9, hspace=0.35, left=0.07, right=0.99, bottom=0.11)
    out_path = os.path.join(root, "liuthesis_my", "figures", "xpert_style_gene_distribution.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
