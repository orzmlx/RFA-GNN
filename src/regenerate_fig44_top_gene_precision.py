import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def xpert_precision_k(y_true, y_pred, k=20, num_pos=100, num_neg=100):
    order_true = np.argsort(y_true, axis=1)
    order_pred = np.argsort(y_pred, axis=1)
    neg_test_set = order_true[:, :num_neg]
    pos_test_set = order_true[:, -num_pos:]
    neg_pred_set = order_pred[:, :k]
    pos_pred_set = order_pred[:, -k:]
    neg_scores = []
    pos_scores = []
    for i in range(len(order_true)):
        neg_scores.append(len(set(neg_test_set[i]).intersection(set(neg_pred_set[i]))) / k)
        pos_scores.append(len(set(pos_test_set[i]).intersection(set(pos_pred_set[i]))) / k)
    return float(np.mean(pos_scores)), float(np.mean(neg_scores))


def compute_metrics(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    y_true = np.asarray(data["y_true"], dtype=np.float32)
    y_pred = np.asarray(data["y_pred"], dtype=np.float32)
    return xpert_precision_k(y_true, y_pred, k=20, num_pos=100, num_neg=100)


def main():
    root = "/Users/liuxi/Desktop/RFA_GNN"
    split_order = ["warm", "cold_drug", "cold_cell"]
    split_labels = ["Warm", "Cold drug", "Cold cell"]
    model_files = {
        "DeepCOP": {
            "warm": os.path.join(root, "deepcop_res", "deepcop_978_allcells_pred.warm.npz"),
            "cold_drug": os.path.join(root, "deepcop_res", "deepcop_978_allcells_pred.cold_drug.npz"),
            "cold_cell": os.path.join(root, "deepcop_res", "deepcop_978_allcells_pred.cold_cell.npz"),
        },
        "GSNN": {
            "warm": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.warm.npz"),
            "cold_drug": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.cold_drug.npz"),
            "cold_cell": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.cold_cell.npz"),
        },
        "CAGNN": {
            "warm": os.path.join(root, "gat_res", "gat_cf_drug_loss_unique_trt_reuse_ctl.eval.warm.npz"),
            "cold_drug": os.path.join(root, "gat_res", "gat_cf_drug_loss_unique_trt_reuse_ctl.eval.cold_drug.npz"),
            "cold_cell": os.path.join(root, "gat_res", "gat_cf_drug_loss_unique_trt_reuse_ctl.eval.cold_cell.npz"),
        },
    }
    colors = {
        "DeepCOP": "#AFC8E8",
        "GSNN": "#F6C89A",
        "CAGNN": "#BFDDB8",
    }
    metrics = {model_name: {} for model_name in model_files}
    for model_name, split_map in model_files.items():
        for split_mode, npz_path in split_map.items():
            pos_p20, neg_p20 = compute_metrics(npz_path)
            metrics[model_name][split_mode] = {
                "pos": pos_p20,
                "neg": neg_p20,
            }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), dpi=220)
    x = np.arange(len(split_order))
    width = 0.22
    model_order = ["DeepCOP", "GSNN", "CAGNN"]
    for idx, model_name in enumerate(model_order):
        xpos = x + (idx - 1) * width
        pos_vals = [metrics[model_name][split_mode]["pos"] for split_mode in split_order]
        neg_vals = [metrics[model_name][split_mode]["neg"] for split_mode in split_order]
        axes[0].bar(xpos, pos_vals, width=width, color=colors[model_name], label=model_name)
        axes[1].bar(xpos, neg_vals, width=width, color=colors[model_name], label=model_name)
    axes[0].set_title("Pos P@20")
    axes[1].set_title("Neg P@20")
    axes[0].set_ylabel("Precision")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels)
        ax.set_ylim(0.0, 0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    out_path = os.path.join(root, "liuthesis_my", "figures", "top_gene_precision_results.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
