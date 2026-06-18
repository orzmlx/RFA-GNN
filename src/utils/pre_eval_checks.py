import argparse
import json
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def resolve_root(root):
    candidate = str(root).strip()
    if os.path.exists(candidate):
        return candidate
    for fallback in ["/local/data1/liume102/rfa", "/local/data1/liume102/src", "/Users/liuxi/Desktop/RFA_GNN"]:
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError("No valid root directory found")


def samplewise_metrics(y_true, y_pred, loss_mask):
    valid_indices = np.where(np.asarray(loss_mask)[0] > 0)[0]
    yt = y_true[:, valid_indices]
    yp = y_pred[:, valid_indices]
    pcc = np.zeros((len(yt),), dtype=np.float32)
    mse = np.zeros((len(yt),), dtype=np.float32)
    for i in range(len(yt)):
        a = yt[i]
        b = yp[i]
        mse[i] = float(np.mean((a - b) ** 2))
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            pcc[i] = float(pearsonr(a, b)[0])
        else:
            pcc[i] = 0.0
    return pcc, mse


def fit_cell_mean_baseline(train_y, train_cells):
    train_y = np.asarray(train_y, dtype=np.float32)
    train_cells = np.asarray(train_cells, dtype=str)
    global_mean = np.mean(train_y, axis=0).astype(np.float32)
    cell_mean = {}
    for cell in np.unique(train_cells):
        mask = train_cells == cell
        cell_mean[str(cell)] = np.mean(train_y[mask], axis=0).astype(np.float32)
    return global_mean, cell_mean


def predict_cell_mean_baseline(test_cells, global_mean, cell_mean):
    test_cells = np.asarray(test_cells, dtype=str)
    preds = [cell_mean.get(str(cell), global_mean) for cell in test_cells]
    return np.asarray(preds, dtype=np.float32)


def fit_group_mean_baseline(train_y, train_keys):
    train_y = np.asarray(train_y, dtype=np.float32)
    train_keys = np.asarray(train_keys, dtype=str)
    global_mean = np.mean(train_y, axis=0).astype(np.float32)
    group_mean = {}
    for key in np.unique(train_keys):
        mask = train_keys == key
        group_mean[str(key)] = np.mean(train_y[mask], axis=0).astype(np.float32)
    return global_mean, group_mean


def predict_group_mean_baseline(test_keys, global_mean, group_mean):
    test_keys = np.asarray(test_keys, dtype=str)
    preds = [group_mean.get(str(key), global_mean) for key in test_keys]
    return np.asarray(preds, dtype=np.float32)


def predict_fallback_baseline(test_drugs, test_cells, global_mean, drug_mean, cell_mean):
    test_drugs = np.asarray(test_drugs, dtype=str)
    test_cells = np.asarray(test_cells, dtype=str)
    preds = []
    for drug, cell in zip(test_drugs, test_cells):
        if str(drug) in drug_mean:
            preds.append(drug_mean[str(drug)])
        elif str(cell) in cell_mean:
            preds.append(cell_mean[str(cell)])
        else:
            preds.append(global_mean)
    return np.asarray(preds, dtype=np.float32)


def compute_matched_average(train_data):
    trt_ids = np.asarray(train_data["trt_distil_ids"], dtype=str)
    cells = np.asarray(train_data["cell_names"], dtype=str)
    y = np.asarray(train_data["y_delta"], dtype=np.float32)
    order = {}
    sums = {}
    counts = {}
    for i, trt_id in enumerate(trt_ids):
        key = str(trt_id)
        if key not in order:
            order[key] = str(cells[i])
            sums[key] = np.array(y[i], dtype=np.float64)
            counts[key] = 1
        else:
            sums[key] += y[i]
            counts[key] += 1
    agg_cells = []
    agg_y = []
    for key in order:
        agg_cells.append(order[key])
        agg_y.append((sums[key] / float(counts[key])).astype(np.float32))
    return np.asarray(agg_cells, dtype=str), np.asarray(agg_y, dtype=np.float32)


def summarize_split(split_mode, train_data, test_data):
    train_drugs = np.asarray(train_data["drug_ids"], dtype=str)
    test_drugs = np.asarray(test_data["drug_ids"], dtype=str)
    train_cells = np.asarray(train_data["cell_names"], dtype=str)
    test_cells = np.asarray(test_data["cell_names"], dtype=str)
    return {
        "split_mode": split_mode,
        "train_samples": int(len(train_drugs)),
        "test_samples": int(len(test_drugs)),
        "train_unique_drugs": int(len(np.unique(train_drugs))),
        "test_unique_drugs": int(len(np.unique(test_drugs))),
        "train_unique_cells": int(len(np.unique(train_cells))),
        "test_unique_cells": int(len(np.unique(test_cells))),
        "shared_drugs": int(len(set(train_drugs.tolist()) & set(test_drugs.tolist()))),
        "shared_cells": int(len(set(train_cells.tolist()) & set(test_cells.tolist()))),
    }


def run_pca_plot(split_payloads, out_path, max_points=4000, seed=42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(split_payloads), figsize=(4.8 * len(split_payloads), 4.3), dpi=220)
    if len(split_payloads) == 1:
        axes = [axes]
    for ax, payload in zip(axes, split_payloads):
        train_y = np.asarray(payload["train"]["y_delta"], dtype=np.float32)
        test_y = np.asarray(payload["test"]["y_delta"], dtype=np.float32)
        n_train = len(train_y)
        n_test = len(test_y)
        train_take = min(max_points, n_train)
        test_take = min(max_points, n_test)
        train_idx = rng.choice(n_train, size=train_take, replace=False) if train_take < n_train else np.arange(n_train)
        test_idx = rng.choice(n_test, size=test_take, replace=False) if test_take < n_test else np.arange(n_test)
        train_sample = train_y[train_idx]
        test_sample = test_y[test_idx]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_sample)
        test_scaled = scaler.transform(test_sample)
        pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)
        ax.scatter(train_pca[:, 0], train_pca[:, 1], s=5, alpha=0.35, label="Train")
        ax.scatter(test_pca[:, 0], test_pca[:, 1], s=5, alpha=0.35, label="Test")
        ax.set_title(str(payload["split_mode"]))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--out_dir", default="/Users/liuxi/Desktop/RFA_GNN/tmp/pre_eval")
    parser.add_argument("--cell_line", default="ALL")
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--pairing_mode", choices=["multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"], default="multi_trt_multi_ctl")
    parser.add_argument("--ctl_pair_k", type=int, default=3)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = resolve_root(args.root)
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = types.ModuleType("tensorflow")

    from data_loader import load_rfa_data, prepare_split_data

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    cell_lines = args.cell_line
    if cell_lines is not None:
        s = str(cell_lines).strip()
        if s == "" or s.upper() in {"ALL", "NONE", "NULL"}:
            cell_lines = None

    data = load_rfa_data(
        ctl_path,
        trt_path,
        drug_target_path=drug_target_path,
        landmark_path=landmark_path,
        siginfo_path=siginfo_path,
        fingerprint_path=fingerprint_path,
        use_landmark_genes=bool(args.use_landmark_genes),
        full_gene_path=full_gene_path,
        cell_lines=cell_lines,
        ctl_residual_pool_size=int(args.ctl_pair_k),
        pairing_mode=str(args.pairing_mode),
    )

    split_modes = ["warm", "cold_drug", "cold_cell"]
    stats_rows = []
    baseline_rows = []
    split_payloads = []
    for split_mode in split_modes:
        train_data, test_data, _, _ = prepare_split_data(
            data=data,
            split_mode=split_mode,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
            train_pairing_mode=str(args.pairing_mode),
            train_ctl_pair_k=int(args.ctl_pair_k),
            test_pairing_mode="unique_trt_reuse_ctl",
        )
        stats_rows.append(summarize_split(split_mode, train_data, test_data))
        global_mean, cell_mean = fit_cell_mean_baseline(train_data["y_delta"], train_data["cell_names"])
        y_true = np.asarray(test_data["y_delta"], dtype=np.float32)
        global_pred = np.repeat(global_mean[None, :], len(y_true), axis=0)
        global_pcc, global_mse = samplewise_metrics(y_true, global_pred, data["loss_mask"])
        baseline_rows.append(
            {
                "split_mode": split_mode,
                "baseline": "global_mean",
                "test_mse": float(np.mean(global_mse)),
                "test_pcc": float(np.mean(global_pcc)),
                "train_cells_seen": int(len(np.unique(np.asarray(train_data["cell_names"], dtype=str)))),
                "test_cells_seen": int(len(np.unique(np.asarray(test_data["cell_names"], dtype=str)))),
            }
        )
        y_pred = predict_cell_mean_baseline(test_data["cell_names"], global_mean, cell_mean)
        sample_pcc, sample_mse = samplewise_metrics(y_true, y_pred, data["loss_mask"])
        baseline_rows.append(
            {
                "split_mode": split_mode,
                "baseline": "cell_mean_with_global_fallback",
                "test_mse": float(np.mean(sample_mse)),
                "test_pcc": float(np.mean(sample_pcc)),
                "train_cells_seen": int(len(np.unique(np.asarray(train_data["cell_names"], dtype=str)))),
                "test_cells_seen": int(len(np.unique(np.asarray(test_data["cell_names"], dtype=str)))),
            }
        )

        global_mean, drug_mean = fit_group_mean_baseline(train_data["y_delta"], train_data["drug_ids"])
        drug_pred = predict_group_mean_baseline(test_data["drug_ids"], global_mean, drug_mean)
        drug_pcc, drug_mse = samplewise_metrics(y_true, drug_pred, data["loss_mask"])
        baseline_rows.append(
            {
                "split_mode": split_mode,
                "baseline": "drug_mean_with_global_fallback",
                "test_mse": float(np.mean(drug_mse)),
                "test_pcc": float(np.mean(drug_pcc)),
                "train_cells_seen": int(len(np.unique(np.asarray(train_data["cell_names"], dtype=str)))),
                "test_cells_seen": int(len(np.unique(np.asarray(test_data["cell_names"], dtype=str)))),
            }
        )
        fallback_pred = predict_fallback_baseline(
            test_data["drug_ids"],
            test_data["cell_names"],
            global_mean,
            drug_mean,
            cell_mean,
        )
        fallback_sample_pcc, fallback_sample_mse = samplewise_metrics(y_true, fallback_pred, data["loss_mask"])
        baseline_rows.append(
            {
                "split_mode": split_mode,
                "baseline": "drug_then_cell_fallback",
                "test_mse": float(np.mean(fallback_sample_mse)),
                "test_pcc": float(np.mean(fallback_sample_pcc)),
                "train_cells_seen": int(len(np.unique(np.asarray(train_data["cell_names"], dtype=str)))),
                "test_cells_seen": int(len(np.unique(np.asarray(test_data["cell_names"], dtype=str)))),
            }
        )
        split_payloads.append({"split_mode": split_mode, "train": train_data, "test": test_data})

    stats_df = pd.DataFrame(stats_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    stats_csv = os.path.join(out_dir, "split_stats.csv")
    baseline_csv = os.path.join(out_dir, "simple_baseline_metrics.csv")
    pca_png = os.path.join(out_dir, "train_test_pca_by_split.png")
    summary_json = os.path.join(out_dir, "pre_eval_summary.json")

    stats_df.to_csv(stats_csv, index=False)
    baseline_df.to_csv(baseline_csv, index=False)
    run_pca_plot(split_payloads, pca_png, max_points=4000, seed=int(args.seed))

    summary = {
        "root": root,
        "out_dir": out_dir,
        "cell_line": "ALL" if cell_lines is None else cell_lines,
        "use_landmark_genes": bool(args.use_landmark_genes),
        "pairing_mode": str(args.pairing_mode),
        "test_frac": float(args.test_frac),
        "stats_csv": stats_csv,
        "baseline_csv": baseline_csv,
        "pca_png": pca_png,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(stats_df.to_string(index=False))
    print()
    print(baseline_df.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
