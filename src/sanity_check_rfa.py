import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SplitResult:
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_drugs: np.ndarray
    test_drugs: np.ndarray
    overlap_count: int


def _samplewise_pcc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    num = (yt * yp).sum(axis=1)
    den = np.sqrt((yt * yt).sum(axis=1) * (yp * yp).sum(axis=1)) + eps
    return float((num / den).mean())


def _cold_drug_split(drug_ids: np.ndarray, test_ratio: float = 0.2, seed: int = 42) -> SplitResult:
    drug_ids = np.asarray(drug_ids, dtype=str)
    unique_drugs = np.unique(drug_ids)
    rng = np.random.default_rng(int(seed))
    n_test = max(1, int(round(len(unique_drugs) * float(test_ratio))))
    test_drugs = rng.choice(unique_drugs, size=n_test, replace=False)
    test_mask = np.isin(drug_ids, test_drugs)
    train_mask = ~test_mask
    train_drugs = np.unique(drug_ids[train_mask])
    overlap = np.intersect1d(train_drugs, np.unique(test_drugs))
    return SplitResult(
        train_mask=train_mask,
        test_mask=test_mask,
        train_drugs=train_drugs,
        test_drugs=np.unique(test_drugs),
        overlap_count=int(len(overlap)),
    )


def _feature_coverage_stats(data: Dict, split: SplitResult) -> Dict[str, float]:
    out: Dict[str, float] = {}
    X_drug = np.asarray(data["X_drug"])
    out["sample_has_target_rate_all"] = float((X_drug.sum(axis=1) > 0).mean())
    out["sample_has_target_rate_test"] = float((X_drug[split.test_mask].sum(axis=1) > 0).mean())

    X_fp = data.get("X_fingerprint")
    if X_fp is None:
        out["fp_enabled"] = 0.0
        out["sample_has_fp_rate_all"] = 0.0
        out["sample_has_fp_rate_test"] = 0.0
    else:
        X_fp = np.asarray(X_fp)
        out["fp_enabled"] = 1.0
        out["fp_dim"] = float(X_fp.shape[1])
        mask = X_fp[:, -1]
        out["sample_has_fp_rate_all"] = float(mask.mean())
        out["sample_has_fp_rate_test"] = float(mask[split.test_mask].mean())

    out["sample_has_any_druginfo_rate_all"] = float(((X_drug.sum(axis=1) > 0) | ((data.get("X_fingerprint") is not None) and (np.asarray(data["X_fingerprint"])[:, -1] > 0))).mean())
    out["sample_has_any_druginfo_rate_test"] = float(((X_drug[split.test_mask].sum(axis=1) > 0) | ((data.get("X_fingerprint") is not None) and (np.asarray(data["X_fingerprint"])[split.test_mask, -1] > 0))).mean())
    return out


def _mean_baselines(data: Dict, split: SplitResult) -> Dict[str, float]:
    y = np.asarray(data["y_delta"], dtype=np.float32)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    y_train = y[split.train_mask]
    y_test = y[split.test_mask]
    cells_train = cell_names[split.train_mask]
    cells_test = cell_names[split.test_mask]

    global_mean = y_train.mean(axis=0, keepdims=True)
    pred_global = np.repeat(global_mean, repeats=len(y_test), axis=0)
    pcc_global = _samplewise_pcc(y_test, pred_global)

    cell_to_mean: Dict[str, np.ndarray] = {}
    for c in np.unique(cells_train):
        cell_to_mean[c] = y_train[cells_train == c].mean(axis=0)
    fallback = global_mean[0]
    pred_cell = np.stack([cell_to_mean.get(c, fallback) for c in cells_test], axis=0)
    pcc_cell = _samplewise_pcc(y_test, pred_cell)

    return {
        "baseline_global_mean_pcc": pcc_global,
        "baseline_cell_mean_pcc": pcc_cell,
    }


def _graph_stats(root: str, data: Dict, tf_path: str, ppi_path: str, string_path: Optional[str]) -> Dict[str, float]:
    sys.path.insert(0, os.path.join(root, "src"))
    from data_loader import build_combined_gnn  # noqa: WPS433

    adj, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        string_path=string_path,
        target_genes=data["target_genes"],
        confid_threshold=0.9,
        directed=False,
        symbol_to_entrez=data.get("symbol_to_entrez"),
    )
    n = int(adj.shape[0])
    if edge_index is None:
        return {"graph_nodes": float(n), "graph_edges": float(0), "graph_isolated_nodes": float(n)}
    edge_index = np.asarray(edge_index)
    if edge_index.size == 0:
        return {"graph_nodes": float(n), "graph_edges": float(0), "graph_isolated_nodes": float(n)}

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    deg = np.bincount(np.concatenate([src, dst]), minlength=n)
    isolated = int((deg == 0).sum())
    return {
        "graph_nodes": float(n),
        "graph_edges": float(edge_index.shape[1]),
        "graph_isolated_nodes": float(isolated),
        "graph_mean_degree": float(deg.mean()),
        "graph_median_degree": float(np.median(deg)),
        "graph_max_degree": float(deg.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--cell_line", default="ALL")
    parser.add_argument("--filter_time", type=int, default=24)
    parser.add_argument("--filter_dose", type=float, default=10.0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--ctl_aug_pool", type=int, default=0)
    parser.add_argument("--graph_stats", action="store_true", default=False)
    args = parser.parse_args()

    root = args.root
    sys.path.insert(0, os.path.join(root, "src"))
    from data_loader import load_rfa_data  # noqa: WPS433

    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    if args.cell_line.upper() == "ALL":
        cell_lines = None
    else:
        cell_lines = [c.strip() for c in args.cell_line.split(",") if c.strip()]

    data = load_rfa_data(
        ctl_path,
        trt_path,
        drug_target_path=drug_target_path,
        siginfo_path=siginfo_path,
        landmark_path=landmark_path,
        fingerprint_path=fingerprint_path,
        use_landmark_genes=bool(args.use_landmark_genes),
        full_gene_path=full_gene_path,
        filter_time=int(args.filter_time) if args.filter_time else None,
        filter_dose=float(args.filter_dose) if args.filter_dose else None,
        cell_lines=cell_lines,
        ctl_residual_pool_size=int(args.ctl_aug_pool),
    )

    X_ctl = np.asarray(data["X_ctl"])
    y = np.asarray(data["y_delta"])
    drug_ids = np.asarray(data["drug_ids"], dtype=str)

    if int(args.max_samples) > 0 and len(drug_ids) > int(args.max_samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(drug_ids), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y = y[idx]
        drug_ids = drug_ids[idx]
        data = dict(data)
        data["X_ctl"] = X_ctl
        data["y_delta"] = y
        data["drug_ids"] = drug_ids
        data["cell_names"] = np.asarray(data["cell_names"], dtype=str)[idx].tolist()
        if data.get("X_fingerprint") is not None:
            data["X_fingerprint"] = np.asarray(data["X_fingerprint"])[idx]
        data["X_drug"] = np.asarray(data["X_drug"])[idx]

    split = _cold_drug_split(drug_ids, test_ratio=float(args.test_ratio), seed=int(args.seed))

    print("=== Split Sanity ===")
    print("samples:", len(drug_ids))
    print("unique_drugs:", len(np.unique(drug_ids)))
    print("train_drugs:", len(split.train_drugs), "test_drugs:", len(split.test_drugs))
    print("overlap_count:", split.overlap_count)
    print("train_samples:", int(split.train_mask.sum()), "test_samples:", int(split.test_mask.sum()))

    print("\n=== Coverage ===")
    cov = _feature_coverage_stats(data, split)
    for k in sorted(cov.keys()):
        print(f"{k}: {cov[k]}")

    print("\n=== Baselines (cold-drug test set) ===")
    b = _mean_baselines(data, split)
    for k in sorted(b.keys()):
        print(f"{k}: {b[k]}")

    if args.graph_stats:
        print("\n=== Graph Stats ===")
        tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
        ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
        string_path = None
        gs = _graph_stats(root, data, tf_path=tf_path, ppi_path=ppi_path, string_path=string_path)
        for k in sorted(gs.keys()):
            print(f"{k}: {gs[k]}")


if __name__ == "__main__":
    main()
