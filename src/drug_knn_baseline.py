import argparse
import os
import sys

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def tanimoto_sim_matrix(query_fp, ref_fp, eps=1e-9):
    query_fp = np.asarray(query_fp, dtype=np.float32)
    ref_fp = np.asarray(ref_fp, dtype=np.float32)
    dot = query_fp @ ref_fp.T
    qsum = np.sum(query_fp, axis=1, keepdims=True)
    rsum = np.sum(ref_fp, axis=1, keepdims=True).T
    denom = qsum + rsum - dot
    return dot / (denom + eps)


def samplewise_pcc_mse(y_true, y_pred, loss_mask):
    valid = np.where(np.asarray(loss_mask)[0] > 0)[0]
    yt = y_true[:, valid]
    yp = y_pred[:, valid]
    pcc_list = []
    for i in range(len(yt)):
        a = yt[i]
        b = yp[i]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            p, _ = pearsonr(a, b)
            pcc_list.append(p)
    pcc = float(np.mean(pcc_list)) if pcc_list else 0.0
    mse = float(mean_squared_error(yt, yp))
    return {"pcc": pcc, "mse": mse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="ALL")
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--split_mode", choices=["cold_drug", "cold_cell"], default="cold_drug")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min_sim", type=float, default=0.0)
    args = parser.parse_args()

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from data_loader import load_rfa_data

    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    cell_lines = args.cell_line
    if cell_lines is None:
        pass
    else:
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
        ctl_residual_pool_size=0,
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    X_ctl = np.asarray(data["X_ctl"])
    y = np.asarray(data["y_delta"], dtype=np.float32)
    fp = data.get("X_fingerprint")
    fp = None if fp is None else np.asarray(fp, dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    if fp is None:
        raise RuntimeError("X_fingerprint missing")

    if int(args.max_samples) > 0 and len(y) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(y), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y = y[idx]
        fp = fp[idx]
        drug_ids = drug_ids[idx]
        cell_names = cell_names[idx]

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names)
    n_cells = int(len(le.classes_))

    unique_drugs = np.unique(drug_ids)
    unique_cells = np.unique(cell_idx)

    rng = np.random.default_rng(int(args.seed))
    test_frac = float(args.test_frac)
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")

    split_mode = str(args.split_mode)
    if split_mode == "cold_cell":
        if len(unique_cells) < 2:
            raise ValueError("cold_cell 需要至少 2 个细胞系；请设置 --cell_line ALL 或增大 --max_samples")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        held_out_cells = rng.choice(unique_cells, size=n_test, replace=False)
        test_mask = np.isin(cell_idx, held_out_cells)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(held_out_cells)}/{len(unique_cells)}")
    else:
        if len(unique_drugs) < 2:
            raise ValueError("cold_drug 需要至少 2 个药物；请增大 --max_samples 或用 ALL")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        held_out_drugs = rng.choice(unique_drugs, size=n_test, replace=False)
        test_mask = np.isin(drug_ids, held_out_drugs)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(held_out_drugs)}/{len(unique_drugs)}")

    y_tr = y[train_mask]
    y_te = y[test_mask]
    fp_tr = fp[train_mask]
    fp_te = fp[test_mask]
    drug_tr = drug_ids[train_mask]
    drug_te = drug_ids[test_mask]
    cell_tr = cell_idx[train_mask]
    cell_te = cell_idx[test_mask]

    train_drugs_unique = np.unique(drug_tr)
    test_drugs_unique = np.unique(drug_te)
    print(f"Train drugs: {len(train_drugs_unique)} | Test drugs: {len(test_drugs_unique)}")
    print(f"Train samples: {len(y_tr)} | Test samples: {len(y_te)}")

    fp_dim = int(fp_tr.shape[1])
    train_drug2row = {d: i for i, d in enumerate(train_drugs_unique)}
    test_drug2row = {d: i for i, d in enumerate(test_drugs_unique)}

    train_drug_fp = np.zeros((len(train_drugs_unique), fp_dim), dtype=np.float32)
    for d in train_drugs_unique:
        train_drug_fp[train_drug2row[d]] = fp_tr[np.where(drug_tr == d)[0][0]]

    test_drug_fp = np.zeros((len(test_drugs_unique), fp_dim), dtype=np.float32)
    for d in test_drugs_unique:
        test_drug_fp[test_drug2row[d]] = fp_te[np.where(drug_te == d)[0][0]]

    sims = tanimoto_sim_matrix(test_drug_fp, train_drug_fp)
    k = int(args.k)
    if k <= 0:
        raise ValueError("--k must be > 0")
    k = min(k, sims.shape[1])
    min_sim = float(args.min_sim)
    alpha = float(args.alpha)

    drug_code_tr = np.array([train_drug2row[d] for d in drug_tr], dtype=np.int32)
    sums_drug = np.zeros((len(train_drugs_unique), y.shape[1]), dtype=np.float32)
    cnts_drug = np.zeros((len(train_drugs_unique),), dtype=np.int64)
    np.add.at(sums_drug, drug_code_tr, y_tr)
    np.add.at(cnts_drug, drug_code_tr, 1)
    drug_mean = sums_drug / np.maximum(cnts_drug[:, None], 1)

    sums_cell = np.zeros((n_cells, y.shape[1]), dtype=np.float32)
    cnts_cell = np.zeros((n_cells,), dtype=np.int64)
    np.add.at(sums_cell, cell_tr, y_tr)
    np.add.at(cnts_cell, cell_tr, 1)
    cell_mean = sums_cell / np.maximum(cnts_cell[:, None], 1)

    global_mean = np.mean(y_tr, axis=0, dtype=np.float32)

    drug_effect_pred = {}
    for d in test_drugs_unique:
        r = test_drug2row[d]
        sim_row = sims[r]
        top_idx = np.argpartition(-sim_row, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sim_row[top_idx])]
        w = sim_row[top_idx]
        keep = w >= min_sim
        top_idx = top_idx[keep]
        w = w[keep]
        if w.size == 0:
            drug_effect_pred[d] = np.zeros_like(global_mean, dtype=np.float32)
            continue
        if alpha != 1.0:
            w = np.power(w, alpha)
        wsum = float(np.sum(w))
        if wsum <= 0:
            drug_effect_pred[d] = np.zeros_like(global_mean, dtype=np.float32)
            continue
        neigh_mean = drug_mean[top_idx] - global_mean[None, :]
        eff = (w[:, None] * neigh_mean).sum(axis=0) / wsum
        drug_effect_pred[d] = eff.astype(np.float32)

    y_pred = np.zeros_like(y_te, dtype=np.float32)
    for i in range(len(y_te)):
        y_pred[i] = cell_mean[cell_te[i]] + drug_effect_pred[drug_te[i]]

    metrics = samplewise_pcc_mse(y_te, y_pred, loss_mask)
    print(f"KNN baseline | PCC: {metrics['pcc']:.4f} | MSE: {metrics['mse']:.4f}")


if __name__ == "__main__":
    main()

