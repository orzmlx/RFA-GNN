import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def eval_pcc_mse(model, ctl, drug, cells, y_true, loss_mask, batch_size=32, max_eval=2048, drug_fp=None):
    if len(ctl) == 0:
        return {"mse": 0.0, "pcc": 0.0}
    if max_eval is not None and len(ctl) > int(max_eval):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ctl), size=int(max_eval), replace=False)
        ctl = ctl[idx]
        drug = drug[idx]
        cells = cells[idx]
        if drug_fp is not None:
            drug_fp = drug_fp[idx]
        y_true = y_true[idx]

    pred = model.predict([ctl, drug, cells, drug_fp], batch_size=batch_size, verbose=0)
    valid_indices = np.where(np.asarray(loss_mask)[0] > 0)[0]
    y_true_valid = y_true[:, valid_indices]
    pred_valid = pred[:, valid_indices]

    pcc_list = []
    for i in range(len(y_true_valid)):
        yt = y_true_valid[i]
        yp = pred_valid[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p, _ = pearsonr(yt, yp)
            pcc_list.append(p)
    avg_pcc = float(np.mean(pcc_list)) if pcc_list else 0.0
    mse = float(mean_squared_error(y_true_valid, pred_valid))
    return {"mse": mse, "pcc": avg_pcc}


class PCCCallback(keras.callbacks.Callback):
    def __init__(self, loss_mask, train_data, val_data, batch_size=32, max_eval=2048):
        super().__init__()
        self.loss_mask = loss_mask
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.max_eval = int(max_eval) if max_eval is not None else None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ctl_tr, drug_tr, cell_tr, fp_tr, y_tr = self.train_data
        ctl_va, drug_va, cell_va, fp_va, y_va = self.val_data
        tr = eval_pcc_mse(
            self.model,
            ctl_tr,
            drug_tr,
            cell_tr,
            y_tr,
            self.loss_mask,
            batch_size=self.batch_size,
            max_eval=self.max_eval,
            drug_fp=fp_tr,
        )
        va = eval_pcc_mse(
            self.model,
            ctl_va,
            drug_va,
            cell_va,
            y_va,
            self.loss_mask,
            batch_size=self.batch_size,
            max_eval=self.max_eval,
            drug_fp=fp_va,
        )
        logs["pcc"] = tr["pcc"]
        logs["val_pcc"] = va["pcc"]
        print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="ALL")
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--per_node_head", action="store_true", default=True)
    parser.add_argument("--no_cell_embedding", action="store_true", default=False)
    parser.add_argument("--no_drug_film", action="store_true", default=False)
    parser.add_argument("--split_mode", choices=["cold_drug", "cold_cell"], default="cold_drug")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--omnipath_consensus_only", action="store_true", default=False)
    parser.add_argument("--omnipath_is_directed_only", action="store_true", default=False)
    parser.add_argument("--include_mirna", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from data_loader import load_rfa_data
    from relational_graph_builder import build_relational_omnipath_graph
    from relational_gnn_tf import RelationalRegulatoryGNN, RelationalWrapperSparse

    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    mirna_path = os.path.join(root, "data/omnipath/omnipath_mirna_targets.csv")
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
    y_delta = np.asarray(data["y_delta"])
    X_drug = np.asarray(data["X_drug"])
    X_fp = data.get("X_fingerprint")
    X_fp = None if X_fp is None else np.asarray(X_fp, dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names_arr = np.asarray(data["cell_names"], dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    if int(args.max_samples) > 0 and len(X_ctl) > int(args.max_samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_ctl), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y_delta = y_delta[idx]
        X_drug = X_drug[idx]
        if X_fp is not None:
            X_fp = X_fp[idx]
        drug_ids = drug_ids[idx]
        cell_names_arr = cell_names_arr[idx]

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names_arr)
    num_cells = int(len(le.classes_))

    test_frac = float(args.test_frac)
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")

    np.random.seed(42)
    split_mode = str(args.split_mode)
    if split_mode == "cold_cell":
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise ValueError("cold_cell 需要至少 2 个细胞系；请设置 --cell_line ALL 并避免过小的 --max_samples")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        held_out = np.random.choice(unique_cells, n_test, replace=False)
        test_mask = np.isin(cell_idx, held_out)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(held_out)}/{len(unique_cells)}")
    else:
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise ValueError("cold_drug 需要至少 2 个药物；请避免过小的 --max_samples")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        held_out = np.random.choice(unique_drugs, n_test, replace=False)
        test_mask = np.isin(drug_ids, held_out)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(held_out)}/{len(unique_drugs)}")

    train_ctl = X_ctl[train_mask]
    train_trt = y_delta[train_mask]
    train_drug = X_drug[train_mask]
    train_cells = cell_idx[train_mask]
    train_fp = None if X_fp is None else X_fp[train_mask]

    test_ctl = X_ctl[test_mask]
    test_trt = y_delta[test_mask]
    test_drug = X_drug[test_mask]
    test_cells = cell_idx[test_mask]
    test_fp = None if X_fp is None else X_fp[test_mask]

    if train_fp is None:
        raise RuntimeError("X_fingerprint missing")

    graph = build_relational_omnipath_graph(
        target_genes=data["target_genes"],
        symbol_to_entrez=data.get("symbol_to_entrez"),
        tf_path=tf_path,
        ppi_path=ppi_path,
        mirna_path=(mirna_path if bool(args.include_mirna) else None),
        omnipath_consensus_only=bool(args.omnipath_consensus_only),
        omnipath_is_directed_only=bool(args.omnipath_is_directed_only),
        include_ppi_undirected=True,
    )
    rel_index = graph["relations"]
    rel_names = sorted(list(rel_index.keys()))
    print(f"Relations: {rel_names}")
    for r in rel_names:
        ei, _ = rel_index[r]
        print(f"{r}: {int(ei.shape[1])} edges")

    fp_dim = int(train_fp.shape[1])
    base = RelationalRegulatoryGNN(
        num_genes=int(train_ctl.shape[1]),
        num_cells=num_cells,
        fingerprint_dim=fp_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        use_cell_embedding=not bool(args.no_cell_embedding),
        use_drug_embedding=not bool(args.no_drug_film),
        relation_names=rel_names,
        per_node_embedding=bool(args.per_node_head),
    )
    model = RelationalWrapperSparse(base, rel_index)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", run_eagerly=False)
    cb = PCCCallback(
        loss_mask=loss_mask,
        train_data=(train_ctl, train_drug, train_cells, train_fp, train_trt),
        val_data=(test_ctl, test_drug, test_cells, test_fp, test_trt),
        batch_size=int(args.batch_size),
        max_eval=2048,
    )

    model.fit(
        x=[train_ctl, train_drug, train_cells, train_fp],
        y=train_trt,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        verbose=0,
        callbacks=[cb],
        validation_data=([test_ctl, test_drug, test_cells, test_fp], test_trt),
    )

    tr = eval_pcc_mse(model, train_ctl, train_drug, train_cells, train_trt, loss_mask, batch_size=int(args.batch_size), drug_fp=train_fp)
    te = eval_pcc_mse(model, test_ctl, test_drug, test_cells, test_trt, loss_mask, batch_size=int(args.batch_size), drug_fp=test_fp)
    print(f"Train | MSE: {tr['mse']:.4f} | Sample-wise PCC: {tr['pcc']:.4f}")
    print(f"Test  | MSE: {te['mse']:.4f} | Sample-wise PCC: {te['pcc']:.4f}")


if __name__ == "__main__":
    main()

