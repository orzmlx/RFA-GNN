import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def samplewise_pcc(y_true, y_pred, loss_mask):
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
    return float(np.mean(pcc_list)) if pcc_list else 0.0


def eval_pcc_mse(model, x_ctl, x_drugfeat, y_true, loss_mask, batch_size=256, max_eval=None):
    if len(x_ctl) == 0:
        return {"mse": 0.0, "pcc": 0.0}
    if max_eval is not None and len(x_ctl) > int(max_eval):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x_ctl), size=int(max_eval), replace=False)
        x_ctl = x_ctl[idx]
        x_drugfeat = x_drugfeat[idx]
        y_true = y_true[idx]
    pred = model.predict([x_ctl, x_drugfeat], batch_size=int(batch_size), verbose=0)
    pcc = samplewise_pcc(y_true, pred, loss_mask)
    valid = np.where(np.asarray(loss_mask)[0] > 0)[0]
    mse = float(mean_squared_error(y_true[:, valid], pred[:, valid]))
    return {"mse": mse, "pcc": pcc}


class PCCCallback(keras.callbacks.Callback):
    def __init__(self, loss_mask, train_data, val_data, batch_size=256, max_eval=20000):
        super().__init__()
        self.loss_mask = loss_mask
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.max_eval = int(max_eval) if max_eval is not None else None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ctl_tr, drug_tr, y_tr = self.train_data
        ctl_va, drug_va, y_va = self.val_data
        tr = eval_pcc_mse(self.model, ctl_tr, drug_tr, y_tr, self.loss_mask, batch_size=self.batch_size, max_eval=self.max_eval)
        va = eval_pcc_mse(self.model, ctl_va, drug_va, y_va, self.loss_mask, batch_size=self.batch_size, max_eval=self.max_eval)
        logs["pcc"] = tr["pcc"]
        logs["val_pcc"] = va["pcc"]
        print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f}")


def build_drug_features(data, drug_feature, include_cell_onehot, cell_names, drop_fp_has_target):
    x_drug_target = np.asarray(data["X_drug"], dtype=np.float32)
    x_fp = data.get("X_fingerprint")
    x_fp = None if x_fp is None else np.asarray(x_fp, dtype=np.float32)
    fp_table = data.get("drug_fp_table")
    fp_idx = data.get("drug_fp_idx")
    if x_fp is None and fp_table is not None and fp_idx is not None:
        fp_table = np.asarray(fp_table, dtype=np.float32)
        fp_idx = np.asarray(fp_idx, dtype=np.int32)
        x_fp = fp_table[fp_idx]

    if x_fp is not None and bool(drop_fp_has_target):
        if "drug_has_target" in data and x_fp.shape[1] == 2050:
            x_fp = x_fp[:, :-1]

    if str(drug_feature) == "target":
        x = x_drug_target
    elif str(drug_feature) == "fingerprint":
        if x_fp is None:
            raise RuntimeError("drug_feature=fingerprint 但 X_fingerprint/drug_fp_table 不存在")
        x = x_fp
    else:
        if x_fp is None:
            raise RuntimeError("drug_feature=target+fingerprint 但 X_fingerprint/drug_fp_table 不存在")
        x = np.concatenate([x_drug_target, x_fp], axis=1).astype(np.float32)

    if bool(include_cell_onehot):
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        x_cell = enc.fit_transform(np.asarray(cell_names, dtype=str).reshape(-1, 1)).astype(np.float32)
        x = np.concatenate([x, x_cell], axis=1).astype(np.float32)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    p.add_argument("--cell_line", default="ALL")
    p.add_argument("--use_landmark_genes", action="store_true", default=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--ctl_pair_k", type=int, default=3)
    p.add_argument("--split_mode", choices=["cold_drug", "cold_cell"], default="cold_drug")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drug_feature", choices=["target", "fingerprint", "target+fingerprint"], default="target")
    p.add_argument("--include_cell_onehot", action="store_true", default=True)
    p.add_argument("--drop_fp_has_target", action="store_true", default=False)
    p.add_argument("--no_residualize_target_by_cell", action="store_true", default=False)
    p.add_argument("--eval_drug_zero", action="store_true", default=False)
    p.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    p.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    p.add_argument("--eval_sanity_seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    root = str(args.root)
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    if os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))

    from data_loader import load_rfa_data
    from deepcop import DeepCOP

    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    cell_lines = args.cell_line
    if cell_lines is not None and isinstance(cell_lines, str):
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
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    x_ctl = np.asarray(data["X_ctl"], dtype=np.float32)
    y_delta = np.asarray(data["y_delta"], dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    if int(args.max_samples) > 0 and len(x_ctl) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(x_ctl), size=int(args.max_samples), replace=False)
        x_ctl = x_ctl[idx]
        y_delta = y_delta[idx]
        drug_ids = drug_ids[idx]
        cell_names = cell_names[idx]
        for k in ["X_drug", "X_fingerprint", "drug_fp_idx", "drug_has_target", "drug_has_fp"]:
            if k in data and data[k] is not None:
                arr = np.asarray(data[k])
                data[k] = arr[idx]

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names)
    num_cells = int(len(le.classes_))

    rng = np.random.default_rng(int(args.seed))
    if str(args.split_mode) == "cold_cell":
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise RuntimeError("cold_cell 需要至少 2 个细胞系")
        n_test = max(1, int(len(unique_cells) * float(args.test_frac)))
        n_test = min(n_test, len(unique_cells) - 1)
        held = rng.choice(unique_cells, size=n_test, replace=False)
        test_mask = np.isin(cell_idx, held)
        print(f"Split=cold_cell | Held-out cells: {len(held)}/{len(unique_cells)}")
    else:
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise RuntimeError("cold_drug 需要至少 2 个药物")
        n_test = max(1, int(len(unique_drugs) * float(args.test_frac)))
        n_test = min(n_test, len(unique_drugs) - 1)
        held = rng.choice(unique_drugs, size=n_test, replace=False)
        test_mask = np.isin(drug_ids, held)
        print(f"Split=cold_drug | Held-out drugs: {len(held)}/{len(unique_drugs)}")
        print(f"Train/Test drug overlap: {len(set(drug_ids[~test_mask]) & set(drug_ids[test_mask]))}")

    train_mask = ~test_mask

    x_drugfeat = build_drug_features(
        data,
        args.drug_feature,
        bool(args.include_cell_onehot),
        cell_names,
        bool(args.drop_fp_has_target),
    )

    train_ctl = x_ctl[train_mask]
    train_drug = x_drugfeat[train_mask]
    train_y_full = y_delta[train_mask]

    test_ctl = x_ctl[test_mask]
    test_drug = x_drugfeat[test_mask]
    test_y_full = y_delta[test_mask]

    residualize_target = not bool(args.no_residualize_target_by_cell)
    if residualize_target:
        sums = np.zeros((num_cells, train_y_full.shape[1]), dtype=np.float32)
        cnts = np.zeros((num_cells,), dtype=np.int64)
        np.add.at(sums, cell_idx[train_mask], train_y_full)
        np.add.at(cnts, cell_idx[train_mask], 1)
        mean = sums / np.maximum(cnts[:, None], 1)
        train_y = train_y_full - mean[cell_idx[train_mask]]
        test_y = test_y_full - mean[cell_idx[test_mask]]
    else:
        train_y = train_y_full
        test_y = test_y_full

    model = DeepCOP(
        num_genes=int(train_y.shape[1]),
        drug_dim=int(train_drug.shape[1]),
        dropout=float(args.dropout),
        use_residual=False,
        go_matrix=None,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(args.lr)), loss="mse", run_eagerly=False)

    cb = PCCCallback(loss_mask=loss_mask, train_data=(train_ctl, train_drug, train_y), val_data=(test_ctl, test_drug, test_y), batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))

    model.fit(
        [train_ctl, train_drug],
        train_y,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        callbacks=[cb],
        verbose=0,
    )

    train_metrics = eval_pcc_mse(model, train_ctl, train_drug, train_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
    test_metrics = eval_pcc_mse(model, test_ctl, test_drug, test_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
    print(f"Train | MSE: {train_metrics['mse']:.4f} | Sample-wise PCC: {train_metrics['pcc']:.4f}")
    print(f"Test  | MSE: {test_metrics['mse']:.4f} | Sample-wise PCC: {test_metrics['pcc']:.4f}")

    if bool(args.eval_drug_zero):
        if str(args.drug_feature) == "target":
            zero = np.zeros_like(test_drug, dtype=np.float32)
        elif str(args.drug_feature) == "fingerprint":
            zero = np.zeros_like(test_drug, dtype=np.float32)
        else:
            zero = np.zeros_like(test_drug, dtype=np.float32)
        m = eval_pcc_mse(model, test_ctl, zero, test_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
        print(f"Sanity(drug_zero) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")

    if bool(args.eval_drug_shuffle):
        rng = np.random.default_rng(int(args.eval_sanity_seed))
        perm = rng.permutation(len(test_ctl))
        shuf = test_drug[perm]
        m = eval_pcc_mse(model, test_ctl, shuf, test_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
        print(f"Sanity(drug_shuffle) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")


if __name__ == "__main__":
    main()
