import argparse
import json
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


def predict_full(model, x_ctl, x_drugfeat, y_true, loss_mask, batch_size=256):
    if len(x_ctl) == 0:
        return {
            "mse": 0.0,
            "pcc": 0.0,
            "sample_pcc": np.zeros((0,), dtype=np.float32),
            "y_true": np.zeros((0, 0), dtype=np.float32),
            "y_pred": np.zeros((0, 0), dtype=np.float32),
        }
    pred = model.predict([x_ctl, x_drugfeat], batch_size=int(batch_size), verbose=0)
    valid = np.where(np.asarray(loss_mask)[0] > 0)[0]
    sample_pcc_list = []
    yt = y_true[:, valid]
    yp = pred[:, valid]
    for i in range(len(yt)):
        a = yt[i]
        b = yp[i]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            p, _ = pearsonr(a, b)
            sample_pcc_list.append(float(p))
        else:
            sample_pcc_list.append(0.0)
    sample_pcc = np.asarray(sample_pcc_list, dtype=np.float32)
    mse = float(mean_squared_error(yt, yp))
    return {
        "mse": mse,
        "pcc": float(np.mean(sample_pcc)) if len(sample_pcc) > 0 else 0.0,
        "sample_pcc": sample_pcc,
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(pred, dtype=np.float32),
    }


def build_split_masks(split_mode, drug_ids, cell_idx, test_frac, seed=42):
    split_mode = str(split_mode).strip()
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")
    rng = np.random.default_rng(int(seed))
    n = len(drug_ids)
    if split_mode == "warm":
        if n < 2:
            raise RuntimeError("warm 需要至少 2 个样本")
        n_test = max(1, int(n * test_frac))
        n_test = min(n_test, n - 1)
        held = rng.choice(np.arange(n), size=n_test, replace=False)
        test_mask = np.zeros((n,), dtype=bool)
        test_mask[held] = True
        train_mask = ~test_mask
        print(f"Split=warm | Held-out samples: {int(np.sum(test_mask))}/{n}")
        return train_mask, test_mask
    if split_mode == "cold_cell":
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise RuntimeError("cold_cell 需要至少 2 个细胞系")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        held = rng.choice(unique_cells, size=n_test, replace=False)
        test_mask = np.isin(cell_idx, held)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(held)}/{len(unique_cells)}")
        return train_mask, test_mask
    if split_mode == "cold_drug":
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise RuntimeError("cold_drug 需要至少 2 个药物")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        held = rng.choice(unique_drugs, size=n_test, replace=False)
        test_mask = np.isin(drug_ids, held)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(held)}/{len(unique_drugs)}")
        print(f"Train/Test drug overlap: {len(set(drug_ids[train_mask]) & set(drug_ids[test_mask]))}")
        return train_mask, test_mask
    raise ValueError(f"未知 split_mode: {split_mode}")


def parse_split_modes(raw, fallback):
    valid = {"warm", "cold_drug", "cold_cell"}
    s = str(raw).strip()
    if s == "":
        return [str(fallback)]
    modes = [t.strip() for t in s.split(",") if t.strip() != ""]
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"--split_modes 包含不支持的值: {bad}")
    seen = []
    for m in modes:
        if m not in seen:
            seen.append(m)
    return seen


def append_split_suffix(path, split_mode):
    s = str(path).strip()
    if s == "":
        return ""
    suffix = f".{str(split_mode).strip()}"
    stem, ext = os.path.splitext(s)
    return f"{stem}{suffix}{ext}"


def save_predictions_npz(npz_path, split_mode, y_true, y_pred, sample_pcc=None, drug_ids=None, cell_names=None, trt_distil_ids=None):
    out_dir = os.path.dirname(npz_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "split_mode": np.asarray(str(split_mode)),
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(y_pred, dtype=np.float32),
    }
    if sample_pcc is not None:
        payload["sample_pcc"] = np.asarray(sample_pcc, dtype=np.float32)
    if drug_ids is not None:
        payload["drug_ids"] = np.asarray(drug_ids, dtype=str)
    if cell_names is not None:
        payload["cell_names"] = np.asarray(cell_names, dtype=str)
    if trt_distil_ids is not None:
        payload["trt_distil_ids"] = np.asarray(trt_distil_ids, dtype=str)
    np.savez_compressed(npz_path, **payload)
    print(f"Saved predictions to: {npz_path}")


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


def build_drug_features(data, drug_feature, include_cell_onehot, cell_names, drop_fp_has_target, fit_cell_names=None):
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
        fit_names = np.asarray(cell_names if fit_cell_names is None else fit_cell_names, dtype=str)
        enc.fit(fit_names.reshape(-1, 1))
        x_cell = enc.transform(np.asarray(cell_names, dtype=str).reshape(-1, 1)).astype(np.float32)
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
    p.add_argument("--pairing_mode", choices=["multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"], default="multi_trt_multi_ctl")
    p.add_argument("--split_mode", choices=["warm", "cold_drug", "cold_cell"], default="cold_drug")
    p.add_argument("--split_modes", default="")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drug_feature", choices=["target", "fingerprint", "target+fingerprint"], default="target")
    p.add_argument("--include_cell_onehot", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--drop_fp_has_target", action="store_true", default=False)
    p.add_argument("--no_residualize_target_by_cell", action="store_true", default=False)
    p.add_argument("--eval_drug_zero", action="store_true", default=False)
    p.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    p.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    p.add_argument("--eval_sanity_seed", type=int, default=0)
    p.add_argument("--save_json", default="")
    p.add_argument("--save_pred_prefix", default="")
    p.add_argument("--use_go_matrix", action="store_true", default=False)
    p.add_argument("--go_fingerprint_path", default="")
    p.add_argument("--original_architecture", action="store_true", default=False)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    root = str(args.root)
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    if os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))

    from data_loader import load_rfa_data, load_go_fingerprints, subset_anchor_data, build_scheme_a_split_data
    from deepcop import DeepCOP

    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")
    default_go_path = os.path.join(root, "DeepCOP", "Data", "go_fingerprints.csv")

    cell_lines = args.cell_line
    if cell_lines is not None and isinstance(cell_lines, str):
        s = str(cell_lines).strip()
        if s == "" or s.upper() in {"ALL", "NONE", "NULL"}:
            cell_lines = None

    go_fingerprint_path = str(args.go_fingerprint_path).strip()
    if go_fingerprint_path == "" and bool(args.use_go_matrix):
        go_fingerprint_path = default_go_path

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
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
    anchor_cell_names = np.asarray(data["anchor_cell_names"], dtype=str)
    anchor_trt_distil_ids = np.asarray(data.get("anchor_trt_distil_ids", [""] * len(anchor_drug_ids)), dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)
    go_matrix = None
    if bool(args.use_go_matrix):
        go_matrix = load_go_fingerprints(go_fingerprint_path, data["target_genes"])
        if go_matrix is None:
            raise RuntimeError(f"加载 GO 特征失败: {go_fingerprint_path}")
        print(f"Using GO matrix: {go_matrix.shape}")

    if int(args.max_samples) > 0 and len(anchor_drug_ids) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(anchor_drug_ids), size=int(args.max_samples), replace=False)
        data = subset_anchor_data(data, idx)
        anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
        anchor_cell_names = np.asarray(data["anchor_cell_names"], dtype=str)
        anchor_trt_distil_ids = np.asarray(data.get("anchor_trt_distil_ids", [""] * len(anchor_drug_ids)), dtype=str)

    le = LabelEncoder()
    cell_idx = le.fit_transform(anchor_cell_names)
    num_cells = int(len(le.classes_))
    results = []
    split_modes = parse_split_modes(args.split_modes, args.split_mode)
    for split_mode in split_modes:
        tf.keras.backend.clear_session()
        tf.random.set_seed(int(args.seed))
        print(f"\n===== Running split: {split_mode} =====")
        train_data, test_data, train_mask, test_mask = build_scheme_a_split_data(
            data=data,
            split_mode=split_mode,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
            train_pairing_mode=str(args.pairing_mode),
            train_ctl_pair_k=int(args.ctl_pair_k),
            test_pairing_mode="unique_trt_reuse_ctl",
        )

        train_ctl = np.asarray(train_data["X_ctl"], dtype=np.float32)
        train_y_full = np.asarray(train_data["y_delta"], dtype=np.float32)
        train_cell_names = np.asarray(train_data["cell_names"], dtype=str)
        test_ctl = np.asarray(test_data["X_ctl"], dtype=np.float32)
        test_y_full = np.asarray(test_data["y_delta"], dtype=np.float32)
        test_drug_ids = np.asarray(test_data["drug_ids"], dtype=str)
        test_cell_names = np.asarray(test_data["cell_names"], dtype=str)
        test_trt_distil_ids = np.asarray(test_data["trt_distil_ids"], dtype=str)

        fit_cell_names = None
        if bool(args.include_cell_onehot):
            fit_cell_names = np.concatenate([train_cell_names, test_cell_names], axis=0)
        train_drug = build_drug_features(
            train_data,
            args.drug_feature,
            bool(args.include_cell_onehot),
            train_cell_names,
            bool(args.drop_fp_has_target),
            fit_cell_names=fit_cell_names,
        )
        test_drug = build_drug_features(
            test_data,
            args.drug_feature,
            bool(args.include_cell_onehot),
            test_cell_names,
            bool(args.drop_fp_has_target),
            fit_cell_names=fit_cell_names,
        )

        residualize_target = not bool(args.no_residualize_target_by_cell)
        if residualize_target:
            sums = np.zeros((num_cells, train_y_full.shape[1]), dtype=np.float32)
            cnts = np.zeros((num_cells,), dtype=np.int64)
            train_cells = le.transform(train_cell_names)
            test_cells = le.transform(test_cell_names)
            np.add.at(sums, train_cells, train_y_full)
            np.add.at(cnts, train_cells, 1)
            mean = sums / np.maximum(cnts[:, None], 1)
            train_y = train_y_full - mean[train_cells]
            test_y = test_y_full - mean[test_cells]
        else:
            train_y = train_y_full
            test_y = test_y_full

        model = DeepCOP(
            num_genes=int(train_y.shape[1]),
            drug_dim=int(train_drug.shape[1]),
            dropout=float(args.dropout),
            use_residual=False,
            go_matrix=go_matrix,
            original_architecture=bool(args.original_architecture),
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(args.lr)), loss="mse", run_eagerly=False)

        cb = PCCCallback(
            loss_mask=loss_mask,
            train_data=(train_ctl, train_drug, train_y),
            val_data=(test_ctl, test_drug, test_y),
            batch_size=int(args.batch_size),
            max_eval=int(args.eval_sanity_max_eval),
        )

        model.fit(
            [train_ctl, train_drug],
            train_y,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            callbacks=[cb],
            verbose=0,
        )

        train_metrics = eval_pcc_mse(model, train_ctl, train_drug, train_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
        test_full = predict_full(model, test_ctl, test_drug, test_y, loss_mask, batch_size=int(args.batch_size))
        test_metrics = {"mse": test_full["mse"], "pcc": test_full["pcc"]}
        print(f"Train | MSE: {train_metrics['mse']:.4f} | Sample-wise PCC: {train_metrics['pcc']:.4f}")
        print(f"Test  | MSE: {test_metrics['mse']:.4f} | Sample-wise PCC: {test_metrics['pcc']:.4f}")

        pred_prefix = str(args.save_pred_prefix).strip()
        if pred_prefix == "" and str(args.save_json).strip() != "":
            pred_prefix = os.path.splitext(str(args.save_json).strip())[0] + ".pred"
        pred_npz_path = None
        if pred_prefix != "":
            pred_npz_path = f"{pred_prefix}.{split_mode}.npz"
            save_predictions_npz(
                pred_npz_path,
                split_mode=split_mode,
                y_true=test_full["y_true"],
                y_pred=test_full["y_pred"],
                sample_pcc=test_full["sample_pcc"],
                drug_ids=test_drug_ids,
                cell_names=test_cell_names,
                trt_distil_ids=test_trt_distil_ids,
            )

        sanity_drug_zero_metrics = None
        if bool(args.eval_drug_zero):
            zero = np.zeros_like(test_drug, dtype=np.float32)
            sanity_drug_zero_metrics = eval_pcc_mse(model, test_ctl, zero, test_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
            m = sanity_drug_zero_metrics
            print(f"Sanity(drug_zero) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")

        sanity_drug_shuffle_metrics = None
        if bool(args.eval_drug_shuffle):
            rng = np.random.default_rng(int(args.eval_sanity_seed))
            perm = rng.permutation(len(test_ctl))
            shuf = test_drug[perm]
            sanity_drug_shuffle_metrics = eval_pcc_mse(model, test_ctl, shuf, test_y, loss_mask, batch_size=int(args.batch_size), max_eval=int(args.eval_sanity_max_eval))
            m = sanity_drug_shuffle_metrics
            print(f"Sanity(drug_shuffle) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")

        results.append(
            {
                "split_mode": str(split_mode),
                "train_n": int(len(train_ctl)),
                "test_n": int(len(test_ctl)),
                "train_anchor_n": int(np.sum(train_mask)),
                "test_anchor_n": int(np.sum(test_mask)),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "pred_npz": pred_npz_path,
                "use_go_matrix": bool(args.use_go_matrix),
                "go_fingerprint_path": go_fingerprint_path if bool(args.use_go_matrix) else "",
                "original_architecture": bool(args.original_architecture),
                "sanity_drug_zero_metrics": sanity_drug_zero_metrics,
                "sanity_drug_shuffle_metrics": sanity_drug_shuffle_metrics,
            }
        )

    if len(results) > 1:
        print("\n===== Summary =====")
        for r in results:
            print(
                f"{r['split_mode']}: "
                f"train_n={r['train_n']} test_n={r['test_n']} | "
                f"test_MSE={r['test_metrics']['mse']:.4f} | "
                f"test_PCC={r['test_metrics']['pcc']:.4f}"
            )

    if str(args.save_json).strip() != "":
        out_path = str(args.save_json).strip()
        out_dir = os.path.dirname(out_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
