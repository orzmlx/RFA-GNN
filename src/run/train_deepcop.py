import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from data_loader import load_rfa_data, subset_anchor_data, prepare_split_data
from deepcop import DeepCOP
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from train_common import (
    parse_split_modes,
    samplewise_masked_metrics,
    samplewise_pcc,
    save_predictions_npz,
    split_mean_logvar,
)
from train_tf_common import GenericPCCCallback
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
    pred_mean, _ = split_mean_logvar(pred)
    valid = np.where(np.asarray(loss_mask)[0] > 0)[0]
    pcc = float(np.mean(samplewise_pcc(y_true, pred_mean, valid_indices=valid))) if len(y_true) else 0.0
    mse = float(np.mean((y_true[:, valid] - pred_mean[:, valid]) ** 2))
    return {"mse": mse, "pcc": pcc}

def predict_full(model, x_ctl, x_drugfeat, y_true, loss_mask, batch_size=256):
    if len(x_ctl) == 0:
        return {
            "mse": 0.0,
            "pcc": 0.0,
            "sample_pcc": np.zeros((0,), dtype=np.float32),
            "y_true": np.zeros((0, 0), dtype=np.float32),
            "y_pred": np.zeros((0, 0), dtype=np.float32),
            "y_logvar": np.zeros((0, 0), dtype=np.float32),
        }
    pred = model.predict([x_ctl, x_drugfeat], batch_size=int(batch_size), verbose=0)
    pred_mean, pred_logvar = split_mean_logvar(pred)
    sample_pcc, sample_mse = samplewise_masked_metrics(y_true, pred_mean, loss_mask)
    valid = np.where(np.asarray(loss_mask)[0] > 0)[0]
    yt = y_true[:, valid]
    yp = pred_mean[:, valid]
    mse = float(np.mean((yt - yp) ** 2))
    return {
        "mse": mse,
        "pcc": float(np.mean(sample_pcc)) if len(sample_pcc) > 0 else 0.0,
        "sample_pcc": sample_pcc,
        "sample_mse": sample_mse,
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(pred_mean, dtype=np.float32),
        "y_logvar": (np.zeros_like(pred_mean, dtype=np.float32) if pred_logvar is None else np.asarray(pred_logvar, dtype=np.float32)),
    }


def _eval_deepcop_pack(model, data_pack, batch_size, max_eval, loss_mask):
    x_ctl, x_drugfeat, y_true = data_pack
    return eval_pcc_mse(
        model,
        x_ctl,
        x_drugfeat,
        y_true,
        loss_mask,
        batch_size=batch_size,
        max_eval=max_eval,
    )


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drug_feature", choices=["target", "fingerprint", "target+fingerprint"], default="target")
    p.add_argument("--include_cell_onehot", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--drop_fp_has_target", action="store_true", default=False)
    p.add_argument("--eval_drug_zero", action="store_true", default=False)
    p.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    p.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    p.add_argument("--eval_sanity_seed", type=int, default=0)
    p.add_argument("--save_json", default="")
    p.add_argument("--save_pred_prefix", default="")
    p.add_argument("--original_architecture", action="store_true", default=False)
    p.add_argument("--predict_uncertainty", action="store_true", default=False)
    p.add_argument("--logvar_clip_min", type=float, default=-6.0)
    p.add_argument("--logvar_clip_max", type=float, default=2.0)
    p.add_argument("--pcc_lambda", type=float, default=5.0)
    p.add_argument("--split_modes", default="cold_target_pattern")
    args = p.parse_args()

    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    root = str(args.root)
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    if os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))



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
        pairing_mode=str(args.pairing_mode),
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
    anchor_cell_names = np.asarray(data["anchor_cell_names"], dtype=str)
    anchor_trt_distil_ids = np.asarray(data.get("anchor_trt_distil_ids", [""] * len(anchor_drug_ids)), dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    def pcc_loss_tf(y_true, y_pred):
        mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
        my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        xm = y_true - mx
        ym = y_pred - my
        r_num = tf.reduce_sum(xm * ym, axis=1)
        r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8)
        r = r_num / r_den
        return 1.0 - tf.reduce_mean(r)

    def combined_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        mask = tf.cast(loss_mask, tf.float32)
        valid_count = tf.reduce_sum(mask)
        batch_n = tf.cast(tf.shape(y_true)[0], tf.float32)
        if bool(args.predict_uncertainty):
            mean_pred = y_pred[..., 0]
            logvar_pred = y_pred[..., 1]
            logvar_pred = tf.clip_by_value(logvar_pred, float(args.logvar_clip_min), float(args.logvar_clip_max))
            inv_var = tf.exp(-logvar_pred)
            nll = 0.5 * (logvar_pred + tf.square(y_true - mean_pred) * inv_var)
            base_loss = tf.reduce_sum(nll * mask) / tf.maximum(valid_count * batch_n, 1.0)
        else:
            mean_pred = y_pred
            mse = tf.reduce_sum(tf.square(y_true - mean_pred) * mask)
            base_loss = mse / tf.maximum(valid_count * batch_n, 1.0)
        valid_indices = tf.where(loss_mask[0] > 0)[:, 0]
        yt = tf.gather(y_true, valid_indices, axis=1)
        yp = tf.gather(mean_pred, valid_indices, axis=1)
        pcc = pcc_loss_tf(yt, yp)
        return base_loss + tf.cast(float(args.pcc_lambda), tf.float32) * pcc

    if int(args.max_samples) > 0 and len(anchor_drug_ids) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(anchor_drug_ids), size=int(args.max_samples), replace=False)
        data = subset_anchor_data(data, idx)
        anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
        anchor_cell_names = np.asarray(data["anchor_cell_names"], dtype=str)
        anchor_trt_distil_ids = np.asarray(data.get("anchor_trt_distil_ids", [""] * len(anchor_drug_ids)), dtype=str)

    results = []
    split_modes = parse_split_modes(args.split_modes, "cold_target_pattern")
    for split_mode in split_modes:
        tf.keras.backend.clear_session()
        tf.random.set_seed(int(args.seed))
        print(f"\n===== Running split: {split_mode} =====")
        train_data, test_data, train_mask, test_mask = prepare_split_data(
            data=data,
            split_mode=split_mode,
            test_frac=0.2,
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

        train_y = train_y_full
        test_y = test_y_full

        model = DeepCOP(
            num_genes=int(train_y.shape[1]),
            drug_dim=int(train_drug.shape[1]),
            dropout=float(args.dropout),
            use_residual=False,
            original_architecture=bool(args.original_architecture),
            predict_uncertainty=bool(args.predict_uncertainty),
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(args.lr)), loss=combined_loss, run_eagerly=False)

        cb = GenericPCCCallback(
            train_data=(train_ctl, train_drug, train_y),
            val_data=(test_ctl, test_drug, test_y),
            evaluate_pack_fn=lambda model_, pack, batch_size, max_eval: _eval_deepcop_pack(
                model_,
                pack,
                batch_size=batch_size,
                max_eval=max_eval,
                loss_mask=loss_mask,
            ),
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
                y_logvar=test_full["y_logvar"],
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
                "original_architecture": bool(args.original_architecture),
                "predict_uncertainty": bool(args.predict_uncertainty),
                "pcc_lambda": float(args.pcc_lambda),
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
