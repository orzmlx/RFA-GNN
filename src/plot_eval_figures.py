import argparse
import json
import os
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

import numpy as np
import tensorflow as tf
from tensorflow import keras


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _valid_gene_indices(loss_mask):
    m = np.asarray(loss_mask)
    if m.ndim == 2:
        m = m[0]
    return np.where(m > 0)[0]


def _samplewise_pcc(y_true, y_pred, valid_idx):
    from scipy.stats import pearsonr

    yt = y_true[:, valid_idx]
    yp = y_pred[:, valid_idx]
    out = np.zeros((len(yt),), dtype=np.float32)
    for i in range(len(yt)):
        a = yt[i]
        b = yp[i]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            out[i] = float(pearsonr(a, b)[0])
        else:
            out[i] = 0.0
    return out


def _samplewise_mse(y_true, y_pred, valid_idx):
    diff = y_true[:, valid_idx] - y_pred[:, valid_idx]
    return np.mean(diff * diff, axis=1).astype(np.float32)


def _agg_metrics(sample_pcc, sample_mse):
    return {
        "pcc": float(np.mean(sample_pcc)) if len(sample_pcc) else 0.0,
        "mse": float(np.mean(sample_mse)) if len(sample_mse) else 0.0,
    }


def _parse_list_arg(x: str):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    return [t.strip() for t in s.split(",") if t.strip() != ""]


def _downsample_pairs(x, y, max_points, seed=0):
    n = len(x)
    if n <= int(max_points):
        return x, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    return x[idx], y[idx]


def _save_npz(path, **kwargs):
    payload = {}
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            try:
                payload[k] = np.asarray([json.dumps(v)], dtype=object)
            except TypeError:
                payload[k] = np.asarray([v], dtype=object)
        else:
            payload[k] = v
    np.savez_compressed(path, **payload)


def _load_npz(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        v = z[k]
        if v.dtype == object and v.shape == (1,) and isinstance(v[0], str):
            try:
                out[k] = json.loads(v[0])
                continue
            except Exception:
                pass
        if v.dtype == object and v.shape == (1,) and isinstance(v[0], (dict, list)):
            out[k] = v[0]
            continue
        out[k] = v
    out["_path"] = path
    return out


def _plot_scatter_true_pred(y_true, y_pred, out_path, title, max_points=200000, seed=0):
    import matplotlib.pyplot as plt

    x = y_true.reshape(-1)
    y = y_pred.reshape(-1)
    x, y = _downsample_pairs(x, y, max_points=max_points, seed=seed)

    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(x, y, s=2, alpha=0.2, linewidths=0)
    lim = np.nanmax(np.abs(np.concatenate([x, y], axis=0)))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    plt.plot([-lim, lim], [-lim, lim], color="black", linewidth=1, alpha=0.6)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_bar(metrics, labels, out_path, title, ylabel):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(6, 0.8 * len(labels)), 3.8), dpi=150)
    xs = np.arange(len(labels))
    plt.bar(xs, metrics, color="#4C78A8")
    plt.xticks(xs, labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_group_box(values, groups, out_path, title, ylabel, top_k=12):
    import matplotlib.pyplot as plt

    groups = np.asarray(groups, dtype=str)
    values = np.asarray(values, dtype=np.float32)
    uniq, counts = np.unique(groups, return_counts=True)
    order = uniq[np.argsort(-counts)]
    order = order[: int(top_k)]
    data = []
    labels = []
    for g in order:
        mask = groups == g
        v = values[mask]
        if len(v) == 0:
            continue
        data.append(v)
        labels.append(f"{g} (n={len(v)})")

    plt.figure(figsize=(max(7, 0.55 * len(labels)), 4.0), dpi=150)
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@dataclass
class ExportArgs:
    root: str
    weights: str
    out: str
    cell_line: str
    use_landmark_genes: bool
    max_samples: int
    split_mode: str
    test_frac: float
    sparse_gat: bool
    use_drug_fp_embedding: bool
    hidden_dim: int
    num_heads: int
    dropout: float
    attention_layers: int
    per_node_head: bool
    no_cell_embedding: bool
    no_residualize_target_by_cell: bool
    eval_drug_zero: bool
    eval_drug_shuffle: bool
    eval_sanity_seed: int
    eval_sanity_max_eval: int
    test_ids_npy: str
    export_attention: bool = False
    attention_max_samples: int = 2000
    attention_batch_size: int = 64
    attention_group_by: str = ""
    attention_groups: str = ""
    attention_top_k_groups: int = 10


def export_predictions(args: ExportArgs):
    from data_loader import load_rfa_data, build_combined_gnn
    from base_gnn import BaseLineGAT

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"

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
        ctl_residual_pool_size=3,
    )

    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        #string_path=None,
        target_genes=data["target_genes"],
        confid_threshold=0.9,
        directed=True,
        omnipath_consensus_only=False,
        omnipath_is_directed_only=False,
        symbol_to_entrez=data.get("symbol_to_entrez"),
    )
    if len(node_list) != len(data["target_genes"]) or node_list[:50] != data["target_genes"][:50]:
        raise ValueError("Graph node_list 与表达 target_genes 顺序/长度不一致")

    X_ctl = np.asarray(data["X_ctl"], dtype=np.float32)
    y_delta = np.asarray(data["y_delta"], dtype=np.float32)
    X_drug = np.asarray(data["X_drug"], dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    batch_ids = np.asarray(data.get("batch_ids", ["Unknown"] * len(drug_ids)), dtype=str)
    det_plate_ids = np.asarray(data.get("det_plate_ids", ["Unknown"] * len(drug_ids)), dtype=str)
    trt_distil_ids = np.asarray(data.get("trt_distil_ids", [""] * len(drug_ids)), dtype=str)

    X_fp = data.get("X_fingerprint")
    X_fp = None if X_fp is None else np.asarray(X_fp, dtype=np.float32)
    fp_table = data.get("drug_fp_table")
    fp_idx = data.get("drug_fp_idx")
    fp_table = None if fp_table is None else np.asarray(fp_table, dtype=np.float32)
    fp_idx = None if fp_idx is None else np.asarray(fp_idx, dtype=np.int32)

    if str(args.test_ids_npy).strip() != "" and int(args.max_samples) > 0:
        raise ValueError("使用 --test_ids_npy 时不允许设置 --max_samples（会改变样本索引）")

    if int(args.max_samples) > 0 and len(X_ctl) > int(args.max_samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_ctl), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y_delta = y_delta[idx]
        X_drug = X_drug[idx]
        if X_fp is not None:
            X_fp = X_fp[idx]
        if X_fp is None and fp_idx is not None:
            fp_idx = fp_idx[idx]
        drug_ids = drug_ids[idx]
        cell_names = cell_names[idx]
        batch_ids = batch_ids[idx]
        det_plate_ids = det_plate_ids[idx]
        trt_distil_ids = trt_distil_ids[idx]


    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names)
    num_cells = int(len(le.classes_))

    split_mode = str(args.split_mode)
    test_frac = float(args.test_frac)
    np.random.seed(42)
    if str(args.test_ids_npy).strip() != "":
        ids = np.load(str(args.test_ids_npy).strip(), allow_pickle=True).astype(str)
        test_mask = np.isin(trt_distil_ids.astype(str), ids)
        if int(np.sum(test_mask)) == 0:
            raise ValueError("test_ids_npy 与当前数据集无交集；请确认数据版本与过滤设置一致")
    else:
        raise Exception("请指定 --test_ids_npy 或 --split_mode")
        # if split_mode == "cold_cell":
        #     unique_cells = np.unique(cell_idx)
        #     n_test = max(1, int(len(unique_cells) * test_frac))
        #     n_test = min(n_test, len(unique_cells) - 1)
        #     test_cells = np.random.choice(unique_cells, n_test, replace=False)
        #     test_mask = np.isin(cell_idx, test_cells)
        # else:
        #     unique_drugs = np.unique(drug_ids)
        #     n_test = max(1, int(len(unique_drugs) * test_frac))
        #     n_test = min(n_test, len(unique_drugs) - 1)
        #     test_drugs = np.random.choice(unique_drugs, n_test, replace=False)
        #     test_mask = np.isin(drug_ids, test_drugs)
    train_mask = ~test_mask

    residualize_target = not bool(args.no_residualize_target_by_cell)
    cell_delta_mean = None
    train_trt_full = y_delta[train_mask]
    test_trt_full = y_delta[test_mask]
    train_cells = cell_idx[train_mask]
    test_cells = cell_idx[test_mask]
    if residualize_target:
        num_genes = int(train_trt_full.shape[1])
        sums = np.zeros((num_cells, num_genes), dtype=np.float32)
        counts = np.zeros((num_cells,), dtype=np.int64)
        np.add.at(sums, train_cells, train_trt_full)
        np.add.at(counts, train_cells, 1)
        cell_delta_mean = sums / np.maximum(counts[:, None], 1)
        #train_trt = train_trt_full - cell_delta_mean[train_cells]
        test_trt = test_trt_full - cell_delta_mean[test_cells]
    else:
        #train_trt = train_trt_full
        test_trt = test_trt_full

    if X_fp is not None:
        fp_dim = int(X_fp.shape[1])
    elif fp_table is not None:
        fp_dim = int(fp_table.shape[1])
    else:
        fp_dim = 0

    model = BaseLineGAT(
        num_genes=int(adj_matrix.shape[0]),
        num_cells=num_cells,
        fingerprint_dim=int(fp_dim),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        use_residual=False,
        use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
        attention_layer_number=int(args.attention_layers),
        per_node_embedding=bool(args.per_node_head),
        use_sparse_adj=bool(args.sparse_gat),
        use_cell_embedding=not bool(args.no_cell_embedding),
    )

    if bool(args.sparse_gat):
        edge_index_np = edge_index.astype(np.int64)
        src = edge_index_np[0]
        dst = edge_index_np[1]
        edge_weight = adj_matrix[dst, src].astype(np.float32)
        n = int(adj_matrix.shape[0])
        self_idx = np.arange(n, dtype=np.int64)
        edge_index_full = np.concatenate([edge_index_np, np.stack([self_idx, self_idx], axis=0)], axis=1)
        edge_weight_full = np.concatenate([edge_weight, np.ones((n,), dtype=np.float32)], axis=0)
        edge_index_tf = tf.constant(edge_index_full, dtype=tf.int32)
        edge_weight_tf = tf.constant(edge_weight_full, dtype=tf.float32)
        fp_table_tf = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

        class Wrapper(keras.Model):
            def __init__(self):
                super().__init__()
                self.edge_index = edge_index_tf
                self.edge_weight = edge_weight_tf
                self.fp_table = fp_table_tf

            def call(self, inputs, training=False):
                if bool(args.use_drug_fp_embedding):
                    ctl, drug_targets, cidx, drug_fp = inputs
                    cidx = tf.cast(cidx, tf.int32)
                    if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                        drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
                    return model([self.edge_index, self.edge_weight, ctl, drug_targets, cidx, drug_fp], training=training)
                ctl, drug_targets, cidx = inputs
                cidx = tf.cast(cidx, tf.int32)
                return model([self.edge_index, self.edge_weight, ctl, drug_targets, cidx], training=training)

        wrapped = Wrapper()
    else:
        adj_tf = tf.constant(adj_matrix, dtype=tf.float32)
        fp_table_tf = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

        class Wrapper(keras.Model):
            def __init__(self):
                super().__init__()
                self.adj = adj_tf
                self.fp_table = fp_table_tf

            def call(self, inputs, training=False):
                if bool(args.use_drug_fp_embedding):
                    ctl, drug_targets, cidx, drug_fp = inputs
                    cidx = tf.cast(cidx, tf.int32)
                    if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                        drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
                    return model([self.adj, ctl, drug_targets, cidx, drug_fp], training=training)
                ctl, drug_targets, cidx = inputs
                cidx = tf.cast(cidx, tf.int32)
                return model([self.adj, ctl, drug_targets, cidx], training=training)

        wrapped = Wrapper()

    wrapped.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    dummy_bs = 2
    if bool(args.use_drug_fp_embedding):
        dummy_fp = np.zeros((dummy_bs, max(fp_dim, 1)), dtype=np.float32) if X_fp is None else np.zeros((dummy_bs, fp_dim), dtype=np.float32)
        dummy_x = [np.zeros((dummy_bs, X_ctl.shape[1]), dtype=np.float32), np.zeros((dummy_bs, X_drug.shape[1]), dtype=np.float32), np.zeros((dummy_bs,), dtype=np.int32), dummy_fp]
    else:
        dummy_x = [np.zeros((dummy_bs, X_ctl.shape[1]), dtype=np.float32), np.zeros((dummy_bs, X_drug.shape[1]), dtype=np.float32), np.zeros((dummy_bs,), dtype=np.int32)]
    _ = wrapped.predict(dummy_x, verbose=0)
    model.load_weights(args.weights)

    # train_ctl = X_ctl[train_mask]
    # train_drug = X_drug[train_mask]
    test_ctl = X_ctl[test_mask]
    test_drug = X_drug[test_mask]
    #train_cell_idx = cell_idx[train_mask].astype(np.int32)
    test_cell_idx = cell_idx[test_mask].astype(np.int32)
   #train_drug_ids = drug_ids[train_mask]
    test_drug_ids = drug_ids[test_mask]
    train_cell_names = cell_names[train_mask]
    test_cell_names = cell_names[test_mask]
    #train_batch_ids = batch_ids[train_mask]
    test_batch_ids = batch_ids[test_mask]
    #train_plate_ids = det_plate_ids[train_mask]
    test_plate_ids = det_plate_ids[test_mask]

    if X_fp is not None:
        #train_fp = X_fp[train_mask]
        test_fp = X_fp[test_mask]
    else:
        #train_fp = None if fp_idx is None else fp_idx[train_mask]
        test_fp = None if fp_idx is None else fp_idx[test_mask]

    if bool(args.use_drug_fp_embedding):
        y_pred_test = wrapped.predict([test_ctl, test_drug, test_cell_idx, test_fp], batch_size=256, verbose=0)
    else:
        y_pred_test = wrapped.predict([test_ctl, test_drug, test_cell_idx], batch_size=256, verbose=0)

    y_true_test = test_trt

    if residualize_target and cell_delta_mean is not None:
        y_true_test_full = test_trt_full
        y_pred_test_full = y_pred_test + cell_delta_mean[test_cells]
    else:
        y_true_test_full = y_true_test
        y_pred_test_full = y_pred_test

    valid_idx = _valid_gene_indices(data["loss_mask"])
    sample_pcc = _samplewise_pcc(y_true_test_full, y_pred_test_full, valid_idx)
    sample_mse = _samplewise_mse(y_true_test_full, y_pred_test_full, valid_idx)
    metrics = _agg_metrics(sample_pcc, sample_mse)

    sanity = {}
    if bool(args.eval_drug_zero):
        zero_drug = np.zeros_like(test_drug, dtype=np.float32)
        if bool(args.use_drug_fp_embedding):
            if isinstance(test_fp, np.ndarray) and test_fp.ndim == 2:
                zero_fp = np.zeros_like(test_fp, dtype=np.float32)
            else:
                zero_fp = np.zeros((len(test_ctl), fp_dim), dtype=np.float32)
            pred_zero = wrapped.predict([test_ctl, zero_drug, test_cell_idx, zero_fp], batch_size=256, verbose=0)
        else:
            pred_zero = wrapped.predict([test_ctl, zero_drug, test_cell_idx], batch_size=256, verbose=0)
        if residualize_target and cell_delta_mean is not None:
            pred_zero = pred_zero + cell_delta_mean[test_cells]
        pcc_zero = _samplewise_pcc(y_true_test_full, pred_zero, valid_idx)
        mse_zero = _samplewise_mse(y_true_test_full, pred_zero, valid_idx)
        sanity["drug_zero"] = _agg_metrics(pcc_zero, mse_zero)

    if bool(args.eval_drug_shuffle):
        rng = np.random.default_rng(int(args.eval_sanity_seed))
        perm = rng.permutation(len(test_ctl))
        shuf_drug = test_drug[perm]
        if bool(args.use_drug_fp_embedding):
            shuf_fp = None if test_fp is None else test_fp[perm]
            pred_shuf = wrapped.predict([test_ctl, shuf_drug, test_cell_idx, shuf_fp], batch_size=256, verbose=0)
        else:
            pred_shuf = wrapped.predict([test_ctl, shuf_drug, test_cell_idx], batch_size=256, verbose=0)
        if residualize_target and cell_delta_mean is not None:
            pred_shuf = pred_shuf + cell_delta_mean[test_cells]
        pcc_shuf = _samplewise_pcc(y_true_test_full, pred_shuf, valid_idx)
        mse_shuf = _samplewise_mse(y_true_test_full, pred_shuf, valid_idx)
        sanity["drug_shuffle"] = _agg_metrics(pcc_shuf, mse_shuf)

    meta = {
        "split_mode": split_mode,
        "test_frac": float(test_frac),
        "use_landmark_genes": bool(args.use_landmark_genes),
        "use_drug_fp_embedding": bool(args.use_drug_fp_embedding),
        "sparse_gat": bool(args.sparse_gat),
        "weights": str(args.weights),
        "num_test": int(len(test_ctl)),
        "test_ids_npy": str(args.test_ids_npy).strip(),
    }

    attention = {}
    if bool(args.export_attention):
        if not bool(args.sparse_gat):
            raise ValueError("export_attention 目前只支持 sparse_gat=True")

        rng = np.random.default_rng(0)
        n_all = int(len(test_ctl))
        n_attn = min(int(args.attention_max_samples), n_all) if int(args.attention_max_samples) > 0 else n_all
        if n_attn <= 0:
            raise ValueError("attention_max_samples must be > 0 or 0 for full")
        sel = rng.choice(n_all, size=n_attn, replace=False) if n_attn < n_all else np.arange(n_all)

        att_ctl = test_ctl[sel]
        att_drug = test_drug[sel]
        att_cell = test_cell_idx[sel]
        group_by = str(getattr(args, "attention_group_by", "")).strip().lower()
        if group_by not in {"", "none", "drug", "cell"}:
            raise ValueError("attention_group_by must be one of: '', 'drug', 'cell'")
        group_labels = None
        if group_by == "drug":
            group_labels = np.asarray(test_drug_ids[sel], dtype=str)
        elif group_by == "cell":
            group_labels = np.asarray(test_cell_names[sel], dtype=str)

        keep_groups = None
        if group_labels is not None:
            specified = _parse_list_arg(getattr(args, "attention_groups", "")) or []
            if len(specified) > 0:
                keep_groups = set([str(x) for x in specified])
            else:
                uniq, counts = np.unique(group_labels, return_counts=True)
                order = uniq[np.argsort(-counts)]
                top_k = int(getattr(args, "attention_top_k_groups", 10) or 10)
                keep_groups = set(order[:top_k].tolist())
        if bool(args.use_drug_fp_embedding):
            att_fp = test_fp[sel]
        else:
            att_fp = None
        att_cells_for_mean = test_cells[sel]

        edge_index_np = edge_index.astype(np.int64)
        src = edge_index_np[0]
        dst = edge_index_np[1]
        edge_weight = adj_matrix[dst, src].astype(np.float32)
        n = int(adj_matrix.shape[0])
        self_idx = np.arange(n, dtype=np.int64)
        edge_index_full = np.concatenate([edge_index_np, np.stack([self_idx, self_idx], axis=0)], axis=1)
        edge_weight_full = np.concatenate([edge_weight, np.ones((n,), dtype=np.float32)], axis=0)
        edge_index_tf = tf.constant(edge_index_full, dtype=tf.int32)
        edge_weight_tf = tf.constant(edge_weight_full, dtype=tf.float32)

        layer_sums = None
        group_sums = None
        group_counts = None
        bs = int(args.attention_batch_size) if int(args.attention_batch_size) > 0 else 64
        for start in range(0, n_attn, bs):
            end = min(start + bs, n_attn)
            b_ctl = tf.constant(att_ctl[start:end], dtype=tf.float32)
            b_drug = tf.constant(att_drug[start:end], dtype=tf.float32)
            b_cell = tf.constant(att_cell[start:end], dtype=tf.int32)
            if bool(args.use_drug_fp_embedding):
                if att_fp is None:
                    raise RuntimeError("use_drug_fp_embedding=True but fp is None")
                if isinstance(att_fp, np.ndarray) and att_fp.ndim == 2:
                    b_fp = tf.constant(att_fp[start:end], dtype=tf.float32)
                else:
                    b_fp = tf.constant(att_fp[start:end], dtype=tf.int32)
                out, attns = model([edge_index_tf, edge_weight_tf, b_ctl, b_drug, b_cell, b_fp], training=False, output_attention=True)
            else:
                out, attns = model([edge_index_tf, edge_weight_tf, b_ctl, b_drug, b_cell], training=False, output_attention=True)

            if residualize_target and cell_delta_mean is not None:
                _ = out + tf.constant(cell_delta_mean[att_cells_for_mean[start:end]], dtype=tf.float32)

            att_np = []
            for li, a in enumerate(attns):
                a = tf.reduce_mean(a, axis=1)
                a_np = a.numpy().astype(np.float64)
                att_np.append(a_np)
                s = np.sum(a_np, axis=0)
                if layer_sums is None:
                    layer_sums = [np.zeros_like(s, dtype=np.float64) for _ in range(len(attns))]
                layer_sums[li] += s

            if group_labels is not None and keep_groups is not None:
                if group_sums is None:
                    group_sums = {g: [np.zeros((att_np[0].shape[1],), dtype=np.float64) for _ in range(len(attns))] for g in keep_groups}
                    group_counts = {g: 0 for g in keep_groups}
                b_labels = group_labels[start:end]
                uniq_b = np.unique(b_labels)
                for g in uniq_b:
                    if g not in keep_groups:
                        continue
                    idx = np.where(b_labels == g)[0]
                    if len(idx) == 0:
                        continue
                    group_counts[g] += int(len(idx))
                    for li in range(len(attns)):
                        group_sums[g][li] += np.sum(att_np[li][idx], axis=0)

        layer_means = [x / float(n_attn) for x in layer_sums]
        attention = {
            "attention_edge_mean": np.stack(layer_means, axis=0).astype(np.float32),
            "edge_index": edge_index_full.astype(np.int32),
            "edge_weight": edge_weight_full.astype(np.float32),
            "attention_num_samples": int(n_attn),
        }
        if group_sums is not None and group_counts is not None:
            group_means = {}
            for g, c in group_counts.items():
                if int(c) <= 0:
                    continue
                group_means[str(g)] = np.stack([group_sums[g][li] / float(c) for li in range(len(group_sums[g]))], axis=0).astype(np.float32)
            attention["group_by"] = group_by
            attention["group_counts"] = {str(k): int(v) for k, v in group_counts.items()}
            attention["group_attention_edge_mean"] = group_means
            meta["attention_group_by"] = group_by
            meta["attention_groups"] = sorted(list(group_means.keys()))
        meta["export_attention"] = True
        meta["attention_num_samples"] = int(n_attn)

    _ensure_dir(os.path.dirname(args.out) or ".")
    _save_npz(
        args.out,
        y_true=y_true_test_full.astype(np.float32),
        y_pred=y_pred_test_full.astype(np.float32),
        sample_pcc=sample_pcc,
        sample_mse=sample_mse,
        cell_names=test_cell_names.astype(object),
        drug_ids=test_drug_ids.astype(object),
        batch_ids=test_batch_ids.astype(object),
        det_plate_ids=test_plate_ids.astype(object),
        trt_distil_ids=np.asarray(trt_distil_ids[test_mask], dtype=object),
        target_genes=np.asarray(data["target_genes"], dtype=object),
        metrics=metrics,
        sanity=sanity,
        attention=attention,
        meta=meta,
    )


def plot_figures(inputs, outdir, cells=None, top_k_cells=6, top_k_drugs=12):
    import matplotlib.pyplot as plt

    _ensure_dir(outdir)
    runs = [_load_npz(p) for p in inputs]

    for r in runs:
        tag = os.path.splitext(os.path.basename(r["_path"]))[0]
        _plot_scatter_true_pred(
            r["y_true"],
            r["y_pred"],
            os.path.join(outdir, f"{tag}_scatter_all.png"),
            title=f"{tag}: Pred vs True (sampled)",
        )

        sample_pcc = r.get("sample_pcc")
        sample_mse = r.get("sample_mse")
        if sample_pcc is not None:
            _plot_group_box(
                sample_pcc,
                r.get("cell_names"),
                os.path.join(outdir, f"{tag}_pcc_by_cell.png"),
                title=f"{tag}: Sample-wise PCC by cell (top)",
                ylabel="PCC",
                top_k=int(top_k_cells),
            )
            _plot_group_box(
                sample_pcc,
                r.get("drug_ids"),
                os.path.join(outdir, f"{tag}_pcc_by_drug.png"),
                title=f"{tag}: Sample-wise PCC by drug (top)",
                ylabel="PCC",
                top_k=int(top_k_drugs),
            )
        if sample_mse is not None:
            _plot_group_box(
                sample_mse,
                r.get("cell_names"),
                os.path.join(outdir, f"{tag}_mse_by_cell.png"),
                title=f"{tag}: Sample-wise MSE by cell (top)",
                ylabel="MSE",
                top_k=int(top_k_cells),
            )

        if cells is not None:
            chosen = set(cells)
        else:
            names = np.asarray(r.get("cell_names"), dtype=str)
            uniq, counts = np.unique(names, return_counts=True)
            order = uniq[np.argsort(-counts)]
            chosen = set(order[: int(top_k_cells)])

        for c in chosen:
            mask = np.asarray(r.get("cell_names"), dtype=str) == str(c)
            if int(np.sum(mask)) == 0:
                continue
            _plot_scatter_true_pred(
                r["y_true"][mask],
                r["y_pred"][mask],
                os.path.join(outdir, f"{tag}_scatter_cell_{str(c)}.png"),
                title=f"{tag}: Pred vs True ({str(c)})",
                max_points=120000,
                seed=1,
            )

        sanity = r.get("sanity", {})
        if isinstance(sanity, dict) and len(sanity) > 0:
            base_pcc = float(r.get("metrics", {}).get("pcc", 0.0))
            labels = ["test"]
            vals = [base_pcc]
            if "drug_zero" in sanity:
                labels.append("drug_zero")
                vals.append(float(sanity["drug_zero"]["pcc"]))
            if "drug_shuffle" in sanity:
                labels.append("drug_shuffle")
                vals.append(float(sanity["drug_shuffle"]["pcc"]))
            _plot_bar(vals, labels, os.path.join(outdir, f"{tag}_sanity_pcc.png"), title=f"{tag}: Sanity PCC", ylabel="PCC")

            deltas = []
            dlabels = []
            for k in ["drug_zero", "drug_shuffle"]:
                if k in sanity:
                    dlabels.append(f"ΔPCC vs {k}")
                    deltas.append(base_pcc - float(sanity[k]["pcc"]))
            if len(deltas) > 0:
                _plot_bar(deltas, dlabels, os.path.join(outdir, f"{tag}_delta_pcc.png"), title=f"{tag}: Drug contribution", ylabel="ΔPCC")

    if len(runs) >= 2:
        labels = []
        pccs = []
        mses = []
        dz = []
        ds = []
        for r in runs:
            tag = os.path.splitext(os.path.basename(r["_path"]))[0]
            labels.append(tag)
            pccs.append(float(r.get("metrics", {}).get("pcc", 0.0)))
            mses.append(float(r.get("metrics", {}).get("mse", 0.0)))
            sanity = r.get("sanity", {})
            if isinstance(sanity, dict) and "drug_zero" in sanity:
                dz.append(float(r.get("metrics", {}).get("pcc", 0.0)) - float(sanity["drug_zero"]["pcc"]))
            else:
                dz.append(0.0)
            if isinstance(sanity, dict) and "drug_shuffle" in sanity:
                ds.append(float(r.get("metrics", {}).get("pcc", 0.0)) - float(sanity["drug_shuffle"]["pcc"]))
            else:
                ds.append(0.0)

        _plot_bar(pccs, labels, os.path.join(outdir, "compare_pcc.png"), title="Compare PCC (test)", ylabel="PCC")
        _plot_bar(mses, labels, os.path.join(outdir, "compare_mse.png"), title="Compare MSE (test)", ylabel="MSE")
        _plot_bar(dz, labels, os.path.join(outdir, "compare_delta_pcc_zero.png"), title="Compare ΔPCC (drug_zero)", ylabel="ΔPCC")
        _plot_bar(ds, labels, os.path.join(outdir, "compare_delta_pcc_shuffle.png"), title="Compare ΔPCC (drug_shuffle)", ylabel="ΔPCC")


def load_eval_npz(path: str):
    return _load_npz(path)


def plot_true_pred_scatter(run, out_path, title=None, max_points=200000, seed=0):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    _plot_scatter_true_pred(
        run["y_true"],
        run["y_pred"],
        out_path,
        title=title or f"{tag}: Pred vs True (sampled)",
        max_points=max_points,
        seed=seed,
    )


def plot_true_pred_scatter_for_cell(run, cell_name, out_path, title=None, max_points=120000, seed=1):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    mask = np.asarray(run.get("cell_names"), dtype=str) == str(cell_name)
    if int(np.sum(mask)) == 0:
        raise ValueError(f"cell_name not found: {cell_name}")
    _plot_scatter_true_pred(
        run["y_true"][mask],
        run["y_pred"][mask],
        out_path,
        title=title or f"{tag}: Pred vs True ({str(cell_name)})",
        max_points=max_points,
        seed=seed,
    )


def plot_sample_pcc_by_cell(run, out_path, top_k=6, title=None):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    _plot_group_box(
        run["sample_pcc"],
        run.get("cell_names"),
        out_path,
        title=title or f"{tag}: Sample-wise PCC by cell (top)",
        ylabel="PCC",
        top_k=int(top_k),
    )


def plot_sample_pcc_by_drug(run, out_path, top_k=12, title=None):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    _plot_group_box(
        run["sample_pcc"],
        run.get("drug_ids"),
        out_path,
        title=title or f"{tag}: Sample-wise PCC by drug (top)",
        ylabel="PCC",
        top_k=int(top_k),
    )


def plot_sample_mse_by_cell(run, out_path, top_k=6, title=None):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    _plot_group_box(
        run["sample_mse"],
        run.get("cell_names"),
        out_path,
        title=title or f"{tag}: Sample-wise MSE by cell (top)",
        ylabel="MSE",
        top_k=int(top_k),
    )


def plot_sanity_pcc(run, out_path, title=None):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    sanity = run.get("sanity", {}) or {}
    base_pcc = float(run.get("metrics", {}).get("pcc", 0.0))
    labels = ["test"]
    vals = [base_pcc]
    if "drug_zero" in sanity:
        labels.append("drug_zero")
        vals.append(float(sanity["drug_zero"]["pcc"]))
    if "drug_shuffle" in sanity:
        labels.append("drug_shuffle")
        vals.append(float(sanity["drug_shuffle"]["pcc"]))
    _plot_bar(vals, labels, out_path, title=title or f"{tag}: Sanity PCC", ylabel="PCC")


def plot_delta_pcc(run, out_path, title=None):
    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    sanity = run.get("sanity", {}) or {}
    base_pcc = float(run.get("metrics", {}).get("pcc", 0.0))
    deltas = []
    labels = []
    for k in ["drug_zero", "drug_shuffle"]:
        if k in sanity:
            labels.append(f"ΔPCC vs {k}")
            deltas.append(base_pcc - float(sanity[k]["pcc"]))
    if len(deltas) == 0:
        raise ValueError("No sanity results in npz (drug_zero/drug_shuffle missing)")
    _plot_bar(deltas, labels, out_path, title=title or f"{tag}: Drug contribution", ylabel="ΔPCC")

def _get_attention_edge_mean(run, layer=-1, group=None):
    att = run.get("attention", {})
    if not isinstance(att, dict):
        raise ValueError("attention not found in npz; run export with --export_attention")
    if group is None:
        a = np.asarray(att.get("attention_edge_mean"))
    else:
        m = att.get("group_attention_edge_mean", {})
        if not isinstance(m, dict) or str(group) not in m:
            raise ValueError(f"attention group not found: {group}")
        a = np.asarray(m[str(group)])
    if a.ndim != 2:
        raise ValueError("attention_edge_mean must be (L, E)")
    return a[int(layer)]


def plot_attention_top_genes(run, out_path, top_k=20, layer=-1, title=None, group=None):
    import matplotlib.pyplot as plt

    att = run.get("attention", {})
    if not isinstance(att, dict) or "edge_index" not in att:
        raise ValueError("edge_index not found in attention")
    edge_index = np.asarray(att["edge_index"], dtype=np.int32)
    src = edge_index[0]
    dst = edge_index[1]
    alpha = _get_attention_edge_mean(run, layer=layer, group=group).astype(np.float64)
    n = int(np.max(edge_index)) + 1 if edge_index.size else 0
    scores = np.zeros((n,), dtype=np.float64)
    non_self = src != dst
    np.add.at(scores, dst[non_self], alpha[non_self])

    genes = np.asarray(run.get("target_genes"), dtype=str)
    if len(genes) != n:
        genes = np.asarray([str(i) for i in range(n)], dtype=str)

    k = min(int(top_k), n)
    idx = np.argsort(-scores)[:k]
    labels = genes[idx].tolist()
    vals = scores[idx].astype(np.float32)

    tag = os.path.splitext(os.path.basename(run.get("_path", "run")))[0]
    plt.figure(figsize=(max(6, 0.35 * k), 4.0), dpi=150)
    xs = np.arange(k)
    plt.bar(xs, vals, color="#F58518")
    plt.xticks(xs, labels, rotation=60, ha="right")
    plt.ylabel("Incoming non-self attention (sum)")
    if group is None:
        plt.title(title or f"{tag}: Top genes by non-self attention (layer {layer})")
    else:
        plt.title(title or f"{tag}: Top genes by non-self attention ({str(group)}, layer {layer})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def export_attention_top_edges_csv(run, out_path, top_k=50, layer=-1, include_self_loops=False, group=None):
    import pandas as pd

    att = run.get("attention", {})
    if not isinstance(att, dict) or "edge_index" not in att:
        raise ValueError("edge_index not found in attention")
    edge_index = np.asarray(att["edge_index"], dtype=np.int32)
    alpha = _get_attention_edge_mean(run, layer=layer, group=group).astype(np.float64)
    src = edge_index[0]
    dst = edge_index[1]
    if not bool(include_self_loops):
        m = src != dst
        src = src[m]
        dst = dst[m]
        alpha = alpha[m]

    genes = np.asarray(run.get("target_genes"), dtype=str)
    if len(genes) <= int(np.max(edge_index)):
        genes = np.asarray([str(i) for i in range(int(np.max(edge_index)) + 1)], dtype=str)

    k = min(int(top_k), len(alpha))
    idx = np.argsort(-alpha)[:k]
    df = pd.DataFrame(
        {
            "src_idx": src[idx].astype(int),
            "dst_idx": dst[idx].astype(int),
            "src_gene": genes[src[idx]].astype(str),
            "dst_gene": genes[dst[idx]].astype(str),
            "attention": alpha[idx].astype(np.float32),
        }
    )
    out_dir = os.path.dirname(str(out_path))
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def generate_test_ids_npy(meta_path: str, out_path: str, root: str = ""):
    import numpy as np
    import os
    import json
    from sklearn.preprocessing import LabelEncoder

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    root_path = str(root).strip()
    if root_path == "":
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(meta_path)))
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"root not found: {root_path}")

    from data_loader import load_rfa_data

    ctl_path = os.path.join(root_path, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root_path, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    siginfo_path = os.path.join(root_path, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root_path, "data/landmark_genes.json")
    full_gene_path = os.path.join(root_path, "data/GSE92742_Broad_LINCS_gene_info.txt")
    drug_target_path = os.path.join(root_path, "data/compound_targets.txt")
    fingerprint_path = os.path.join(root_path, "data/new_morgan_fingerprints.csv")

    cell_line = meta.get("cell_line", "ALL")
    if cell_line is not None:
        s = str(cell_line).strip()
        if s == "" or s.upper() in {"ALL", "NONE", "NULL"}:
            cell_line = None

    data = load_rfa_data(
        ctl_path,
        trt_path,
        drug_target_path=drug_target_path,
        landmark_path=landmark_path,
        siginfo_path=siginfo_path,
        fingerprint_path=fingerprint_path,
        use_landmark_genes=bool(meta.get("use_landmark_genes", True)),
        full_gene_path=full_gene_path,
        cell_lines=cell_line,
        ctl_residual_pool_size=int(meta.get("ctl_pair_k", 3)),
    )

    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    trt_distil_ids = np.asarray(data.get("trt_distil_ids", [""] * len(drug_ids)), dtype=str)
    if trt_distil_ids.size == 0 or len(trt_distil_ids) != len(drug_ids):
        raise RuntimeError("data_loader 未返回 trt_distil_ids，无法生成 test_ids.npy")

    split_mode = str(meta.get("split_mode", "cold_drug"))
    test_frac = float(meta.get("test_frac", 0.2))
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("test_frac must be in (0, 1)")

    np.random.seed(42)
    if split_mode == "cold_cell":
        le = LabelEncoder()
        cell_idx = le.fit_transform(cell_names)
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise ValueError("cold_cell requires at least 2 cells")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        test_cells = np.random.choice(unique_cells, n_test, replace=False)
        test_mask = np.isin(cell_idx, test_cells)
    else:
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise ValueError("cold_drug requires at least 2 drugs")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        test_drugs = np.random.choice(unique_drugs, n_test, replace=False)
        test_mask = np.isin(drug_ids, test_drugs)

    out_dir = os.path.dirname(str(out_path))
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    np.save(str(out_path), trt_distil_ids[test_mask].astype(str))
    return str(out_path)


def build_cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    e.add_argument("--weights", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--cell_line", default="ALL")
    e.add_argument("--use_landmark_genes", action="store_true", default=True)
    e.add_argument("--max_samples", type=int, default=0)
    e.add_argument("--split_mode", choices=["cold_drug", "cold_cell"], default="cold_drug")
    e.add_argument("--test_frac", type=float, default=0.2)
    e.add_argument("--sparse_gat", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--use_drug_fp_embedding", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--hidden_dim", type=int, default=64)
    e.add_argument("--num_heads", type=int, default=4)
    e.add_argument("--dropout", type=float, default=0.2)
    e.add_argument("--attention_layers", type=int, default=4)
    e.add_argument("--per_node_head", action="store_true", default=True)
    e.add_argument("--no_cell_embedding", action="store_true", default=False)
    e.add_argument("--no_residualize_target_by_cell", action="store_true", default=False)
    e.add_argument("--eval_drug_zero", action="store_true", default=False)
    e.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    e.add_argument("--eval_sanity_seed", type=int, default=0)
    e.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    e.add_argument("--test_ids_npy", default="")
    e.add_argument("--export_attention", action="store_true", default=False)
    e.add_argument("--attention_max_samples", type=int, default=2000)
    e.add_argument("--attention_batch_size", type=int, default=64)
    e.add_argument("--attention_group_by", choices=["", "drug", "cell"], default="")
    e.add_argument("--attention_groups", default="", help="Comma-separated group ids (drug ids or cell names)")
    e.add_argument("--attention_top_k_groups", type=int, default=10)

    pl = sub.add_parser("plot")
    pl.add_argument("--inputs", required=True, help="Comma-separated npz files")
    pl.add_argument("--outdir", required=True)
    pl.add_argument("--cells", default="")
    pl.add_argument("--top_k_cells", type=int, default=6)
    pl.add_argument("--top_k_drugs", type=int, default=12)
    return p


def main():
    cli = build_cli()
    args = cli.parse_args()

    if args.cmd == "export":
        export_predictions(
            ExportArgs(
                root=args.root,
                weights=args.weights,
                out=args.out,
                cell_line=args.cell_line,
                use_landmark_genes=bool(args.use_landmark_genes),
                max_samples=int(args.max_samples),
                split_mode=str(args.split_mode),
                test_frac=float(args.test_frac),
                sparse_gat=bool(args.sparse_gat),
                use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
                hidden_dim=int(args.hidden_dim),
                num_heads=int(args.num_heads),
                dropout=float(args.dropout),
                attention_layers=int(args.attention_layers),
                per_node_head=bool(args.per_node_head),
                no_cell_embedding=bool(args.no_cell_embedding),
                no_residualize_target_by_cell=bool(args.no_residualize_target_by_cell),
                eval_drug_zero=bool(args.eval_drug_zero),
                eval_drug_shuffle=bool(args.eval_drug_shuffle),
                eval_sanity_seed=int(args.eval_sanity_seed),
                eval_sanity_max_eval=int(args.eval_sanity_max_eval),
                test_ids_npy=str(args.test_ids_npy),
                export_attention=bool(args.export_attention),
                attention_max_samples=int(args.attention_max_samples),
                attention_batch_size=int(args.attention_batch_size),
                attention_group_by=str(args.attention_group_by),
                attention_groups=str(args.attention_groups),
                attention_top_k_groups=int(args.attention_top_k_groups),
            )
        )
        return

    if args.cmd == "plot":
        inps = _parse_list_arg(args.inputs) or []
        cells = _parse_list_arg(args.cells)
        plot_figures(inps, args.outdir, cells=cells, top_k_cells=int(args.top_k_cells), top_k_drugs=int(args.top_k_drugs))
        return


if __name__ == "__main__":
    main()
