import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def eval_pcc_mse(
    model,
    ctl,
    drug,
    cells,
    y_true,
    loss_mask,
    batch_size=32,
    max_eval=None,
    drug_fp=None,
    cell_mean=None,
    y_is_residual=False,
):
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

    if drug_fp is None:
        pred = model.predict([ctl, drug, cells], batch_size=batch_size, verbose=0)
    else:
        pred = model.predict([ctl, drug, cells, drug_fp], batch_size=batch_size, verbose=0)

    if bool(y_is_residual) and cell_mean is not None:
        cm = np.asarray(cell_mean, dtype=np.float32)[cells]
        y_true = y_true + cm
        pred = pred + cm
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
    def __init__(self, loss_mask, train_data, val_data, batch_size=32, max_eval=2048, cell_mean=None, y_is_residual=False):
        super().__init__()
        self.loss_mask = loss_mask
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.max_eval = int(max_eval) if max_eval is not None else None
        self.cell_mean = cell_mean
        self.y_is_residual = bool(y_is_residual)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        target_scale = None
        try:
            inner = getattr(self.model, "gat", None)
            if inner is not None and hasattr(inner, "target_scale_logit"):
                target_scale = float(tf.nn.softplus(inner.target_scale_logit).numpy())
        except Exception:
            target_scale = None
        if len(self.train_data) == 4:
            ctl_tr, drug_tr, cell_tr, y_tr = self.train_data
            ctl_va, drug_va, cell_va, y_va = self.val_data
            tr = eval_pcc_mse(
                self.model,
                ctl_tr,
                drug_tr,
                cell_tr,
                y_tr,
                self.loss_mask,
                batch_size=self.batch_size,
                max_eval=self.max_eval,
                cell_mean=self.cell_mean,
                y_is_residual=self.y_is_residual,
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
                cell_mean=self.cell_mean,
                y_is_residual=self.y_is_residual,
            )
        else:
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
                cell_mean=self.cell_mean,
                y_is_residual=self.y_is_residual,
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
                cell_mean=self.cell_mean,
                y_is_residual=self.y_is_residual,
            )
        logs["pcc"] = tr["pcc"]
        logs["val_pcc"] = va["pcc"]
        if target_scale is None:
            print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f}")
        else:
            print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f} target_scale={target_scale:.4f}")


class GATWrapper(keras.Model):
    def __init__(self, gat_model, adj_matrix, use_drug_fp_embedding=False, fp_table=None):
        super().__init__()
        self.gat = gat_model
        self.adj = tf.constant(adj_matrix, dtype=tf.float32)
        self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
        self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

    def call(self, inputs):
        if self.use_drug_fp_embedding:
            ctl, drug_targets, cell_idx, drug_fp = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
            return self.gat([self.adj, ctl, drug_targets, cell_idx, drug_fp])
        ctl, drug_targets, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        return self.gat([self.adj, ctl, drug_targets, cell_idx])


class GATWrapperSparse(keras.Model):
    def __init__(self, gat_model, edge_index, edge_weight, use_drug_fp_embedding=False, fp_table=None):
        super().__init__()
        self.gat = gat_model
        self.edge_index = tf.constant(edge_index, dtype=tf.int32)
        self.edge_weight = tf.constant(edge_weight, dtype=tf.float32)
        self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
        self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

    def call(self, inputs):
        if self.use_drug_fp_embedding:
            ctl, drug_targets, cell_idx, drug_fp = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
            return self.gat([self.edge_index, self.edge_weight, ctl, drug_targets, cell_idx, drug_fp])
        ctl, drug_targets, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        return self.gat([self.edge_index, self.edge_weight, ctl, drug_targets, cell_idx])

class CtlResidualAugmentDataset(keras.utils.PyDataset):
    def __init__(
        self,
        ctl_mean,
        drug_targets,
        cell_idx,
        drug_fp,
        y_true,
        cell_names,
        batch_ids,
        ctl_residual_pool,
        batch_size=32,
        alpha=1.0,
        prob=1.0,
        seed=42,
    ):
        super().__init__()
        self.ctl_mean = np.asarray(ctl_mean, dtype=np.float32)
        self.drug_targets = np.asarray(drug_targets, dtype=np.float32)
        self.cell_idx = np.asarray(cell_idx)
        if drug_fp is None:
            self.drug_fp = None
        else:
            arr = np.asarray(drug_fp)
            if arr.dtype.kind in {"i", "u"} and arr.ndim == 1:
                self.drug_fp = arr.astype(np.int32)
            else:
                self.drug_fp = arr.astype(np.float32)
        self.y_true = np.asarray(y_true, dtype=np.float32)
        self.cell_names = np.asarray(cell_names, dtype=str)
        self.batch_ids = np.asarray(batch_ids, dtype=str)
        self.pool = ctl_residual_pool or {}
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.prob = float(prob)
        self.rng = np.random.default_rng(int(seed))
        self.indices = np.arange(len(self.ctl_mean))

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.indices))
        bidx = self.indices[start:end]

        ctl = self.ctl_mean[bidx].copy()
        if self.prob > 0.0 and len(self.pool) > 0:
            for t, pos in enumerate(bidx):
                if self.rng.random() > self.prob:
                    continue
                cell = self.cell_names[pos]
                batch = self.batch_ids[pos]
                rpool = self.pool.get((cell, batch))
                if rpool is None:
                    rpool = self.pool.get(cell)
                if rpool is None:
                    rpool = self.pool.get("__global__")
                if rpool is None or len(rpool) == 0:
                    continue
                r = rpool[self.rng.integers(0, len(rpool))]
                ctl[t] = ctl[t] + self.alpha * r

        if self.drug_fp is None:
            x = (ctl, self.drug_targets[bidx], self.cell_idx[bidx])
        else:
            x = (ctl, self.drug_targets[bidx], self.cell_idx[bidx], self.drug_fp[bidx])
        y = self.y_true[bidx]
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="MCF7,LNCAP")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--use_drug_fp_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_drug_embedding", dest="use_drug_fp_embedding", action="store_true")
    parser.add_argument("--no-use_drug_embedding", dest="use_drug_fp_embedding", action="store_false")
    parser.add_argument("--sparse_gat", action="store_true", default=False)
    parser.add_argument("--ctl_aug_pool", type=int, default=3)
    parser.add_argument("--ctl_aug_prob", type=float, default=0.3)
    parser.add_argument("--ctl_aug_alpha", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention_layers", type=int, default=4)
    parser.add_argument("--per_node_head", action="store_true", default=True)
    parser.add_argument("--run_eagerly", action="store_true", default=True)
    parser.add_argument("--no_residualize_target_by_cell", action="store_true", default=False)
    parser.add_argument("--no_cell_embedding", action="store_true", default=False)
    parser.add_argument("--omnipath_consensus_only", action="store_true", default=False)
    parser.add_argument("--omnipath_is_directed_only", action="store_true", default=False)
    parser.add_argument("--split_mode", choices=["cold_drug", "cold_cell"], default="cold_drug")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--eval_drug_zero", action="store_true", default=False)
    parser.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    parser.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    parser.add_argument("--eval_sanity_seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    if float(args.ctl_aug_prob) > 0.0 and int(args.ctl_aug_pool) <= 0:
        raise ValueError("开启 ctl augmentation 需要设置 --ctl_aug_pool > 0")

    root = args.root
    if not os.path.exists(root):
        root = '/local/data1/liume102/rfa'
        if not os.path.exists(root):
             root = '/local/data1/liume102/src'
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from base_gnn import BaseLineGAT
    from data_loader import load_rfa_data, build_combined_gnn

    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    string_path = os.path.join(root, "data/string_interactions_mapped.csv")
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
        ctl_residual_pool_size=int(args.ctl_aug_pool),
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        string_path=None,
        target_genes=data["target_genes"],
        confid_threshold=0.9,
        directed=True,
        omnipath_consensus_only=bool(args.omnipath_consensus_only),
        omnipath_is_directed_only=bool(args.omnipath_is_directed_only),
        symbol_to_entrez=data.get("symbol_to_entrez"),
    )
    if len(node_list) != len(data["target_genes"]) or node_list[:50] != data["target_genes"][:50]:
        raise ValueError("Graph node_list 与表达 target_genes 顺序/长度不一致")

    X_ctl = np.asarray(data["X_ctl"])
    y_delta = np.asarray(data["y_delta"])
    X_drug = np.asarray(data["X_drug"])
    X_fp = data.get("X_fingerprint")
    X_fp = None if X_fp is None else np.asarray(X_fp)
    fp_table = data.get("drug_fp_table")
    fp_idx = data.get("drug_fp_idx")
    fp_table = None if fp_table is None else np.asarray(fp_table, dtype=np.float32)
    fp_idx = None if fp_idx is None else np.asarray(fp_idx, dtype=np.int32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names_arr = np.asarray(data["cell_names"], dtype=str)
    batch_ids_arr = np.asarray(data.get("batch_ids", ["Unknown"] * len(drug_ids)), dtype=str)

    if not bool(args.use_landmark_genes):
        print(
            "WARNING: use_landmark_genes=False (full genes). 当前实现会构建 dense adj(N,N) 并计算 scores(B,N,N)，"
            "对 N=12328 需要极大内存/显存。建议在服务器上运行或改为基于 edge_index 的稀疏 GAT。"
        )

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
        cell_names_arr = cell_names_arr[idx]
        batch_ids_arr = batch_ids_arr[idx]

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names_arr)
    num_cells = int(len(le.classes_))

    split_mode = str(args.split_mode)
    test_frac = float(args.test_frac)
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")

    np.random.seed(42)
    if split_mode == "cold_cell":
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise ValueError("cold_cell 需要至少 2 个细胞系；请设置 --cell_line ALL 并避免过小的 --max_samples")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        test_cells_set = np.random.choice(unique_cells, n_test, replace=False)
        test_mask = np.isin(cell_idx, test_cells_set)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(test_cells_set)}/{len(unique_cells)}")
    else:
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise ValueError("cold_drug 需要至少 2 个药物；请避免过小的 --max_samples")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        test_drugs = np.random.choice(unique_drugs, n_test, replace=False)
        test_mask = np.isin(drug_ids, test_drugs)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(test_drugs)}/{len(unique_drugs)}")

    train_ctl = X_ctl[train_mask]
    train_trt_full = y_delta[train_mask]
    train_drug = X_drug[train_mask]
    train_cells = cell_idx[train_mask]
    if X_fp is not None:
        train_fp = X_fp[train_mask]
    else:
        train_fp = None if fp_idx is None else fp_idx[train_mask]

    test_ctl = X_ctl[test_mask]
    test_trt_full = y_delta[test_mask]
    test_drug = X_drug[test_mask]
    test_cells = cell_idx[test_mask]
    if X_fp is not None:
        test_fp = X_fp[test_mask]
    else:
        test_fp = None if fp_idx is None else fp_idx[test_mask]

    residualize_target = not bool(args.no_residualize_target_by_cell)
    cell_delta_mean = None
    if residualize_target:
        num_genes = int(train_trt_full.shape[1])
        sums = np.zeros((num_cells, num_genes), dtype=np.float32)
        counts = np.zeros((num_cells,), dtype=np.int64)
        np.add.at(sums, train_cells, train_trt_full)
        np.add.at(counts, train_cells, 1)
        denom = np.maximum(counts[:, None], 1)
        cell_delta_mean = sums / denom
        train_trt = train_trt_full - cell_delta_mean[train_cells]
        test_trt = test_trt_full - cell_delta_mean[test_cells]
    else:
        train_trt = train_trt_full
        test_trt = test_trt_full

    if X_fp is not None:
        fp_dim = int(X_fp.shape[1])
    elif fp_table is not None:
        fp_dim = int(fp_table.shape[1])
    else:
        fp_dim = 0
    if args.use_drug_fp_embedding and fp_dim <= 0:
        raise RuntimeError("use_drug_embedding=True 但指纹不存在（X_fingerprint 与 drug_fp_table 均为空）")

    model = BaseLineGAT(
        num_genes=int(adj_matrix.shape[0]),
        num_cells=num_cells,
        fingerprint_dim=fp_dim,
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        use_residual=False, # 这里不要设成True,当前预测的是delta, 而不是绝对的表达值
        use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
        attention_layer_number=int(args.attention_layers),
        per_node_embedding=bool(args.per_node_head),
        use_sparse_adj=bool(args.sparse_gat),
        use_cell_embedding=not bool(args.no_cell_embedding),
    )

    loss_mask = tf.constant(data["loss_mask"], dtype=tf.float32)

    def pcc_loss(y_true, y_pred):
        mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
        my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        xm = y_true - mx
        ym = y_pred - my
        r_num = tf.reduce_sum(xm * ym, axis=1)
        r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8)
        r = r_num / r_den
        return 1.0 - tf.reduce_mean(r)

    def masked_combined_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(loss_mask, tf.float32)

        mse = tf.reduce_sum(tf.square(y_true - y_pred) * mask)
        valid_count = tf.reduce_sum(mask)
        batch_n = tf.cast(tf.shape(y_true)[0], tf.float32)
        mse = mse / tf.maximum(valid_count * batch_n, 1.0)
        valid_indices = tf.where(loss_mask[0] > 0)[:, 0]
        yt = tf.gather(y_true, valid_indices, axis=1)
        yp = tf.gather(y_pred, valid_indices, axis=1)
        pcc = pcc_loss(yt, yp)
        return mse + 5.0 * pcc

    if bool(args.sparse_gat):
        edge_index_np = edge_index.astype(np.int64)
        src = edge_index_np[0]
        dst = edge_index_np[1]
        edge_weight = adj_matrix[dst, src].astype(np.float32)
        n = int(adj_matrix.shape[0])
        self_idx = np.arange(n, dtype=np.int64)
        edge_index_full = np.concatenate([edge_index_np, np.stack([self_idx, self_idx], axis=0)], axis=1)
        edge_weight_full = np.concatenate([edge_weight, np.ones((n,), dtype=np.float32)], axis=0)
        wrapped_model = GATWrapperSparse(
            model,
            edge_index_full,
            edge_weight_full,
            use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
            fp_table=fp_table,
        )
        adj_matrix = None
    else:
        wrapped_model = GATWrapper(model, adj_matrix, use_drug_fp_embedding=bool(args.use_drug_fp_embedding), fp_table=fp_table)
    wrapped_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=masked_combined_loss,
        metrics=[keras.metrics.MeanSquaredError()],
        run_eagerly=bool(args.run_eagerly),
    )

    if args.use_drug_fp_embedding:
        pcc_cb = PCCCallback(
            loss_mask=data["loss_mask"],
            train_data=(train_ctl, train_drug, train_cells, train_fp, train_trt),
            val_data=(test_ctl, test_drug, test_cells, test_fp, test_trt),
            batch_size=int(args.batch_size),
            max_eval=2048,
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )
    else:
        pcc_cb = PCCCallback(
            loss_mask=data["loss_mask"],
            train_data=(train_ctl, train_drug, train_cells, train_trt),
            val_data=(test_ctl, test_drug, test_cells, test_trt),
            batch_size=int(args.batch_size),
            max_eval=2048,
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )

    if args.use_drug_fp_embedding:
        wrapped_model.fit(
            CtlResidualAugmentDataset(
                ctl_mean=train_ctl,
                drug_targets=train_drug,
                cell_idx=train_cells,
                drug_fp=train_fp,
                y_true=train_trt,
                cell_names=cell_names_arr[train_mask],
                batch_ids=batch_ids_arr[train_mask],
                ctl_residual_pool=data.get("ctl_residual_pool"),
                batch_size=int(args.batch_size),
                alpha=float(args.ctl_aug_alpha),
                prob=float(args.ctl_aug_prob),
                seed=42,
            )
            if float(args.ctl_aug_prob) > 0.0
            else [train_ctl, train_drug, train_cells, train_fp],
            None if float(args.ctl_aug_prob) > 0.0 else train_trt,
            epochs=int(args.epochs),
            batch_size=None if float(args.ctl_aug_prob) > 0.0 else int(args.batch_size),
            callbacks=[pcc_cb],
            validation_data=([test_ctl, test_drug, test_cells, test_fp], test_trt),
            verbose=0,
        )
    else:
        wrapped_model.fit(
            CtlResidualAugmentDataset(
                ctl_mean=train_ctl,
                drug_targets=train_drug,
                cell_idx=train_cells,
                drug_fp=None,
                y_true=train_trt,
                cell_names=cell_names_arr[train_mask],
                batch_ids=batch_ids_arr[train_mask],
                ctl_residual_pool=data.get("ctl_residual_pool"),
                batch_size=int(args.batch_size),
                alpha=float(args.ctl_aug_alpha),
                prob=float(args.ctl_aug_prob),
                seed=42,
            )
            if float(args.ctl_aug_prob) > 0.0
            else [train_ctl, train_drug, train_cells],
            None if float(args.ctl_aug_prob) > 0.0 else train_trt,
            epochs=int(args.epochs),
            batch_size=None if float(args.ctl_aug_prob) > 0.0 else int(args.batch_size),
            callbacks=[pcc_cb],
            validation_data=([test_ctl, test_drug, test_cells], test_trt),
            verbose=0,
        )

    train_metrics = eval_pcc_mse(
        wrapped_model,
        train_ctl,
        train_drug,
        train_cells,
        train_trt,
        data["loss_mask"],
        batch_size=int(args.batch_size),
        max_eval=20000,
        drug_fp=(train_fp if bool(args.use_drug_fp_embedding) else None),
        cell_mean=cell_delta_mean,
        y_is_residual=residualize_target,
    )
    test_metrics = eval_pcc_mse(
        wrapped_model,
        test_ctl,
        test_drug,
        test_cells,
        test_trt,
        data["loss_mask"],
        batch_size=int(args.batch_size),
        max_eval=20000,
        drug_fp=(test_fp if bool(args.use_drug_fp_embedding) else None),
        cell_mean=cell_delta_mean,
        y_is_residual=residualize_target,
    )
    print(f"Train | MSE: {train_metrics['mse']:.4f} | Sample-wise PCC: {train_metrics['pcc']:.4f}")
    print(f"Test  | MSE: {test_metrics['mse']:.4f} | Sample-wise PCC: {test_metrics['pcc']:.4f}")

    if bool(args.eval_drug_zero):
        zero_drug = np.zeros_like(test_drug, dtype=np.float32)
        if bool(args.use_drug_fp_embedding) and fp_dim > 0:
            zero_fp = np.zeros((len(test_ctl), int(fp_dim)), dtype=np.float32)
        else:
            zero_fp = None
        m = eval_pcc_mse(
            wrapped_model,
            test_ctl,
            zero_drug,
            test_cells,
            test_trt,
            data["loss_mask"],
            batch_size=int(args.batch_size),
            max_eval=int(args.eval_sanity_max_eval),
            drug_fp=zero_fp,
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )
        print(f"Sanity(drug_zero) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")

    if bool(args.eval_drug_shuffle):
        rng = np.random.default_rng(int(args.eval_sanity_seed))
        n = len(test_ctl)
        perm = rng.permutation(n)
        shuf_drug = test_drug[perm]
        if bool(args.use_drug_fp_embedding):
            shuf_fp = None if test_fp is None else test_fp[perm]
        else:
            shuf_fp = None
        m = eval_pcc_mse(
            wrapped_model,
            test_ctl,
            shuf_drug,
            test_cells,
            test_trt,
            data["loss_mask"],
            batch_size=int(args.batch_size),
            max_eval=int(args.eval_sanity_max_eval),
            drug_fp=shuf_fp,
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )
        print(f"Sanity(drug_shuffle) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")


if __name__ == "__main__":
    main()
