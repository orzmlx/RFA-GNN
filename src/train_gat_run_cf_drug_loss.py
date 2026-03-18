import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import json
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
            inner = getattr(getattr(self.model, "core", None), "gat", None)
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

        extra = []
        for k in ["loss_full", "loss_cf", "cf_gap", "loss_hinge"]:
            v = logs.get(k)
            if v is not None:
                try:
                    extra.append(f"{k}={float(v):.4f}")
                except Exception:
                    pass
        if target_scale is not None:
            extra.append(f"target_scale={target_scale:.4f}")
        if len(extra) == 0:
            print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f}")
        else:
            print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f} " + " ".join(extra))


class GATWrapper(keras.Model):
    def __init__(self, gat_model, adj_matrix, use_drug_fp_embedding=False, fp_table=None):
        super().__init__()
        self.gat = gat_model
        self.adj = tf.constant(adj_matrix, dtype=tf.float32)
        self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
        self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

    def call(self, inputs, training=False):
        if self.use_drug_fp_embedding:
            ctl, drug_targets, cell_idx, drug_fp = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
            return self.gat([self.adj, ctl, drug_targets, cell_idx, drug_fp], training=training)
        ctl, drug_targets, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        return self.gat([self.adj, ctl, drug_targets, cell_idx], training=training)


class GATWrapperSparse(keras.Model):
    def __init__(self, gat_model, edge_index, edge_weight, use_drug_fp_embedding=False, fp_table=None):
        super().__init__()
        self.gat = gat_model
        self.edge_index = tf.constant(edge_index, dtype=tf.int32)
        self.edge_weight = tf.constant(edge_weight, dtype=tf.float32)
        self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
        self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

    def call(self, inputs, training=False):
        if self.use_drug_fp_embedding:
            ctl, drug_targets, cell_idx, drug_fp = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
            return self.gat([self.edge_index, self.edge_weight, ctl, drug_targets, cell_idx, drug_fp], training=training)
        ctl, drug_targets, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        return self.gat([self.edge_index, self.edge_weight, ctl, drug_targets, cell_idx], training=training)


"""
## Counterfactual drug hinge regularization

`DrugContrastiveTrainer` implements a form of **counterfactual regularization**. The goal is to encourage the model to
**actually use drug information** (targets / fingerprints) at prediction time, instead of fitting the training set mostly from
the control expression and the cell embedding.

### core operation

For the same minibatch we run two forward passes:

- factual input: `x = (ctl, drug, cell[, fp])`, producing `y_full`
- counterfactual input: we only corrupt drug information, `x_cf = (ctl, drug_cf, cell[, fp_cf])`, producing `y_cf`

The counterfactual `drug_cf` is constructed according to `cf_mode`:

- `zero`: replace drug targets / fingerprint with zeros
- `shuffle`: shuffle drug targets / fingerprint within the minibatch (equivalent to providing the wrong drug)

Importantly, **`ctl` and `cell_idx` are kept identical** between factual and counterfactual inputs; only drug inputs are modified.

### 为什么要 hinge（margin）约束

Let the supervised loss be `L(·)` (here: masked MSE + a PCC term). We compute:

- `loss_full = L(y, y_full)`
- `loss_cf   = L(y, y_cf)`
- `gap = loss_cf - loss_full`
- `hinge = relu(cf_margin - gap)`
- `loss = loss_full + cf_lambda * hinge`

The hinge term enforces the margin constraint:

`loss_cf - loss_full >= cf_margin`

That is: **once drug information is corrupted, the error against the true label must increase by at least a margin**.

This helps prevent a common failure mode where the model largely ignores drug inputs, making `y_full ≈ y_cf` and thus
`loss_full ≈ loss_cf`. In that case `gap` is small, the hinge stays positive, and an extra penalty is applied.
To reduce this penalty to zero, the model must increase `gap`, and the only available difference between the two inputs is
the drug information. Therefore the model is pushed to learn prediction logic that is sensitive to drug inputs.

### 超参数含义

- `cf_margin`: how much worse the counterfactual must be compared with the factual (default: 0.1)
- `cf_lambda`: weight of the hinge penalty; larger values emphasize drug contribution more strongly

### 重要说明

This constraint encourages "factual is better than counterfactual", but it does not guarantee improved generalization in all cases.
After training, it should be validated with sanity checks (performance should drop under drug_zero / drug_shuffle) and with
cold drug / cold cell evaluation to confirm that the learned drug dependence is meaningful.
"""
class DrugContrastiveTrainer(keras.Model):
    def __init__(self, core_model, loss_mask, cf_mode="zero", eval_cf_mode=None, cf_lambda=1.0, cf_margin=0.1, fp_dim=0):
        super().__init__()
        self.core = core_model
        self.loss_mask = tf.constant(loss_mask, dtype=tf.float32)
        self.cf_mode = str(cf_mode)
        self.eval_cf_mode = str(eval_cf_mode) if eval_cf_mode is not None else str(cf_mode)
        self.cf_lambda = float(cf_lambda)
        self.cf_margin = float(cf_margin)
        self.fp_dim = int(fp_dim)
        self.metric_total = keras.metrics.Mean(name="loss")
        self.metric_full = keras.metrics.Mean(name="loss_full")
        self.metric_cf = keras.metrics.Mean(name="loss_cf")
        self.metric_hinge = keras.metrics.Mean(name="loss_hinge")
        self.metric_gap = keras.metrics.Mean(name="cf_gap")

    @property
    def metrics(self):
        return [self.metric_total, self.metric_full, self.metric_cf, self.metric_hinge, self.metric_gap]

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

    def _pcc_loss(self, y_true, y_pred):
        mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
        my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        xm = y_true - mx
        ym = y_pred - my
        r_num = tf.reduce_sum(xm * ym, axis=1)
        r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8)
        r = r_num / r_den
        return 1.0 - tf.reduce_mean(r)

    def _masked_combined_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(self.loss_mask, tf.float32)
        mse = tf.reduce_sum(tf.square(y_true - y_pred) * mask)
        valid_count = tf.reduce_sum(mask)
        batch_n = tf.cast(tf.shape(y_true)[0], tf.float32)
        mse = mse / tf.maximum(valid_count * batch_n, 1.0)
        valid_indices = tf.where(self.loss_mask[0] > 0)[:, 0]
        # y_pred 中按指定索引取出某些列
        yt = tf.gather(y_true, valid_indices, axis=1) 
        yp = tf.gather(y_pred, valid_indices, axis=1)
        pcc = self._pcc_loss(yt, yp)
        return mse + 5.0 * pcc

    def _make_counterfactual_inputs(self, x, mode=None):
        mode = self.cf_mode if mode is None else str(mode)
        if len(x) == 3:
            ctl, drug_targets, cell_idx = x
            drug_fp = None
        else:
            ctl, drug_targets, cell_idx, drug_fp = x

        if mode == "zero":
            drug_cf = tf.zeros_like(drug_targets, dtype=tf.float32)
            if drug_fp is None:
                return (ctl, drug_cf, cell_idx)
            if drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                fp_cf = tf.zeros((tf.shape(ctl)[0], self.fp_dim), dtype=tf.float32)
            else:
                fp_cf = tf.zeros_like(drug_fp, dtype=tf.float32)
            return (ctl, drug_cf, cell_idx, fp_cf)

        perm = tf.random.shuffle(tf.range(tf.shape(ctl)[0]))
        drug_cf = tf.gather(drug_targets, perm, axis=0)
        if drug_fp is None:
            return (ctl, drug_cf, cell_idx)
        fp_cf = tf.gather(drug_fp, perm, axis=0)
        return (ctl, drug_cf, cell_idx, fp_cf)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_full = self.core(x, training=True)
            x_cf = self._make_counterfactual_inputs(x, mode=self.cf_mode)
            y_cf = self.core(x_cf, training=True)
            loss_full = self._masked_combined_loss(y, y_full)
            loss_cf = self._masked_combined_loss(y, y_cf)
            gap = loss_cf - loss_full
            hinge = tf.nn.relu(self.cf_margin - gap)
            loss = loss_full + tf.cast(self.cf_lambda, tf.float32) * hinge
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.metric_total.update_state(loss)
        self.metric_full.update_state(loss_full)
        self.metric_cf.update_state(loss_cf)
        self.metric_hinge.update_state(hinge)
        self.metric_gap.update_state(gap)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_full = self.core(x, training=False)
        x_cf = self._make_counterfactual_inputs(x, mode=self.eval_cf_mode)
        y_cf = self.core(x_cf, training=False)
        loss_full = self._masked_combined_loss(y, y_full)
        loss_cf = self._masked_combined_loss(y, y_cf)
        gap = loss_cf - loss_full
        hinge = tf.nn.relu(self.cf_margin - gap)
        loss = loss_full + tf.cast(self.cf_lambda, tf.float32) * hinge
        self.metric_total.update_state(loss)
        self.metric_full.update_state(loss_full)
        self.metric_cf.update_state(loss_cf)
        self.metric_hinge.update_state(hinge)
        self.metric_gap.update_state(gap)
        return {m.name: m.result() for m in self.metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="MCF7,LNCAP")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_landmark_genes", action="store_true", default=False)
    parser.add_argument("--use_drug_fp_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_drug_embedding", dest="use_drug_fp_embedding", action="store_true")
    parser.add_argument("--no-use_drug_embedding", dest="use_drug_fp_embedding", action="store_false")
    parser.add_argument("--sparse_gat", action="store_true", default=False)
    parser.add_argument("--ctl_pair_k", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention_layers", type=int, default=4)
    parser.add_argument("--per_node_head", action="store_true", default=True)
    parser.add_argument("--run_eagerly", action="store_true", default=False)
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
    parser.add_argument("--save_gat_weights", default="")
    parser.add_argument("--save_meta_json", default="")
    parser.add_argument("--cf_mode", choices=["shuffle", "zero"], default="zero")
    parser.add_argument("--eval_cf_mode", choices=["", "shuffle", "zero"], default="")
    parser.add_argument("--cf_lambda", type=float, default=1.0)
    parser.add_argument("--cf_margin", type=float, default=0.1)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
        if not os.path.exists(root):
            root = "/local/data1/liume102/src"
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from base_gnn import BaseLineGAT
    from data_loader import load_rfa_data, build_combined_gnn

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
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
       # string_path=None,
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
    trt_distil_ids_arr = np.asarray(data.get("trt_distil_ids", [""] * len(drug_ids)), dtype=str)

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
        trt_distil_ids_arr = trt_distil_ids_arr[idx]

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names_arr)
    num_cells = int(len(le.classes_))

    split_mode = str(args.split_mode)
    test_frac = float(args.test_frac)
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")

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

    if trt_distil_ids_arr.size == 0 or (len(trt_distil_ids_arr) != len(X_ctl)):
        raise RuntimeError("trt_distil_ids 缺失或长度不匹配，无法保存 test_ids.npy")

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
        core = GATWrapperSparse(
            model,
            edge_index_full,
            edge_weight_full,
            use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
            fp_table=fp_table,
        )
        adj_matrix = None
    else:
        core = GATWrapper(model, adj_matrix, use_drug_fp_embedding=bool(args.use_drug_fp_embedding), fp_table=fp_table)

    trainer = DrugContrastiveTrainer(
        core_model=core,
        loss_mask=data["loss_mask"],
        cf_mode=str(args.cf_mode),
        eval_cf_mode=(str(args.eval_cf_mode).strip() or str(args.cf_mode)),
        cf_lambda=float(args.cf_lambda),
        cf_margin=float(args.cf_margin),
        fp_dim=int(fp_dim),
    )
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), run_eagerly=bool(args.run_eagerly))

    if bool(args.use_drug_fp_embedding):
        train_x = (train_ctl, train_drug, train_cells, train_fp)
        val_x = (test_ctl, test_drug, test_cells, test_fp)
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
        train_x = (train_ctl, train_drug, train_cells)
        val_x = (test_ctl, test_drug, test_cells)
        pcc_cb = PCCCallback(
            loss_mask=data["loss_mask"],
            train_data=(train_ctl, train_drug, train_cells, train_trt),
            val_data=(test_ctl, test_drug, test_cells, test_trt),
            batch_size=int(args.batch_size),
            max_eval=2048,
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )

    ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_trt)).shuffle(20000, seed=42, reshuffle_each_iteration=True).batch(int(args.batch_size))
    ds_val = tf.data.Dataset.from_tensor_slices((val_x, test_trt)).batch(int(args.batch_size))
    trainer.fit(ds_train, epochs=int(args.epochs), callbacks=[pcc_cb], validation_data=ds_val, verbose=0)

    train_metrics = eval_pcc_mse(
        trainer,
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
        trainer,
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
            trainer,
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
            trainer,
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

    if str(args.save_gat_weights).strip() != "":
        out_path = str(args.save_gat_weights).strip()
        out_dir = os.path.dirname(out_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        model.save_weights(out_path)
        print(f"Saved GAT weights to: {out_path}")

    if str(args.save_meta_json).strip() != "":
        out_path = str(args.save_meta_json).strip()
        out_dir = os.path.dirname(out_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        test_ids_path = os.path.splitext(out_path)[0] + ".test_ids.npy"
        np.save(test_ids_path, trt_distil_ids_arr[test_mask].astype(str))
        meta = {
            "cell_line": args.cell_line,
            "use_landmark_genes": bool(args.use_landmark_genes),
            "split_mode": str(args.split_mode),
            "test_frac": float(args.test_frac),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "sparse_gat": bool(args.sparse_gat),
            "use_drug_fp_embedding": bool(args.use_drug_fp_embedding),
            "ctl_pair_k": int(args.ctl_pair_k),
            "hidden_dim": int(args.hidden_dim),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "attention_layers": int(args.attention_layers),
            "per_node_head": bool(args.per_node_head),
            "no_cell_embedding": bool(args.no_cell_embedding),
            "no_residualize_target_by_cell": bool(args.no_residualize_target_by_cell),
            "cf_mode": str(args.cf_mode),
            "cf_lambda": float(args.cf_lambda),
            "cf_margin": float(args.cf_margin),
            "test_ids_npy": test_ids_path,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved run meta to: {out_path}")
        print(f"Saved test ids to: {test_ids_path}")


if __name__ == "__main__":
    main()
