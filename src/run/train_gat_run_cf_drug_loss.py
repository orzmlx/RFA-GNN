import argparse
import inspect
import os
import shutil
import sys
import tempfile
from base_gnn import BaseLineGAT
from data_loader import load_rfa_data, build_combined_gnn, subset_anchor_data, build_scheme_a_split_data
import numpy as np
import tensorflow as tf
import json
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from train_common import (
    append_split_suffix,
    parse_split_modes,
    samplewise_masked_metrics,
    save_npz,
    split_mean_logvar,
)
from train_tf_common import (
    GenericPCCCallback,
    GraphModelWrapper,
    build_uncertainty_stats_fn,
    collect_gat_scale_stats,
)


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
    pred, _ = split_mean_logvar(pred)

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
    mse = float(np.mean((y_true_valid - pred_valid) ** 2))
    return {"mse": mse, "pcc": avg_pcc}


def _eval_gat_pack(model, data_pack, batch_size, max_eval, loss_mask, cell_mean=None, y_is_residual=False):
    if len(data_pack) == 4:
        ctl, drug, cells, y_true = data_pack
        drug_fp = None
    else:
        ctl, drug, cells, drug_fp, y_true = data_pack
    return eval_pcc_mse(
        model,
        ctl,
        drug,
        cells,
        y_true,
        loss_mask,
        batch_size=batch_size,
        max_eval=max_eval,
        drug_fp=drug_fp,
        cell_mean=cell_mean,
        y_is_residual=y_is_residual,
    )


def _parse_csv_list(s):
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


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
    def __init__(
        self,
        core_model,
        loss_mask,
        cf_mode="zero",
        eval_cf_mode=None,
        cf_lambda=1.0,
        cf_margin=0.1,
        fp_dim=0,
        predict_uncertainty=False,
        pcc_lambda=5.0,
        logvar_clip_min=-6.0,
        logvar_clip_max=2.0,
    ):
        super().__init__()
        self.core = core_model
        self.loss_mask = tf.constant(loss_mask, dtype=tf.float32)
        self.cf_mode = str(cf_mode)
        self.eval_cf_mode = str(eval_cf_mode) if eval_cf_mode is not None else str(cf_mode)
        self.cf_lambda = float(cf_lambda)
        self.cf_margin = float(cf_margin)
        self.fp_dim = int(fp_dim)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.pcc_lambda = float(pcc_lambda)
        self.logvar_clip_min = float(logvar_clip_min)
        self.logvar_clip_max = float(logvar_clip_max)
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

    def _split_prediction(self, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        if self.predict_uncertainty:
            if len(y_pred.shape) != 3 or y_pred.shape[-1] != 2:
                raise ValueError("predict_uncertainty=True requires model output shape (B, N, 2)")
            return y_pred[..., 0], y_pred[..., 1]
        return y_pred, None

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
        mean_pred, logvar_pred = self._split_prediction(y_pred)
        mask = tf.cast(self.loss_mask, tf.float32)
        valid_count = tf.reduce_sum(mask)
        batch_n = tf.cast(tf.shape(y_true)[0], tf.float32)
        if self.predict_uncertainty:
            logvar_pred = tf.clip_by_value(logvar_pred, self.logvar_clip_min, self.logvar_clip_max)
            inv_var = tf.exp(-logvar_pred)
            nll = 0.5 * (logvar_pred + tf.square(y_true - mean_pred) * inv_var)
            base_loss = tf.reduce_sum(nll * mask) / tf.maximum(valid_count * batch_n, 1.0)
        else:
            mse = tf.reduce_sum(tf.square(y_true - mean_pred) * mask)
            base_loss = mse / tf.maximum(valid_count * batch_n, 1.0)
        valid_indices = tf.where(self.loss_mask[0] > 0)[:, 0]
        yt = tf.gather(y_true, valid_indices, axis=1) 
        yp = tf.gather(mean_pred, valid_indices, axis=1)
        pcc = self._pcc_loss(yt, yp)
        return base_loss + tf.cast(self.pcc_lambda, tf.float32) * pcc

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
        loss_gap = tf.nn.relu(self.cf_margin - gap)
        loss = loss_full + tf.cast(self.cf_lambda, tf.float32) * loss_gap
        self.metric_total.update_state(loss)
        self.metric_full.update_state(loss_full)
        self.metric_cf.update_state(loss_cf)
        self.metric_hinge.update_state(loss_gap)
        self.metric_gap.update_state(gap)
        return {m.name: m.result() for m in self.metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="MCF7,LNCAP")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_landmark_genes", dest="use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--no-use_landmark_genes", dest="use_landmark_genes", action="store_false")
    parser.add_argument("--use_drug_fp_embedding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_drug_embedding", dest="use_drug_fp_embedding", action="store_true")
    parser.add_argument("--no-use_drug_embedding", dest="use_drug_fp_embedding", action="store_false")
    parser.add_argument("--sparse_gat", action="store_true", default=False)
    parser.add_argument("--ctl_pair_k", type=int, default=3)
    parser.add_argument("--pairing_mode", choices=["multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"], default="multi_trt_multi_ctl")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cell_dropout_rate", type=float, default=0.3)
    parser.add_argument("--attention_layers", type=int, default=4)
    parser.add_argument("--per_node_head", action="store_true", default=True)
    parser.add_argument("--run_eagerly", action="store_true", default=False)
    parser.add_argument("--no_residualize_target_by_cell", action="store_true", default=False)
    parser.add_argument("--no_cell_embedding", action="store_true", default=False)
    parser.add_argument("--omnipath_consensus_only", action="store_true", default=False)
    parser.add_argument("--omnipath_is_directed_only", action="store_true", default=False)
    parser.add_argument("--split_modes", default="warm,cold_target_pattern,cold_cell")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--eval_drug_zero", action="store_true", default=True)
    parser.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    parser.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    parser.add_argument("--eval_sanity_seed", type=int, default=0)
    parser.add_argument("--save_gat_weights", default="")
    parser.add_argument("--save_meta_json", default="")
    parser.add_argument("--save_eval_npz", default="")
    parser.add_argument("--export_attention", action="store_true", default=False)
    parser.add_argument("--attention_max_samples", type=int, default=2000)
    parser.add_argument("--attention_batch_size", type=int, default=64)
    parser.add_argument("--attention_group_by", choices=["", "drug", "cell"], default="")
    parser.add_argument("--attention_groups", default="")
    parser.add_argument("--attention_top_k_groups", type=int, default=12)
    parser.add_argument("--cf_mode", choices=["shuffle", "zero"], default="zero")
    parser.add_argument("--eval_cf_mode", choices=["", "shuffle", "zero"], default="")
    parser.add_argument("--cf_lambda", type=float, default=1.0)
    parser.add_argument("--cf_margin", type=float, default=0.1)
    parser.add_argument("--predict_uncertainty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pcc_lambda", type=float, default=5.0)
    parser.add_argument("--logvar_clip_min", type=float, default=-6.0)
    parser.add_argument("--logvar_clip_max", type=float, default=2.0)
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
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    if bool(args.sparse_gat):
        adj_matrix, node_list, _gene2idx, edge_index, edge_weight = build_combined_gnn(
            tf_path=tf_path,
            ppi_path=ppi_path,
           # string_path=None,
            target_genes=data["target_genes"],
            confid_threshold=0.9,
            directed=True,
            omnipath_consensus_only=bool(args.omnipath_consensus_only),
            omnipath_is_directed_only=bool(args.omnipath_is_directed_only),
            symbol_to_entrez=data.get("symbol_to_entrez"),
            return_edge_weight=True,
        )
    else:
        adj_matrix, node_list, _gene2idx, edge_index = build_combined_gnn(
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
        edge_weight = None
    if len(node_list) != len(data["target_genes"]) or node_list[:50] != data["target_genes"][:50]:
        raise ValueError("Graph node_list 与表达 target_genes 顺序/长度不一致")

    anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
    anchor_cell_names_arr = np.asarray(data["anchor_cell_names"], dtype=str)
    fp_table = data.get("drug_fp_table")
    fp_table = None if fp_table is None else np.asarray(fp_table, dtype=np.float32)

    if int(args.max_samples) > 0 and len(anchor_drug_ids) > int(args.max_samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(anchor_drug_ids), size=int(args.max_samples), replace=False)
        data = subset_anchor_data(data, idx)
        anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
        anchor_cell_names_arr = np.asarray(data["anchor_cell_names"], dtype=str)

    le = LabelEncoder()
    le.fit(anchor_cell_names_arr)
    num_cells = int(len(le.classes_))

    split_modes = parse_split_modes(args.split_modes, "cold_target_pattern")
    fp_dim = int(fp_table.shape[1]) if fp_table is not None else 0
    if args.use_drug_fp_embedding and fp_dim <= 0:
        raise RuntimeError("use_drug_embedding=True 但指纹不存在（X_fingerprint 与 drug_fp_table 均为空）")

    all_results = []
    for split_mode in split_modes:
        keras.backend.clear_session()
        tf.random.set_seed(42)
        print(f"\n===== Running split: {split_mode} =====")
        train_data, test_data, train_anchor_mask, test_anchor_mask = build_scheme_a_split_data(
            data=data,
            split_mode=split_mode,
            test_frac=float(args.test_frac),
            seed=42,
            train_pairing_mode=str(args.pairing_mode),
            train_ctl_pair_k=int(args.ctl_pair_k),
            test_pairing_mode="unique_trt_reuse_ctl",
        )

        train_ctl = np.asarray(train_data["X_ctl"], dtype=np.float32)
        train_trt_full = np.asarray(train_data["y_delta"], dtype=np.float32)
        train_drug = np.asarray(train_data["X_drug"], dtype=np.float32)
        train_cells = le.transform(np.asarray(train_data["cell_names"], dtype=str))
        train_fp = train_data.get("X_fingerprint")
        train_fp = None if train_fp is None else np.asarray(train_fp)
        if train_fp is None:
            train_fp = np.asarray(train_data["drug_fp_idx"], dtype=np.int32)

        test_ctl = np.asarray(test_data["X_ctl"], dtype=np.float32)
        test_trt_full = np.asarray(test_data["y_delta"], dtype=np.float32)
        test_drug = np.asarray(test_data["X_drug"], dtype=np.float32)
        test_cells = le.transform(np.asarray(test_data["cell_names"], dtype=str))
        test_fp = test_data.get("X_fingerprint")
        test_fp = None if test_fp is None else np.asarray(test_fp)
        if test_fp is None:
            test_fp = np.asarray(test_data["drug_fp_idx"], dtype=np.int32)
        test_drug_ids_arr = np.asarray(test_data.get("drug_ids", ["Unknown"] * len(test_ctl)), dtype=str)
        test_cell_names_arr = np.asarray(test_data.get("cell_names", ["Unknown"] * len(test_ctl)), dtype=str)
        test_batch_ids_arr = np.asarray(test_data.get("batch_ids", ["Unknown"] * len(test_ctl)), dtype=str)
        test_trt_distil_ids_arr = np.asarray(test_data.get("trt_distil_ids", [""] * len(test_ctl)), dtype=str)

        residualize_target = not bool(args.no_residualize_target_by_cell)
        cell_delta_mean = None
        # if residualize_target:
        #     num_genes = int(train_trt_full.shape[1])
        #     sums = np.zeros((num_cells, num_genes), dtype=np.float32)
        #     counts = np.zeros((num_cells,), dtype=np.int64)
        #     np.add.at(sums, train_cells, train_trt_full)
        #     np.add.at(counts, train_cells, 1)
        #     denom = np.maximum(counts[:, None], 1)
        #     cell_delta_mean = sums / denom
        #     train_trt = train_trt_full - cell_delta_mean[train_cells]
        #     test_trt = test_trt_full - cell_delta_mean[test_cells]
        # else:
        train_trt = train_trt_full
        test_trt = test_trt_full

        model_kwargs = dict(
            num_genes=int(len(node_list)),
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
        effective_predict_uncertainty = False
        try:
            sig = inspect.signature(BaseLineGAT.__init__)
            if "cell_dropout_rate" in sig.parameters:
                model_kwargs["cell_dropout_rate"] = float(args.cell_dropout_rate)
            if "predict_uncertainty" in sig.parameters:
                effective_predict_uncertainty = bool(args.predict_uncertainty)
                model_kwargs["predict_uncertainty"] = effective_predict_uncertainty
            elif bool(args.predict_uncertainty):
                print("Warning: current BaseLineGAT implementation does not support uncertainty output; falling back to deterministic output.")
        except Exception:
            if bool(args.predict_uncertainty):
                print("Warning: failed to inspect BaseLineGAT signature; falling back to deterministic output.")
        model = BaseLineGAT(**model_kwargs)

        if bool(args.sparse_gat):
            edge_index_np = edge_index.astype(np.int64)
            edge_weight_np = np.asarray(edge_weight, dtype=np.float32)
            n = int(len(node_list))
            self_idx = np.arange(n, dtype=np.int64)
            edge_index_full = np.concatenate([edge_index_np, np.stack([self_idx, self_idx], axis=0)], axis=1)
            edge_weight_full = np.concatenate([edge_weight_np, np.ones((n,), dtype=np.float32)], axis=0)
            core = GraphModelWrapper(
                model,
                graph_inputs=[edge_index_full, edge_weight_full],
                graph_input_dtypes=[tf.int32, tf.float32],
                use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
                fp_table=fp_table,
                pass_training=True,
            )
        else:
            core = GraphModelWrapper(
                model,
                graph_inputs=[adj_matrix],
                graph_input_dtypes=[tf.float32],
                use_drug_fp_embedding=bool(args.use_drug_fp_embedding),
                fp_table=fp_table,
                pass_training=True,
            )

        trainer = DrugContrastiveTrainer(
            core_model=core,
            loss_mask=data["loss_mask"],
            cf_mode=str(args.cf_mode),
            eval_cf_mode=(str(args.eval_cf_mode).strip() or str(args.cf_mode)),
            cf_lambda=float(args.cf_lambda),
            cf_margin=float(args.cf_margin),
            fp_dim=int(fp_dim),
            predict_uncertainty=effective_predict_uncertainty,
            pcc_lambda=float(args.pcc_lambda),
            logvar_clip_min=float(args.logvar_clip_min),
            logvar_clip_max=float(args.logvar_clip_max),
        )
        trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), run_eagerly=bool(args.run_eagerly))
        best_ckpt_dir = tempfile.mkdtemp(prefix=f"gat_best_{split_mode}_")
        best_weights_path = os.path.join(best_ckpt_dir, "best_val_pcc.weights.h5")

        if bool(args.use_drug_fp_embedding):
            train_x = (train_ctl, train_drug, train_cells, train_fp)
            val_x = (test_ctl, test_drug, test_cells, test_fp)
            pcc_cb = GenericPCCCallback(
                train_data=(train_ctl, train_drug, train_cells, train_fp, train_trt),
                val_data=(test_ctl, test_drug, test_cells, test_fp, test_trt),
                evaluate_pack_fn=lambda model_, pack, batch_size, max_eval: _eval_gat_pack(
                    model_,
                    pack,
                    batch_size=batch_size,
                    max_eval=max_eval,
                    loss_mask=data["loss_mask"],
                    cell_mean=cell_delta_mean,
                    y_is_residual=residualize_target,
                ),
                batch_size=int(args.batch_size),
                max_eval=2048,
                extra_log_keys=["loss_full", "loss_cf", "cf_gap", "loss_hinge"],
                scale_stats_fn=collect_gat_scale_stats,
                uncertainty_stats_fn=build_uncertainty_stats_fn(data["loss_mask"]),
                best_weights_path=best_weights_path,
            )
        else:
            train_x = (train_ctl, train_drug, train_cells)
            val_x = (test_ctl, test_drug, test_cells)
            pcc_cb = GenericPCCCallback(
                train_data=(train_ctl, train_drug, train_cells, train_trt),
                val_data=(test_ctl, test_drug, test_cells, test_trt),
                evaluate_pack_fn=lambda model_, pack, batch_size, max_eval: _eval_gat_pack(
                    model_,
                    pack,
                    batch_size=batch_size,
                    max_eval=max_eval,
                    loss_mask=data["loss_mask"],
                    cell_mean=cell_delta_mean,
                    y_is_residual=residualize_target,
                ),
                batch_size=int(args.batch_size),
                max_eval=2048,
                extra_log_keys=["loss_full", "loss_cf", "cf_gap", "loss_hinge"],
                scale_stats_fn=collect_gat_scale_stats,
                uncertainty_stats_fn=build_uncertainty_stats_fn(data["loss_mask"]),
                best_weights_path=best_weights_path,
            )

        ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_trt)).shuffle(20000, seed=42, reshuffle_each_iteration=True).batch(int(args.batch_size))
        ds_val = tf.data.Dataset.from_tensor_slices((val_x, test_trt)).batch(int(args.batch_size))
        trainer.fit(ds_train, epochs=int(args.epochs), callbacks=[pcc_cb], validation_data=ds_val, verbose=0)
        if pcc_cb.best_epoch is not None and os.path.exists(best_weights_path):
            trainer.load_weights(best_weights_path)
            print(
                f"Reloaded best checkpoint from epoch {pcc_cb.best_epoch} "
                f"(val_pcc={pcc_cb.best_val_pcc:.4f}, val_mse={pcc_cb.best_val_mse:.4f})"
            )

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

        sanity_drug_zero_metrics = None
        if bool(args.eval_drug_zero):
            zero_drug = np.zeros_like(test_drug, dtype=np.float32)
            if bool(args.use_drug_fp_embedding) and fp_dim > 0:
                zero_fp = np.zeros((len(test_ctl), int(fp_dim)), dtype=np.float32)
            else:
                zero_fp = None
            sanity_drug_zero_metrics = eval_pcc_mse(
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
            m = sanity_drug_zero_metrics
            print(f"Sanity(drug_zero) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")

        sanity_drug_shuffle_metrics = None
        if bool(args.eval_drug_shuffle):
            rng = np.random.default_rng(int(args.eval_sanity_seed))
            n = len(test_ctl)
            perm = rng.permutation(n)
            shuf_drug = test_drug[perm]
            if bool(args.use_drug_fp_embedding):
                shuf_fp = None if test_fp is None else test_fp[perm]
            else:
                shuf_fp = None
            sanity_drug_shuffle_metrics = eval_pcc_mse(
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
            m = sanity_drug_shuffle_metrics
            print(f"Sanity(drug_shuffle) | MSE: {m['mse']:.4f} | Sample-wise PCC: {m['pcc']:.4f}")

        if str(args.save_gat_weights).strip() != "":
            out_path = append_split_suffix(args.save_gat_weights, split_mode)
            out_dir = os.path.dirname(out_path)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
            model.save_weights(out_path)
            print(f"Saved GAT weights to: {out_path}")

        if str(args.save_meta_json).strip() != "":
            out_path = append_split_suffix(args.save_meta_json, split_mode)
            out_dir = os.path.dirname(out_path)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
            test_ids_path = os.path.splitext(out_path)[0] + ".test_ids.npy"
            np.save(test_ids_path, test_trt_distil_ids_arr.astype(str))
            meta = {
                "cell_line": args.cell_line,
                "use_landmark_genes": bool(args.use_landmark_genes),
                "split_mode": str(split_mode),
                "split_modes": split_modes,
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
                "predict_uncertainty": bool(effective_predict_uncertainty),
                "pcc_lambda": float(args.pcc_lambda),
                "logvar_clip_min": float(args.logvar_clip_min),
                "logvar_clip_max": float(args.logvar_clip_max),
                "pairing_mode": str(args.pairing_mode),
                "train_anchor_n": int(np.sum(train_anchor_mask)),
                "test_anchor_n": int(np.sum(test_anchor_mask)),
                "best_epoch": (None if pcc_cb.best_epoch is None else int(pcc_cb.best_epoch)),
                "best_val_pcc": (None if pcc_cb.best_val_pcc is None else float(pcc_cb.best_val_pcc)),
                "best_val_mse": (None if pcc_cb.best_val_mse is None else float(pcc_cb.best_val_mse)),
                "test_ids_npy": test_ids_path,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "sanity_drug_zero_metrics": sanity_drug_zero_metrics,
                "sanity_drug_shuffle_metrics": sanity_drug_shuffle_metrics,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved run meta to: {out_path}")
            print(f"Saved test ids to: {test_ids_path}")

        if str(args.save_eval_npz).strip() != "":
            out_path = append_split_suffix(args.save_eval_npz, split_mode)
            if bool(args.export_attention) and not bool(args.sparse_gat):
                raise ValueError("export_attention 目前只支持 --sparse_gat")
            if bool(args.use_drug_fp_embedding):
                y_pred_test = trainer.predict([test_ctl, test_drug, test_cells, test_fp], batch_size=256, verbose=0)
            else:
                y_pred_test = trainer.predict([test_ctl, test_drug, test_cells], batch_size=256, verbose=0)
            y_true_test = np.asarray(test_trt, dtype=np.float32)
            y_pred_test = np.asarray(y_pred_test, dtype=np.float32)
            y_pred_mean, y_pred_logvar = split_mean_logvar(y_pred_test)
            if residualize_target and cell_delta_mean is not None:
                cm = np.asarray(cell_delta_mean, dtype=np.float32)[test_cells]
                y_true_test = y_true_test + cm
                y_pred_mean = y_pred_mean + cm

            sample_pcc, sample_mse = samplewise_masked_metrics(y_true_test, y_pred_mean, data["loss_mask"])

            sanity = {}
            if sanity_drug_zero_metrics is not None:
                sanity["drug_zero"] = sanity_drug_zero_metrics
            if sanity_drug_shuffle_metrics is not None:
                sanity["drug_shuffle"] = sanity_drug_shuffle_metrics

            attention = {}
            if bool(args.export_attention):
                rng = np.random.default_rng(0)
                n_all = int(len(test_ctl))
                n_attn = min(int(args.attention_max_samples), n_all) if int(args.attention_max_samples) > 0 else n_all
                sel = rng.choice(n_all, size=n_attn, replace=False) if n_attn < n_all else np.arange(n_all)

                group_by = str(args.attention_group_by).strip().lower()
                group_labels = None
                if group_by == "drug":
                    group_labels = np.asarray(test_drug_ids_arr[sel], dtype=str)
                elif group_by == "cell":
                    group_labels = np.asarray(test_cell_names_arr[sel], dtype=str)
                keep_groups = None
                if group_labels is not None:
                    specified = _parse_csv_list(args.attention_groups)
                    if len(specified) > 0:
                        keep_groups = set([str(x) for x in specified])
                    else:
                        uniq, counts = np.unique(group_labels, return_counts=True)
                        order = uniq[np.argsort(-counts)]
                        keep_groups = set(order[: int(args.attention_top_k_groups)].tolist())

                edge_index_tf = tf.constant(edge_index_full.astype(np.int32), dtype=tf.int32)
                edge_weight_tf = tf.constant(edge_weight_full.astype(np.float32), dtype=tf.float32)

                layer_sums = None
                group_sums = None
                group_counts = None
                bs = int(args.attention_batch_size) if int(args.attention_batch_size) > 0 else 64
                for start in range(0, len(sel), bs):
                    end = min(start + bs, len(sel))
                    idx = sel[start:end]
                    b_ctl = tf.constant(test_ctl[idx], dtype=tf.float32)
                    b_drug = tf.constant(test_drug[idx], dtype=tf.float32)
                    b_cell = tf.constant(test_cells[idx], dtype=tf.int32)
                    if bool(args.use_drug_fp_embedding):
                        if isinstance(test_fp, np.ndarray) and test_fp.ndim == 2:
                            b_fp = tf.constant(test_fp[idx], dtype=tf.float32)
                        else:
                            b_fp = tf.constant(test_fp[idx], dtype=tf.int32)
                        _, attns = model([edge_index_tf, edge_weight_tf, b_ctl, b_drug, b_cell, b_fp], training=False, output_attention=True)
                    else:
                        _, attns = model([edge_index_tf, edge_weight_tf, b_ctl, b_drug, b_cell], training=False, output_attention=True)

                    att_np = []
                    for li, a in enumerate(attns):
                        a = tf.reduce_mean(a, axis=1).numpy().astype(np.float64)
                        att_np.append(a)
                        s = np.sum(a, axis=0)
                        if layer_sums is None:
                            layer_sums = [np.zeros_like(s, dtype=np.float64) for _ in range(len(attns))]
                        layer_sums[li] += s

                    if group_labels is not None and keep_groups is not None:
                        if group_sums is None:
                            group_sums = {g: [np.zeros((att_np[0].shape[1],), dtype=np.float64) for _ in range(len(attns))] for g in keep_groups}
                            group_counts = {g: 0 for g in keep_groups}
                        b_labels = group_labels[start:end]
                        for g in np.unique(b_labels):
                            if g not in keep_groups:
                                continue
                            m = np.where(b_labels == g)[0]
                            if len(m) == 0:
                                continue
                            group_counts[g] += int(len(m))
                            for li in range(len(attns)):
                                group_sums[g][li] += np.sum(att_np[li][m], axis=0)

                layer_means = [x / float(len(sel)) for x in layer_sums]
                attention = {
                    "attention_edge_mean": np.stack(layer_means, axis=0).astype(np.float32),
                    "edge_index": edge_index_full.astype(np.int32),
                    "edge_weight": edge_weight_full.astype(np.float32),
                    "attention_num_samples": int(len(sel)),
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

            meta_out = {
                "cell_line": args.cell_line,
                "use_landmark_genes": bool(args.use_landmark_genes),
                "split_mode": str(split_mode),
                "split_modes": split_modes,
                "pairing_mode": str(args.pairing_mode),
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
                "predict_uncertainty": bool(effective_predict_uncertainty),
                "pcc_lambda": float(args.pcc_lambda),
                "logvar_clip_min": float(args.logvar_clip_min),
                "logvar_clip_max": float(args.logvar_clip_max),
                "train_anchor_n": int(np.sum(train_anchor_mask)),
                "test_anchor_n": int(np.sum(test_anchor_mask)),
                "best_epoch": (None if pcc_cb.best_epoch is None else int(pcc_cb.best_epoch)),
                "best_val_pcc": (None if pcc_cb.best_val_pcc is None else float(pcc_cb.best_val_pcc)),
                "best_val_mse": (None if pcc_cb.best_val_mse is None else float(pcc_cb.best_val_mse)),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "export_attention": bool(args.export_attention),
                "attention_group_by": str(args.attention_group_by),
            }

            save_npz(
                out_path,
                y_true=y_true_test.astype(np.float32),
                y_pred=y_pred_mean.astype(np.float32),
                y_logvar=(np.zeros((0,), dtype=np.float32) if y_pred_logvar is None else y_pred_logvar.astype(np.float32)),
                sample_pcc=sample_pcc.astype(np.float32),
                sample_mse=sample_mse.astype(np.float32),
                cell_names=np.asarray(test_cell_names_arr, dtype=object),
                drug_ids=np.asarray(test_drug_ids_arr, dtype=object),
                batch_ids=np.asarray(test_batch_ids_arr, dtype=object),
                trt_distil_ids=np.asarray(test_trt_distil_ids_arr, dtype=object),
                target_genes=np.asarray(data["target_genes"], dtype=object),
                metrics=test_metrics,
                sanity=sanity,
                attention=attention,
                meta=meta_out,
            )
            print(f"Saved eval npz to: {out_path}")

        shutil.rmtree(best_ckpt_dir, ignore_errors=True)

        all_results.append(
            {
                "split_mode": split_mode,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "n_train": int(len(train_ctl)),
                "n_test": int(len(test_ctl)),
                "train_anchor_n": int(np.sum(train_anchor_mask)),
                "test_anchor_n": int(np.sum(test_anchor_mask)),
            }
        )

    if len(all_results) > 1:
        print("\n===== Summary =====")
        for r in all_results:
            print(
                f"{r['split_mode']}: "
                f"train_n={r['n_train']} test_n={r['n_test']} | "
                f"test_MSE={r['test_metrics']['mse']:.4f} | "
                f"test_PCC={r['test_metrics']['pcc']:.4f}"
            )


if __name__ == "__main__":
    main()
