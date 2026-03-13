import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import json


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
    out_dir = os.path.dirname(str(path))
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(path, **payload)


def _samplewise_metrics(y_true, y_pred, loss_mask):
    valid_indices = np.where(np.asarray(loss_mask)[0] > 0)[0]
    yt = y_true[:, valid_indices]
    yp = y_pred[:, valid_indices]
    pcc = np.zeros((len(yt),), dtype=np.float32)
    mse = np.zeros((len(yt),), dtype=np.float32)
    for i in range(len(yt)):
        a = yt[i]
        b = yp[i]
        mse[i] = float(np.mean((a - b) ** 2))
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            pcc[i] = float(pearsonr(a, b)[0])
        else:
            pcc[i] = 0.0
    return pcc, mse


def _parse_csv_list(s):
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


class CounterfactualTrainer(keras.Model):
    def __init__(
        self,
        core_model,
        loss_mask,
        cf_mode="shuffle",
        cf_lambda=1.0,
        cf_margin=0.1,
        fp_dim=0,
        aux_target_alpha=0.0,
        wmss_alpha=0.0,
        wmss_mode="residual",
        wmss_weak="drug_zero",
    ):
        super().__init__()
        self.core = core_model
        self.loss_mask = tf.constant(loss_mask, dtype=tf.float32)
        self.cf_mode = str(cf_mode)
        self.cf_lambda = float(cf_lambda)
        self.cf_margin = float(cf_margin)
        self.fp_dim = int(fp_dim)
        self.aux_target_alpha = float(aux_target_alpha)
        self.wmss_alpha = float(wmss_alpha)
        self.wmss_mode = str(wmss_mode)
        self.wmss_weak = str(wmss_weak)
        self.aux_target_head = layers.Dense(1)
        self.metric_total = keras.metrics.Mean(name="loss")
        self.metric_full = keras.metrics.Mean(name="loss_full")
        self.metric_cf = keras.metrics.Mean(name="loss_cf")
        self.metric_hinge = keras.metrics.Mean(name="loss_hinge")
        self.metric_aux = keras.metrics.Mean(name="loss_aux")
        self.metric_gap = keras.metrics.Mean(name="cf_gap")

    @property
    def metrics(self):
        return [self.metric_total, self.metric_full, self.metric_cf, self.metric_hinge, self.metric_aux, self.metric_gap]

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
        yt = tf.gather(y_true, valid_indices, axis=1)
        yp = tf.gather(y_pred, valid_indices, axis=1)
        pcc = self._pcc_loss(yt, yp)
        return mse + 5.0 * pcc

    def _make_counterfactual_inputs(self, inputs):
        if len(inputs) == 3:
            ctl, drug_targets, cell_idx = inputs
            drug_fp = None
        else:
            ctl, drug_targets, cell_idx, drug_fp = inputs

        if self.cf_mode == "zero":
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

    def _make_drug_zero_inputs(self, inputs):
        if len(inputs) == 3:
            ctl, drug_targets, cell_idx = inputs
            drug_fp = None
        else:
            ctl, drug_targets, cell_idx, drug_fp = inputs

        drug_cf = tf.zeros_like(drug_targets, dtype=tf.float32)
        if drug_fp is None:
            return (ctl, drug_cf, cell_idx)
        if drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
            fp_cf = tf.zeros((tf.shape(ctl)[0], self.fp_dim), dtype=tf.float32)
        else:
            fp_cf = tf.zeros_like(drug_fp, dtype=tf.float32)
        return (ctl, drug_cf, cell_idx, fp_cf)

    def _make_weak_inputs(self, inputs):
        if self.wmss_weak == "cf":
            return self._make_counterfactual_inputs(inputs)
        return self._make_drug_zero_inputs(inputs)

    def _apply_wmss(self, y_true, y_pred_full, y_pred_cf, y_weak):
        alpha = tf.cast(self.wmss_alpha, tf.float32)
        if self.wmss_mode == "mix":
            y_t = (1.0 - alpha) * tf.cast(y_true, tf.float32) + alpha * tf.cast(y_weak, tf.float32)
            return y_t, tf.cast(y_pred_full, tf.float32), tf.cast(y_pred_cf, tf.float32)
        y_w = tf.cast(y_weak, tf.float32)
        y_t = tf.cast(y_true, tf.float32) - y_w
        return y_t, tf.cast(y_pred_full, tf.float32) - y_w, tf.cast(y_pred_cf, tf.float32) - y_w

    def _aux_target_loss(self, drug_targets, embeddings, sample_mask):
        logits = tf.squeeze(self.aux_target_head(embeddings), axis=-1)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(drug_targets, tf.float32), logits=tf.cast(logits, tf.float32))
        gene_mask = tf.cast(self.loss_mask[0], tf.float32)[None, :]
        bce = bce * gene_mask
        per_sample = tf.reduce_mean(bce, axis=1)
        per_sample = per_sample * tf.cast(sample_mask, tf.float32)
        denom = tf.reduce_sum(tf.cast(sample_mask, tf.float32)) + 1e-6
        return tf.reduce_sum(per_sample) / denom

    def train_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, aux_mask = data
        else:
            x, y = data
            aux_mask = None
        with tf.GradientTape() as tape:
            if len(x) == 3:
                ctl, drug_targets, cell_idx = x
                drug_fp = None
            else:
                ctl, drug_targets, cell_idx, drug_fp = x

            y_full, emb_full = self.core(x, training=True, return_embeddings=True)
            x_cf = self._make_counterfactual_inputs(x)
            y_cf = self.core(x_cf, training=True)

            if self.wmss_alpha > 0.0:
                x_w = self._make_weak_inputs(x)
                y_w = tf.stop_gradient(self.core(x_w, training=False))
                y_t, y_full_eff, y_cf_eff = self._apply_wmss(y, y_full, y_cf, y_w)
            else:
                y_t, y_full_eff, y_cf_eff = tf.cast(y, tf.float32), tf.cast(y_full, tf.float32), tf.cast(y_cf, tf.float32)

            loss_full = self._masked_combined_loss(y_t, y_full_eff)

            if aux_mask is None or self.aux_target_alpha <= 0.0:
                loss_aux = tf.constant(0.0, dtype=tf.float32)
            else:
                loss_aux = self._aux_target_loss(drug_targets, emb_full, aux_mask)
            loss_full = loss_full + tf.cast(self.aux_target_alpha, tf.float32) * loss_aux

            loss_cf = self._masked_combined_loss(y_t, y_cf_eff)

            gap = loss_cf - loss_full
            hinge = tf.nn.relu(self.cf_margin - gap)
            loss = loss_full + tf.cast(self.cf_lambda, tf.float32) * hinge

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.metric_total.update_state(loss)
        self.metric_full.update_state(loss_full)
        self.metric_cf.update_state(loss_cf)
        self.metric_hinge.update_state(hinge)
        self.metric_aux.update_state(loss_aux)
        self.metric_gap.update_state(gap)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, aux_mask = data
        else:
            x, y = data
            aux_mask = None
        if len(x) == 3:
            ctl, drug_targets, cell_idx = x
        else:
            ctl, drug_targets, cell_idx, _ = x

        y_full, emb_full = self.core(x, training=False, return_embeddings=True)
        x_cf = self._make_counterfactual_inputs(x)
        y_cf = self.core(x_cf, training=False)

        if self.wmss_alpha > 0.0:
            x_w = self._make_weak_inputs(x)
            y_w = tf.stop_gradient(self.core(x_w, training=False))
            y_t, y_full_eff, y_cf_eff = self._apply_wmss(y, y_full, y_cf, y_w)
        else:
            y_t, y_full_eff, y_cf_eff = tf.cast(y, tf.float32), tf.cast(y_full, tf.float32), tf.cast(y_cf, tf.float32)

        loss_full = self._masked_combined_loss(y_t, y_full_eff)
        if aux_mask is None or self.aux_target_alpha <= 0.0:
            loss_aux = tf.constant(0.0, dtype=tf.float32)
        else:
            loss_aux = self._aux_target_loss(drug_targets, emb_full, aux_mask)
        loss_full = loss_full + tf.cast(self.aux_target_alpha, tf.float32) * loss_aux
        loss_cf = self._masked_combined_loss(y_t, y_cf_eff)
        gap = loss_cf - loss_full
        hinge = tf.nn.relu(self.cf_margin - gap)
        loss = loss_full + tf.cast(self.cf_lambda, tf.float32) * hinge
        self.metric_total.update_state(loss)
        self.metric_full.update_state(loss_full)
        self.metric_cf.update_state(loss_cf)
        self.metric_hinge.update_state(hinge)
        self.metric_aux.update_state(loss_aux)
        self.metric_gap.update_state(gap)
        return {m.name: m.result() for m in self.metrics}


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
            target_scale = None
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
            target_scale = None
            try:
                inner = getattr(getattr(self.model, "core", None), "gat", None)
                if inner is not None and hasattr(inner, "target_scale_logit"):
                    target_scale = float(tf.nn.softplus(inner.target_scale_logit).numpy())
            except Exception:
                target_scale = None

        logs["pcc"] = tr["pcc"]
        logs["val_pcc"] = va["pcc"]
        extra = []
        for k in ["loss_full", "loss_cf", "cf_gap", "loss_hinge", "loss_aux"]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="MCF7")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_landmark_genes", action="store_true", default=False)
    parser.add_argument("--use_drug_fp_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_drug_embedding", dest="use_drug_fp_embedding", action="store_true")
    parser.add_argument("--no-use_drug_embedding", dest="use_drug_fp_embedding", action="store_false")
    parser.add_argument("--sparse_gat", action="store_true", default=True)
    parser.add_argument("--ctl_pair_k", type=int, default=3)
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
    parser.add_argument("--cf_mode", choices=["shuffle", "zero"], default="shuffle")
    parser.add_argument("--cf_lambda", type=float, default=1.0)
    parser.add_argument("--cf_margin", type=float, default=0.1)
    parser.add_argument("--aux_target_alpha", type=float, default=0.0)
    parser.add_argument("--wmss_alpha", type=float, default=0.0)
    parser.add_argument("--wmss_mode", choices=["residual", "mix"], default="residual")
    parser.add_argument("--wmss_weak", choices=["drug_zero", "cf"], default="drug_zero")
    parser.add_argument("--eval_drug_zero", action="store_true", default=False)
    parser.add_argument("--eval_drug_shuffle", action="store_true", default=False)
    parser.add_argument("--eval_sanity_max_eval", type=int, default=20000)
    parser.add_argument("--eval_sanity_seed", type=int, default=0)
    parser.add_argument("--save_gat_weights", default="")
    parser.add_argument("--save_meta_json", default="")
    parser.add_argument("--save_test_ids_npy", default="")
    parser.add_argument("--save_eval_npz", default="")
    parser.add_argument("--export_attention", action="store_true", default=False)
    parser.add_argument("--attention_max_samples", type=int, default=2000)
    parser.add_argument("--attention_batch_size", type=int, default=64)
    parser.add_argument("--attention_group_by", choices=["", "drug", "cell"], default="")
    parser.add_argument("--attention_groups", default="")
    parser.add_argument("--attention_top_k_groups", type=int, default=12)
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

    class GATWrapperCF(keras.Model):
        def __init__(self, gat_model, adj_matrix, use_drug_fp_embedding=False, fp_table=None):
            super().__init__()
            self.gat = gat_model
            self.adj = tf.constant(adj_matrix, dtype=tf.float32)
            self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
            self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

        def call(self, inputs, return_embeddings=False, training=False):
            if self.use_drug_fp_embedding:
                ctl, drug_targets, cell_idx, drug_fp = inputs
                cell_idx = tf.cast(cell_idx, tf.int32)
                if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                    drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
                return self.gat([self.adj, ctl, drug_targets, cell_idx, drug_fp], return_embeddings=return_embeddings)
            ctl, drug_targets, cell_idx = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            return self.gat([self.adj, ctl, drug_targets, cell_idx], return_embeddings=return_embeddings)

    class GATWrapperSparseCF(keras.Model):
        def __init__(self, gat_model, edge_index, edge_weight, use_drug_fp_embedding=False, fp_table=None):
            super().__init__()
            self.gat = gat_model
            self.edge_index = tf.constant(edge_index, dtype=tf.int32)
            self.edge_weight = tf.constant(edge_weight, dtype=tf.float32)
            self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
            self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)

        def call(self, inputs, return_embeddings=False, training=False):
            if self.use_drug_fp_embedding:
                ctl, drug_targets, cell_idx, drug_fp = inputs
                cell_idx = tf.cast(cell_idx, tf.int32)
                if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
                    drug_fp = tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
                return self.gat([self.edge_index, self.edge_weight, ctl, drug_targets, cell_idx, drug_fp], return_embeddings=return_embeddings)
            ctl, drug_targets, cell_idx = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            return self.gat([self.edge_index, self.edge_weight, ctl, drug_targets, cell_idx], return_embeddings=return_embeddings)

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
        ctl_residual_pool_size=int(args.ctl_pair_k),
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
    drug_has_target = data.get("drug_has_target")
    drug_has_target = None if drug_has_target is None else np.asarray(drug_has_target, dtype=np.float32)

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
        if drug_has_target is not None:
            drug_has_target = drug_has_target[idx]

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
        raise RuntimeError("use_drug_fp_embedding=True 但指纹不存在（X_fingerprint 与 drug_fp_table 均为空）")

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
        core = GATWrapperSparseCF(model, edge_index_full, edge_weight_full, use_drug_fp_embedding=bool(args.use_drug_fp_embedding), fp_table=fp_table)
        adj_matrix = None
    else:
        core = GATWrapperCF(model, adj_matrix, use_drug_fp_embedding=bool(args.use_drug_fp_embedding), fp_table=fp_table)

    trainer = CounterfactualTrainer(
        core_model=core,
        loss_mask=data["loss_mask"],
        cf_mode=str(args.cf_mode),
        cf_lambda=float(args.cf_lambda),
        cf_margin=float(args.cf_margin),
        fp_dim=int(fp_dim),
        aux_target_alpha=float(args.aux_target_alpha),
        wmss_alpha=float(args.wmss_alpha),
        wmss_mode=str(args.wmss_mode),
        wmss_weak=str(args.wmss_weak),
    )
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), run_eagerly=bool(args.run_eagerly))

    if args.use_drug_fp_embedding:
        pcc_cb = PCCCallback(
            loss_mask=data["loss_mask"],
            train_data=(train_ctl, train_drug, train_cells, train_fp, train_trt),
            val_data=(test_ctl, test_drug, test_cells, test_fp, test_trt),
            batch_size=int(args.batch_size),
            max_eval=int(args.eval_sanity_max_eval),
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )
        train_x = [train_ctl, train_drug, train_cells, train_fp]
        test_x = [test_ctl, test_drug, test_cells, test_fp]
    else:
        pcc_cb = PCCCallback(
            loss_mask=data["loss_mask"],
            train_data=(train_ctl, train_drug, train_cells, train_trt),
            val_data=(test_ctl, test_drug, test_cells, test_trt),
            batch_size=int(args.batch_size),
            max_eval=int(args.eval_sanity_max_eval),
            cell_mean=cell_delta_mean,
            y_is_residual=residualize_target,
        )
        train_x = [train_ctl, train_drug, train_cells]
        test_x = [test_ctl, test_drug, test_cells]

    if drug_has_target is None:
        train_has_t = np.ones((len(train_ctl),), dtype=np.float32)
        test_has_t = np.ones((len(test_ctl),), dtype=np.float32)
    else:
        train_has_t = drug_has_target[train_mask].astype(np.float32)
        test_has_t = drug_has_target[test_mask].astype(np.float32)

    ds_train = tf.data.Dataset.from_tensor_slices((tuple(train_x), train_trt, train_has_t)).batch(int(args.batch_size))
    ds_val = tf.data.Dataset.from_tensor_slices((tuple(test_x), test_trt, test_has_t)).batch(int(args.batch_size))

    trainer.fit(ds_train, epochs=int(args.epochs), callbacks=[pcc_cb], validation_data=ds_val, verbose=0)

    train_metrics = eval_pcc_mse(
        trainer,
        train_ctl,
        train_drug,
        train_cells,
        train_trt,
        data["loss_mask"],
        batch_size=int(args.batch_size),
        max_eval=int(args.eval_sanity_max_eval),
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
        max_eval=int(args.eval_sanity_max_eval),
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
        perm = rng.permutation(len(test_ctl))
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
        _ = model.save_weights(out_path)
        print(f"Saved GAT weights to: {out_path}")

    trt_distil_ids = np.asarray(data.get("trt_distil_ids", [""] * len(drug_ids)), dtype=object)
    if trt_distil_ids.size == 0 or len(trt_distil_ids) != len(drug_ids):
        trt_distil_ids = np.asarray([str(i) for i in range(len(drug_ids))], dtype=object)

    test_ids_path = str(args.save_test_ids_npy).strip()
    if test_ids_path == "" and str(args.save_meta_json).strip() != "":
        test_ids_path = os.path.splitext(str(args.save_meta_json).strip())[0] + ".test_ids.npy"
    if test_ids_path != "":
        out_dir = os.path.dirname(test_ids_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        np.save(test_ids_path, np.asarray(trt_distil_ids[test_mask], dtype=object))
        print(f"Saved test ids to: {test_ids_path}")

    if str(args.save_meta_json).strip() != "":
        out_path = str(args.save_meta_json).strip()
        out_dir = os.path.dirname(out_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        meta = {
            "cell_line": args.cell_line,
            "use_landmark_genes": bool(args.use_landmark_genes),
            "split_mode": str(args.split_mode),
            "test_frac": float(args.test_frac),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "sparse_gat": bool(args.sparse_gat),
            "use_drug_fp_embedding": bool(args.use_drug_fp_embedding),
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
            "aux_target_alpha": float(args.aux_target_alpha),
            "wmss_alpha": float(args.wmss_alpha),
            "wmss_mode": str(args.wmss_mode),
            "wmss_weak": str(args.wmss_weak),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_ids_npy": test_ids_path,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved run meta to: {out_path}")

    save_eval_npz = str(args.save_eval_npz).strip()
    if save_eval_npz != "":
        if bool(args.export_attention) and not bool(args.sparse_gat):
            raise ValueError("export_attention 目前只支持 --sparse_gat")

        if bool(args.use_drug_fp_embedding):
            y_pred_test = trainer.predict([test_ctl, test_drug, test_cells, test_fp], batch_size=256, verbose=0)
        else:
            y_pred_test = trainer.predict([test_ctl, test_drug, test_cells], batch_size=256, verbose=0)
        y_true_test = np.asarray(test_trt, dtype=np.float32)
        y_pred_test = np.asarray(y_pred_test, dtype=np.float32)
        if residualize_target and cell_delta_mean is not None:
            cm = np.asarray(cell_delta_mean, dtype=np.float32)[test_cells]
            y_true_test = y_true_test + cm
            y_pred_test = y_pred_test + cm

        sample_pcc, sample_mse = _samplewise_metrics(y_true_test, y_pred_test, data["loss_mask"])

        sanity = {}
        if bool(args.eval_drug_zero):
            zero_drug = np.zeros_like(test_drug, dtype=np.float32)
            if bool(args.use_drug_fp_embedding) and fp_dim > 0:
                zero_fp = np.zeros((len(test_ctl), int(fp_dim)), dtype=np.float32)
            else:
                zero_fp = None
            if zero_fp is None:
                y0 = trainer.predict([test_ctl, zero_drug, test_cells], batch_size=256, verbose=0)
            else:
                y0 = trainer.predict([test_ctl, zero_drug, test_cells, zero_fp], batch_size=256, verbose=0)
            y0 = np.asarray(y0, dtype=np.float32)
            if residualize_target and cell_delta_mean is not None:
                y0 = y0 + np.asarray(cell_delta_mean, dtype=np.float32)[test_cells]
            sanity["drug_zero"] = eval_pcc_mse(
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
        if bool(args.eval_drug_shuffle):
            rng = np.random.default_rng(int(args.eval_sanity_seed))
            perm = rng.permutation(len(test_ctl))
            shuf_drug = test_drug[perm]
            if bool(args.use_drug_fp_embedding):
                shuf_fp = None if test_fp is None else test_fp[perm]
            else:
                shuf_fp = None
            sanity["drug_shuffle"] = eval_pcc_mse(
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

        attention = {}
        if bool(args.export_attention):
            rng = np.random.default_rng(0)
            n_all = int(len(test_ctl))
            n_attn = min(int(args.attention_max_samples), n_all) if int(args.attention_max_samples) > 0 else n_all
            sel = rng.choice(n_all, size=n_attn, replace=False) if n_attn < n_all else np.arange(n_all)

            group_by = str(args.attention_group_by).strip().lower()
            group_labels = None
            if group_by == "drug":
                group_labels = np.asarray(drug_ids[test_mask][sel], dtype=str)
            elif group_by == "cell":
                group_labels = np.asarray(cell_names_arr[test_mask][sel], dtype=str)
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
                    b_fp = tf.constant(test_fp[idx], dtype=tf.float32) if (isinstance(test_fp, np.ndarray) and test_fp.ndim == 2) else tf.constant(test_fp[idx], dtype=tf.int32)
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
            "split_mode": str(args.split_mode),
            "test_frac": float(args.test_frac),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "sparse_gat": bool(args.sparse_gat),
            "use_drug_fp_embedding": bool(args.use_drug_fp_embedding),
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
            "aux_target_alpha": float(args.aux_target_alpha),
            "wmss_alpha": float(args.wmss_alpha),
            "wmss_mode": str(args.wmss_mode),
            "wmss_weak": str(args.wmss_weak),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_ids_npy": test_ids_path,
            "export_attention": bool(args.export_attention),
            "attention_group_by": str(args.attention_group_by),
        }

        _save_npz(
            save_eval_npz,
            y_true=y_true_test.astype(np.float32),
            y_pred=y_pred_test.astype(np.float32),
            sample_pcc=sample_pcc.astype(np.float32),
            sample_mse=sample_mse.astype(np.float32),
            cell_names=np.asarray(cell_names_arr[test_mask], dtype=object),
            drug_ids=np.asarray(drug_ids[test_mask], dtype=object),
            batch_ids=np.asarray(batch_ids_arr[test_mask], dtype=object),
            trt_distil_ids=np.asarray(trt_distil_ids[test_mask], dtype=object),
            target_genes=np.asarray(data["target_genes"], dtype=object),
            metrics=test_metrics,
            sanity=sanity,
            attention=attention,
            meta=meta_out,
        )
        print(f"Saved eval npz to: {save_eval_npz}")


if __name__ == "__main__":
    main()
