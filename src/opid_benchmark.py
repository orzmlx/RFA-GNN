import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def _norm_symbol(s):
    return str(s).strip().upper()


def _norm_entrez(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


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


def samplewise_pcc_only(y_true, y_pred, loss_mask):
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


def gene_standardize_fit(y_train, eps=1e-6):
    mu = np.mean(y_train, axis=0, dtype=np.float32)
    sd = np.std(y_train, axis=0, dtype=np.float32)
    sd = np.maximum(sd, float(eps)).astype(np.float32)
    return mu.astype(np.float32), sd


def gene_standardize_apply(y, mu, sd):
    return (y - mu[None, :]) / sd[None, :]


def cold_drug_split(drug_ids, test_frac=0.2, seed=42):
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(drug_ids)
    if len(uniq) < 2:
        raise ValueError("cold_drug 需要至少 2 个药物")
    n_test = max(1, int(len(uniq) * float(test_frac)))
    n_test = min(n_test, len(uniq) - 1)
    held = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(drug_ids, held)
    return ~test_mask, test_mask


def load_signed_adjs_landmark(
    tf_path,
    ppi_path,
    mirna_path,
    symbol_to_entrez,
    target_genes,
    consensus_only=True,
    include_ppi_undirected=True,
):
    target_genes = [_norm_entrez(x) for x in target_genes]
    target_genes = [x for x in target_genes if x]
    target_genes = list(dict.fromkeys(target_genes))
    gene2idx = {g: i for i, g in enumerate(target_genes)}
    target_set = set(target_genes)
    n = len(target_genes)

    def _sign(stim, inhib):
        return -1.0 if (bool(inhib) and not bool(stim)) else 1.0

    rows_tf_pos, cols_tf_pos, data_tf_pos = [], [], []
    rows_tf_neg, cols_tf_neg, data_tf_neg = [], [], []
    rows_ppi_pos, cols_ppi_pos, data_ppi_pos = [], [], []
    rows_ppi_neg, cols_ppi_neg, data_ppi_neg = [], [], []
    rows_undir, cols_undir, data_undir = [], [], []
    rows_mi_neg, cols_mi_neg, data_mi_neg = [], [], []

    df_tf = pd.read_csv(tf_path, dtype=str, low_memory=False)
    for c in ["consensus_direction", "consensus_stimulation", "consensus_inhibition"]:
        if c not in df_tf.columns:
            df_tf[c] = False
    df_tf["consensus_direction"] = df_tf["consensus_direction"].fillna(False).astype(bool)
    df_tf["consensus_stimulation"] = df_tf["consensus_stimulation"].fillna(False).astype(bool)
    df_tf["consensus_inhibition"] = df_tf["consensus_inhibition"].fillna(False).astype(bool)
    for _, r in df_tf.iterrows():
        if bool(consensus_only) and not bool(r["consensus_direction"]):
            continue
        s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
        t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
        if not s or not t:
            continue
        s = _norm_entrez(s)
        t = _norm_entrez(t)
        if s not in target_set or t not in target_set or s == t:
            continue
        si = gene2idx[s]
        ti = gene2idx[t]
        sign = _sign(r["consensus_stimulation"], r["consensus_inhibition"])
        if sign >= 0:
            rows_tf_pos.append(si)
            cols_tf_pos.append(ti)
            data_tf_pos.append(1.0)
        else:
            rows_tf_neg.append(si)
            cols_tf_neg.append(ti)
            data_tf_neg.append(1.0)

    df_ppi = pd.read_csv(ppi_path, dtype=str, low_memory=False)
    for c in ["consensus_direction", "is_directed", "consensus_stimulation", "consensus_inhibition"]:
        if c not in df_ppi.columns:
            df_ppi[c] = False
    df_ppi["consensus_direction"] = df_ppi["consensus_direction"].fillna(False).astype(bool)
    df_ppi["is_directed"] = df_ppi["is_directed"].fillna(False).astype(bool)
    df_ppi["consensus_stimulation"] = df_ppi["consensus_stimulation"].fillna(False).astype(bool)
    df_ppi["consensus_inhibition"] = df_ppi["consensus_inhibition"].fillna(False).astype(bool)
    for _, r in df_ppi.iterrows():
        if bool(consensus_only) and not bool(r["consensus_direction"]):
            continue
        s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
        t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
        if not s or not t:
            continue
        s = _norm_entrez(s)
        t = _norm_entrez(t)
        if s not in target_set or t not in target_set or s == t:
            continue
        si = gene2idx[s]
        ti = gene2idx[t]
        directed = bool(r["consensus_direction"]) or bool(r["is_directed"])
        sign = _sign(r["consensus_stimulation"], r["consensus_inhibition"])
        if directed:
            if sign >= 0:
                rows_ppi_pos.append(si)
                cols_ppi_pos.append(ti)
                data_ppi_pos.append(1.0)
            else:
                rows_ppi_neg.append(si)
                cols_ppi_neg.append(ti)
                data_ppi_neg.append(1.0)
        elif bool(include_ppi_undirected) and not bool(consensus_only):
            rows_undir.extend([si, ti])
            cols_undir.extend([ti, si])
            data_undir.extend([1.0, 1.0])

    if mirna_path and os.path.exists(mirna_path):
        df_mi = pd.read_csv(mirna_path, dtype=str, low_memory=False)
        for c in ["source_genesymbol", "target_genesymbol"]:
            if c not in df_mi.columns:
                df_mi[c] = ""
        if "consensus_direction" not in df_mi.columns:
            df_mi["consensus_direction"] = False
        df_mi["consensus_direction"] = df_mi["consensus_direction"].fillna(False).astype(bool)
        for _, r in df_mi.iterrows():
            if bool(consensus_only) and not bool(r["consensus_direction"]):
                continue
            s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
            t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
            if not s or not t:
                continue
            s = _norm_entrez(s)
            t = _norm_entrez(t)
            if s not in target_set or t not in target_set or s == t:
                continue
            si = gene2idx[s]
            ti = gene2idx[t]
            rows_mi_neg.append(si)
            cols_mi_neg.append(ti)
            data_mi_neg.append(1.0)

    def _row_normalize(rows, cols, data):
        if not rows:
            return None
        A = tf.sparse.SparseTensor(indices=np.asarray(list(zip(rows, cols)), dtype=np.int64), values=np.asarray(data, dtype=np.float32), dense_shape=(n, n))
        A = tf.sparse.reorder(A)
        deg = tf.sparse.reduce_sum(A, axis=1)
        deg = tf.maximum(deg, 1.0)
        inv = 1.0 / deg
        v = A.values * tf.gather(inv, tf.cast(A.indices[:, 0], tf.int32))
        A = tf.sparse.SparseTensor(indices=A.indices, values=v, dense_shape=A.dense_shape)
        A = tf.sparse.reorder(A)
        return A

    return {
        "TF_POS": _row_normalize(rows_tf_pos, cols_tf_pos, data_tf_pos),
        "TF_NEG": _row_normalize(rows_tf_neg, cols_tf_neg, data_tf_neg),
        "PPI_POS": _row_normalize(rows_ppi_pos, cols_ppi_pos, data_ppi_pos),
        "PPI_NEG": _row_normalize(rows_ppi_neg, cols_ppi_neg, data_ppi_neg),
        "UNDIR": _row_normalize(rows_undir, cols_undir, data_undir),
        "MIRNA_NEG": _row_normalize(rows_mi_neg, cols_mi_neg, data_mi_neg),
    }


def _spmm_right(batch_h, A):
    if A is None:
        return tf.zeros_like(batch_h)
    h_t = tf.transpose(batch_h, perm=[1, 0])
    out_t = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(A), h_t)
    return tf.transpose(out_t, perm=[1, 0])


class OPID(tf.keras.Model):
    def __init__(
        self,
        adjs,
        num_steps=6,
        hidden_dim=64,
        cons_weight=0.2,
        target_dropout=0.4,
        use_cell_embedding=True,
        num_cells=1,
        use_ctl=True,
    ):
        super().__init__()
        self.adjs = dict(adjs)
        self.num_steps = int(num_steps)
        self.hidden_dim = int(hidden_dim)
        self.cons_weight = float(cons_weight)
        self.target_dropout = float(target_dropout)
        self.use_cell_embedding = bool(use_cell_embedding)
        self.use_ctl = bool(use_ctl)

        self.g_tf_pos = self.add_weight(shape=(), initializer="zeros", trainable=True, name="g_tf_pos")
        self.g_tf_neg = self.add_weight(shape=(), initializer="zeros", trainable=True, name="g_tf_neg")
        self.g_ppi_pos = self.add_weight(shape=(), initializer="zeros", trainable=True, name="g_ppi_pos")
        self.g_ppi_neg = self.add_weight(shape=(), initializer="zeros", trainable=True, name="g_ppi_neg")
        self.g_undir = self.add_weight(shape=(), initializer="zeros", trainable=True, name="g_undir")
        self.g_mirna_neg = self.add_weight(shape=(), initializer="zeros", trainable=True, name="g_mirna_neg")
        self.alpha_logits = self.add_weight(shape=(self.num_steps,), initializer="zeros", trainable=True, name="alpha_logits")

        if self.use_cell_embedding:
            self.cell_emb = tf.keras.layers.Embedding(int(num_cells), int(hidden_dim))

        in_dim = 2 if self.use_ctl else 1
        self.decode_in = tf.keras.layers.Dense(int(hidden_dim), activation="relu")
        self.decode_out = tf.keras.layers.Dense(1)
        self.in_dim = int(in_dim)

    def _g(self, x):
        return tf.nn.softplus(x)

    def propagate(self, u0):
        h0 = u0
        h = u0
        g_tf_pos = self._g(self.g_tf_pos)
        g_tf_neg = self._g(self.g_tf_neg)
        g_ppi_pos = self._g(self.g_ppi_pos)
        g_ppi_neg = self._g(self.g_ppi_neg)
        g_undir = self._g(self.g_undir)
        g_mi_neg = self._g(self.g_mirna_neg)
        alphas = tf.nn.sigmoid(self.alpha_logits)

        for k in range(self.num_steps):
            msg = tf.zeros_like(h)
            msg = msg + g_tf_pos * _spmm_right(h, self.adjs.get("TF_POS"))
            msg = msg - g_tf_neg * _spmm_right(h, self.adjs.get("TF_NEG"))
            msg = msg + g_ppi_pos * _spmm_right(h, self.adjs.get("PPI_POS"))
            msg = msg - g_ppi_neg * _spmm_right(h, self.adjs.get("PPI_NEG"))
            msg = msg + g_undir * _spmm_right(h, self.adjs.get("UNDIR"))
            msg = msg - g_mi_neg * _spmm_right(h, self.adjs.get("MIRNA_NEG"))
            a = alphas[k]
            h = a * h0 + (1.0 - a) * msg
        return h

    def decode(self, ctl_base, u_raw, hK, cell_idx):
        feats = [u_raw, hK]
        if self.use_ctl:
            feats = [ctl_base] + feats
        x = tf.stack(feats, axis=-1)
        h = self.decode_in(x)
        if self.use_cell_embedding:
            c = self.cell_emb(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(h)[1], 1])
            h = h + c
        y = tf.squeeze(self.decode_out(h), axis=-1)
        return y

    def call(self, inputs, training=False):
        ctl_base, u_raw, cell_idx = inputs
        if not training or self.target_dropout <= 0.0:
            h = self.propagate(u_raw)
            return self.decode(ctl_base, u_raw, h, cell_idx)

        keep = 1.0 - float(self.target_dropout)
        mask1 = tf.cast(tf.random.uniform(tf.shape(u_raw)) < keep, tf.float32)
        mask2 = tf.cast(tf.random.uniform(tf.shape(u_raw)) < keep, tf.float32)
        u1 = u_raw * mask1
        u2 = u_raw * mask2
        h1 = self.propagate(u1)
        h2 = self.propagate(u2)
        y1 = self.decode(ctl_base, u1, h1, cell_idx)
        y2 = self.decode(ctl_base, u2, h2, cell_idx)
        return y1, y2

    def train_step(self, data):
        (ctl_base, u_raw, cell_idx), y_true = data
        loss_mask = self.loss_mask
        with tf.GradientTape() as tape:
            y1, y2 = self((ctl_base, u_raw, cell_idx), training=True)
            w = tf.cast(loss_mask, tf.float32)
            mse1 = tf.reduce_mean(tf.square((y_true - y1) * w))
            mse2 = tf.reduce_mean(tf.square((y_true - y2) * w))
            cons = tf.reduce_mean(tf.square((y1 - y2) * w))
            loss = 0.5 * (mse1 + mse2) + float(self.cons_weight) * cons
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "mse": 0.5 * (mse1 + mse2), "cons": cons}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    p.add_argument("--cell_line", default="ALL")
    p.add_argument("--max_samples", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--cons_weight", type=float, default=0.2)
    p.add_argument("--target_dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--no_ctl", action="store_true", default=False)
    p.add_argument("--std_resid", action="store_true", default=False)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--consensus_only", action="store_true", default=True)
    p.add_argument("--include_mirna", action="store_true", default=False)
    args = p.parse_args()

    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from data_loader import load_rfa_data

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
        use_landmark_genes=True,
        full_gene_path=full_gene_path,
        cell_lines=cell_lines,
        ctl_residual_pool_size=0,
    )

    X_ctl = np.asarray(data["X_ctl"], dtype=np.float32)
    y = np.asarray(data["y_delta"], dtype=np.float32)
    X_drug = np.asarray(data["X_drug"], dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    if int(args.max_samples) > 0 and len(y) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(y), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y = y[idx]
        X_drug = X_drug[idx]
        drug_ids = drug_ids[idx]
        cell_names = cell_names[idx]

    if X_ctl.ndim == 3 and X_ctl.shape[-1] == 2:
        ctl_base = X_ctl[..., 0]
    elif X_ctl.ndim == 2:
        ctl_base = X_ctl
    else:
        ctl_base = np.squeeze(X_ctl, axis=-1)

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names).astype(np.int32)
    num_cells = int(len(le.classes_))

    train_mask, test_mask = cold_drug_split(drug_ids, test_frac=float(args.test_frac), seed=int(args.seed))
    tr = np.where(train_mask)[0]
    te = np.where(test_mask)[0]
    print(f"Train samples: {len(tr)} | Test samples: {len(te)}")

    sums = np.zeros((num_cells, y.shape[1]), dtype=np.float32)
    cnts = np.zeros((num_cells,), dtype=np.int64)
    np.add.at(sums, cell_idx[tr], y[tr])
    np.add.at(cnts, cell_idx[tr], 1)
    cell_mean = sums / np.maximum(cnts[:, None], 1)

    y_tr_raw = y[tr] - cell_mean[cell_idx[tr]]
    y_te = y[te]
    y_te_raw = y[te] - cell_mean[cell_idx[te]]

    if bool(args.std_resid):
        mu, sd = gene_standardize_fit(y_tr_raw)
        y_tr = gene_standardize_apply(y_tr_raw, mu, sd)
        print("Train residual: per-gene standardized")
    else:
        mu, sd = None, None
        y_tr = y_tr_raw

    adjs = load_signed_adjs_landmark(
        tf_path=tf_path,
        ppi_path=ppi_path,
        mirna_path=(mirna_path if bool(args.include_mirna) else ""),
        symbol_to_entrez=data.get("symbol_to_entrez") or {},
        target_genes=data["target_genes"],
        consensus_only=bool(args.consensus_only),
        include_ppi_undirected=True,
    )

    model = OPID(
        adjs=adjs,
        num_steps=int(args.steps),
        hidden_dim=int(args.hidden_dim),
        cons_weight=float(args.cons_weight),
        target_dropout=float(args.target_dropout),
        use_cell_embedding=True,
        num_cells=num_cells,
        use_ctl=(not bool(args.no_ctl)),
    )
    model.loss_mask = tf.constant(loss_mask, dtype=tf.float32)
    model.compile(optimizer=tf.keras.optimizers.Adam(float(args.lr)))

    x_train = (ctl_base[tr], X_drug[tr], cell_idx[tr])
    x_test = (ctl_base[te], X_drug[te], cell_idx[te])

    baseline_pred = cell_mean[cell_idx[te]]
    baseline_pcc = samplewise_pcc_only(y_te, baseline_pred, loss_mask)
    print(f"Cell-mean baseline PCC: {baseline_pcc:.4f}")

    best = -1e9
    for ep in range(int(args.epochs)):
        model.fit(x_train, y_tr, batch_size=int(args.batch_size), epochs=1, verbose=0)
        pred_resid = model.predict(x_test, batch_size=int(args.batch_size), verbose=0)
        pred_resid_mean_abs = float(np.mean(np.abs(pred_resid)))

        if mu is not None and sd is not None:
            pred_resid_unstd = pred_resid * sd[None, :] + mu[None, :]
        else:
            pred_resid_unstd = pred_resid

        resid_pcc = samplewise_pcc_only(y_te_raw, pred_resid_unstd, loss_mask)

        pred = pred_resid_unstd + cell_mean[cell_idx[te]]
        m = samplewise_pcc_mse(y_te, pred, loss_mask)
        best = max(best, m["pcc"])
        print(f"OPID | epoch {ep+1} | val_pcc={m['pcc']:.4f} | val_mse={m['mse']:.4f}")
        print(f"resid_pcc={resid_pcc:.4f} | mean(|pred_resid|)={pred_resid_mean_abs:.6f}")
        g = [float(tf.nn.softplus(v).numpy()) for v in [model.g_tf_pos, model.g_tf_neg, model.g_ppi_pos, model.g_ppi_neg, model.g_undir, model.g_mirna_neg]]
        a = tf.nn.sigmoid(model.alpha_logits).numpy().tolist()
        print(f"gains tf+ tf- ppi+ ppi- undir mi-: {[round(x,6) for x in g]}")
        print(f"alphas: {[round(x,4) for x in a]}")
    print(f"OPID best_val_pcc={best:.4f}")


if __name__ == "__main__":
    main()
