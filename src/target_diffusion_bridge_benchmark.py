import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers


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


def build_symbol_to_entrez(full_gene_path):
    df = pd.read_csv(full_gene_path, sep="\t", dtype=str, low_memory=False)
    m = {}
    for sym, eid in zip(df["pr_gene_symbol"].astype(str), df["pr_gene_id"].astype(str)):
        s = _norm_symbol(sym)
        e = _norm_entrez(eid)
        if s and e:
            m[s] = e
    return m


def load_full_gene_list(full_gene_path):
    df = pd.read_csv(full_gene_path, sep="\t", dtype=str, low_memory=False)
    gene_ids = [_norm_entrez(x) for x in df["pr_gene_id"].astype(str).tolist()]
    gene_ids = [g for g in gene_ids if g]
    gene_ids = list(dict.fromkeys(gene_ids))
    return gene_ids


def parse_compound_targets(compound_targets_path, symbol_to_entrez):
    df = pd.read_csv(compound_targets_path, sep="\t", dtype=str, low_memory=False)
    if "pert_id" not in df.columns or "target" not in df.columns:
        raise ValueError("compound_targets.txt 缺少 pert_id/target 列")
    pid2targets = {}
    for pid, tgt in zip(df["pert_id"].astype(str), df["target"].astype(str)):
        pid = str(pid)
        if pid == "" or pid.lower() == "nan":
            continue
        if tgt is None or str(tgt).lower() == "nan":
            pid2targets.setdefault(pid, [])
            continue
        syms = [x.strip() for x in str(tgt).split(",") if x.strip()]
        eids = []
        for s in syms:
            eid = symbol_to_entrez.get(_norm_symbol(s))
            if eid:
                eids.append(_norm_entrez(eid))
        pid2targets[pid] = list(dict.fromkeys([e for e in eids if e]))
    return pid2targets


def load_omnipath_sparse_transition(
    tf_path,
    ppi_path,
    gene2idx_full,
    symbol_to_entrez,
    consensus_only=True,
    include_undirected_ppi=True,
):
    def _sign(stim, inhib):
        return -1.0 if (bool(inhib) and not bool(stim)) else 1.0

    rows_pos = []
    cols_pos = []
    data_pos = []
    rows_neg = []
    cols_neg = []
    data_neg = []

    tf_df = pd.read_csv(tf_path, dtype=str, low_memory=False)
    for c in ["consensus_direction", "consensus_stimulation", "consensus_inhibition"]:
        if c not in tf_df.columns:
            tf_df[c] = False
    tf_df["consensus_direction"] = tf_df["consensus_direction"].fillna(False).astype(bool)
    tf_df["consensus_stimulation"] = tf_df["consensus_stimulation"].fillna(False).astype(bool)
    tf_df["consensus_inhibition"] = tf_df["consensus_inhibition"].fillna(False).astype(bool)
    for _, r in tf_df.iterrows():
        if bool(consensus_only) and not bool(r["consensus_direction"]):
            continue
        s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
        t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
        if not s or not t:
            continue
        s = _norm_entrez(s)
        t = _norm_entrez(t)
        si = gene2idx_full.get(s)
        ti = gene2idx_full.get(t)
        if si is None or ti is None or si == ti:
            continue
        sign = _sign(r["consensus_stimulation"], r["consensus_inhibition"])
        if sign >= 0:
            rows_pos.append(si)
            cols_pos.append(ti)
            data_pos.append(1.0)
        else:
            rows_neg.append(si)
            cols_neg.append(ti)
            data_neg.append(1.0)

    ppi_df = pd.read_csv(ppi_path, dtype=str, low_memory=False)
    for c in ["consensus_direction", "is_directed", "consensus_stimulation", "consensus_inhibition"]:
        if c not in ppi_df.columns:
            ppi_df[c] = False
    ppi_df["consensus_direction"] = ppi_df["consensus_direction"].fillna(False).astype(bool)
    ppi_df["is_directed"] = ppi_df["is_directed"].fillna(False).astype(bool)
    ppi_df["consensus_stimulation"] = ppi_df["consensus_stimulation"].fillna(False).astype(bool)
    ppi_df["consensus_inhibition"] = ppi_df["consensus_inhibition"].fillna(False).astype(bool)

    for _, r in ppi_df.iterrows():
        if bool(consensus_only) and not bool(r["consensus_direction"]):
            continue
        s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
        t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
        if not s or not t:
            continue
        s = _norm_entrez(s)
        t = _norm_entrez(t)
        si = gene2idx_full.get(s)
        ti = gene2idx_full.get(t)
        if si is None or ti is None or si == ti:
            continue
        directed = bool(r["consensus_direction"]) or bool(r["is_directed"])
        sign = _sign(r["consensus_stimulation"], r["consensus_inhibition"])
        if directed:
            if sign >= 0:
                rows_pos.append(si)
                cols_pos.append(ti)
                data_pos.append(1.0)
            else:
                rows_neg.append(si)
                cols_neg.append(ti)
                data_neg.append(1.0)
        elif bool(include_undirected_ppi) and not bool(consensus_only):
            rows_pos.extend([si, ti])
            cols_pos.extend([ti, si])
            data_pos.extend([1.0, 1.0])

    n = len(gene2idx_full)
    P_pos = sp.csr_matrix((data_pos, (rows_pos, cols_pos)), shape=(n, n), dtype=np.float32)
    P_neg = sp.csr_matrix((data_neg, (rows_neg, cols_neg)), shape=(n, n), dtype=np.float32)

    def _row_normalize(M):
        deg = np.asarray(M.sum(axis=1)).reshape(-1)
        deg = np.maximum(deg, 1.0).astype(np.float32)
        inv = 1.0 / deg
        Dinv = sp.diags(inv)
        return Dinv @ M

    return _row_normalize(P_pos), _row_normalize(P_neg)


def diffuse_targets_batch(S0, P_pos, P_neg, steps=6, alpha=0.2):
    S = S0.astype(np.float32, copy=True)
    S_pos = S.copy()
    S_neg = S.copy()
    for _ in range(int(steps)):
        S_pos = (1.0 - float(alpha)) * (S_pos @ P_pos) + float(alpha) * S
        S_neg = (1.0 - float(alpha)) * (S_neg @ P_neg) + float(alpha) * S
    return (S_pos - S_neg).astype(np.float32)


class DiffusionMLP(keras.Model):
    def __init__(self, num_cells, hidden_dim=64, use_cell=True):
        super().__init__()
        self.use_cell = bool(use_cell)
        self.shared = layers.Dense(int(hidden_dim), activation="relu")
        if self.use_cell:
            self.cell_emb = layers.Embedding(int(num_cells), int(hidden_dim))
        self.out = layers.Dense(1)

    def call(self, inputs):
        ctl, tgt_raw, z_diff, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        x = tf.stack([ctl, tgt_raw, z_diff], axis=-1)
        h = self.shared(x)
        if self.use_cell:
            c = self.cell_emb(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(h)[1], 1])
            h = h + c
        y = tf.squeeze(self.out(h), axis=-1)
        return y


class TargetOnlyMLP(keras.Model):
    def __init__(self, num_cells, hidden_dim=64, use_cell=True):
        super().__init__()
        self.use_cell = bool(use_cell)
        self.shared = layers.Dense(int(hidden_dim), activation="relu")
        if self.use_cell:
            self.cell_emb = layers.Embedding(int(num_cells), int(hidden_dim))
        self.out = layers.Dense(1)

    def call(self, inputs):
        ctl, tgt_raw, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        x = tf.stack([ctl, tgt_raw], axis=-1)
        h = self.shared(x)
        if self.use_cell:
            c = self.cell_emb(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(h)[1], 1])
            h = h + c
        y = tf.squeeze(self.out(h), axis=-1)
        return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    p.add_argument("--cell_line", default="ALL")
    p.add_argument("--max_samples", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diff_steps", type=int, default=6)
    p.add_argument("--diff_alpha", type=float, default=0.2)
    p.add_argument("--target_dropout", type=float, default=0.3)
    p.add_argument("--consensus_only", action="store_true", default=True)
    args = p.parse_args()

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from data_loader import load_rfa_data

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
        use_landmark_genes=True,
        full_gene_path=full_gene_path,
        cell_lines=cell_lines,
        ctl_residual_pool_size=0,
    )
    X_ctl = np.asarray(data["X_ctl"], dtype=np.float32)
    y = np.asarray(data["y_delta"], dtype=np.float32)
    tgt_landmark = np.asarray(data["X_drug"], dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    if int(args.max_samples) > 0 and len(y) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(y), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y = y[idx]
        tgt_landmark = tgt_landmark[idx]
        drug_ids = drug_ids[idx]
        cell_names = cell_names[idx]

    if X_ctl.ndim == 3 and X_ctl.shape[-1] == 2:
        ctl_base = X_ctl[..., 0]
    elif X_ctl.ndim == 2:
        ctl_base = X_ctl
    else:
        ctl_base = np.squeeze(X_ctl, axis=-1)

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names)
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
    y_tr_resid = y[tr] - cell_mean[cell_idx[tr]]
    y_te = y[te]

    symbol_to_entrez = build_symbol_to_entrez(full_gene_path)
    full_genes = load_full_gene_list(full_gene_path)
    gene2idx_full = {g: i for i, g in enumerate(full_genes)}
    landmark_genes = [_norm_entrez(x) for x in data["target_genes"]]
    landmark_full_idx = np.asarray([gene2idx_full[g] for g in landmark_genes if g in gene2idx_full], dtype=np.int32)
    if len(landmark_full_idx) != len(landmark_genes):
        raise RuntimeError("Landmark genes mapping to full list failed")

    pid2targets = parse_compound_targets(drug_target_path, symbol_to_entrez)
    uniq_drugs = np.unique(drug_ids)
    drug2row = {d: i for i, d in enumerate(uniq_drugs)}
    S0 = np.zeros((len(uniq_drugs), len(full_genes)), dtype=np.float32)
    for d in uniq_drugs:
        tgts = pid2targets.get(str(d), [])
        idxs = [gene2idx_full.get(t) for t in tgts if gene2idx_full.get(t) is not None]
        if idxs:
            S0[drug2row[d], np.asarray(idxs, dtype=np.int32)] = 1.0

    P_pos, P_neg = load_omnipath_sparse_transition(
        tf_path=tf_path,
        ppi_path=ppi_path,
        gene2idx_full=gene2idx_full,
        symbol_to_entrez=symbol_to_entrez,
        consensus_only=bool(args.consensus_only),
        include_undirected_ppi=True,
    )

    rng = np.random.default_rng(int(args.seed))
    drop_p = float(args.target_dropout)
    if drop_p < 0.0 or drop_p >= 1.0:
        raise ValueError("--target_dropout should be in [0,1)")

    def compute_z_landmark(drug_id_batch):
        rows = np.asarray([drug2row[d] for d in drug_id_batch], dtype=np.int32)
        S = S0[rows].copy()
        if drop_p > 0.0:
            mask = rng.random(size=S.shape, dtype=np.float32) >= drop_p
            S *= mask.astype(np.float32)
        Z = diffuse_targets_batch(S, P_pos, P_neg, steps=int(args.diff_steps), alpha=float(args.diff_alpha))
        return Z[:, landmark_full_idx]

    x_train_ctl = ctl_base[tr]
    x_train_tgt = tgt_landmark[tr]
    x_train_cell = cell_idx[tr].astype(np.int32)
    y_train = y_tr_resid
    drug_train = drug_ids[tr]

    x_test_ctl = ctl_base[te]
    x_test_tgt = tgt_landmark[te]
    x_test_cell = cell_idx[te].astype(np.int32)
    y_test = y[te]
    drug_test = drug_ids[te]

    class DiffusionSequence(keras.utils.PyDataset):
        def __init__(self, ctl, tgt, cell, drug_id, y_true, batch_size):
            super().__init__()
            self.ctl = np.asarray(ctl, dtype=np.float32)
            self.tgt = np.asarray(tgt, dtype=np.float32)
            self.cell = np.asarray(cell, dtype=np.int32)
            self.drug_id = np.asarray(drug_id, dtype=str)
            self.y_true = np.asarray(y_true, dtype=np.float32)
            self.batch_size = int(batch_size)

        def __len__(self):
            return int(np.ceil(len(self.ctl) / self.batch_size))

        def __getitem__(self, idx):
            b0 = idx * self.batch_size
            b1 = min(len(self.ctl), (idx + 1) * self.batch_size)
            z = compute_z_landmark(self.drug_id[b0:b1])
            x = (self.ctl[b0:b1], self.tgt[b0:b1], z, self.cell[b0:b1])
            yb = self.y_true[b0:b1]
            return x, yb

    seq_train = DiffusionSequence(x_train_ctl, x_train_tgt, x_train_cell, drug_train, y_train, batch_size=int(args.batch_size))

    model_diff = DiffusionMLP(num_cells=num_cells, hidden_dim=int(args.hidden_dim), use_cell=True)
    model_diff.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    best = -1e9
    for ep in range(int(args.epochs)):
        model_diff.fit(seq_train, epochs=1, verbose=0)
        z_test = compute_z_landmark(drug_test)
        pred_resid = model_diff.predict((x_test_ctl, x_test_tgt, z_test, x_test_cell), batch_size=int(args.batch_size), verbose=0)
        pred = pred_resid + cell_mean[x_test_cell]
        m = samplewise_pcc_mse(y_test, pred, loss_mask)
        best = max(best, m["pcc"])
        print(f"Diffusion+Dropout | epoch {ep+1} | val_pcc={m['pcc']:.4f} | val_mse={m['mse']:.4f}")
    print(f"Diffusion+Dropout best_val_pcc={best:.4f}")

    model_tgt = TargetOnlyMLP(num_cells=num_cells, hidden_dim=int(args.hidden_dim), use_cell=True)
    model_tgt.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    best2 = -1e9
    for ep in range(int(args.epochs)):
        model_tgt.fit((x_train_ctl, x_train_tgt, x_train_cell), y_train, batch_size=int(args.batch_size), epochs=1, verbose=0)
        pred_resid = model_tgt.predict((x_test_ctl, x_test_tgt, x_test_cell), batch_size=int(args.batch_size), verbose=0)
        pred = pred_resid + cell_mean[x_test_cell]
        m = samplewise_pcc_mse(y_test, pred, loss_mask)
        best2 = max(best2, m["pcc"])
        print(f"TargetOnly | epoch {ep+1} | val_pcc={m['pcc']:.4f} | val_mse={m['mse']:.4f}")
    print(f"TargetOnly best_val_pcc={best2:.4f}")


if __name__ == "__main__":
    main()

