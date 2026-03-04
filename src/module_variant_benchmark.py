import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers


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


def samplewise_mse(y_true, y_pred, loss_mask):
    valid = np.where(np.asarray(loss_mask)[0] > 0)[0]
    yt = y_true[:, valid]
    yp = y_pred[:, valid]
    return float(mean_squared_error(yt, yp))


def build_signed_module_adj(A_pos, A_neg, A_undir):
    A = A_pos - A_neg + A_undir
    A = np.asarray(A, dtype=np.float32)
    row_norm = np.sum(np.abs(A), axis=1, keepdims=True)
    row_norm = np.maximum(row_norm, 1.0)
    return A / row_norm


class ModuleAPPNP(layers.Layer):
    def __init__(self, steps=10, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.steps = int(steps)
        self.alpha = float(alpha)

    def call(self, S, A):
        Z = tf.identity(S)
        for _ in range(self.steps):
            Z = (1.0 - self.alpha) * tf.matmul(Z, A) + self.alpha * S
        return Z


class SchemeA(keras.Model):
    def __init__(self, M_gk, A_kk, num_cells, hidden_dim=64, steps=10, alpha=0.1, use_cell=True):
        super().__init__()
        self.M = tf.constant(M_gk, dtype=tf.float32)
        self.Mt = tf.transpose(self.M)
        self.A = tf.constant(A_kk, dtype=tf.float32)
        self.appnp = ModuleAPPNP(steps=steps, alpha=alpha)
        self.use_cell = bool(use_cell)
        self.shared = layers.Dense(int(hidden_dim), activation="relu")
        if self.use_cell:
            self.cell_emb = layers.Embedding(int(num_cells), int(hidden_dim))
        self.out = layers.Dense(1)

    def call(self, inputs):
        ctl, drug_targets, cell_idx, drug_fp = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        if len(ctl.shape) == 3 and ctl.shape[-1] == 2:
            ctl_base = ctl[..., 0]
        elif len(ctl.shape) == 2:
            ctl_base = ctl
        else:
            ctl_base = tf.squeeze(ctl, axis=-1)

        if len(drug_targets.shape) == 2:
            t = drug_targets
        else:
            t = tf.squeeze(drug_targets, axis=-1)

        X = tf.expand_dims(t, axis=-1) * self.M[None, :, :]
        t_mod = tf.reduce_max(X, axis=1)
        z_mod = self.appnp(t_mod, self.A)
        z_gene = tf.matmul(z_mod, self.Mt)

        x = tf.stack([ctl_base, t, z_gene], axis=-1)
        h = self.shared(x)
        if self.use_cell:
            c = self.cell_emb(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(h)[1], 1])
            h = h + c
        y = tf.squeeze(self.out(h), axis=-1)
        return y


class SchemeB(keras.Model):
    def __init__(self, M_gk, A_kk, num_cells, hidden_dim=64, steps=6, alpha=0.1, use_cell=True):
        super().__init__()
        self.M = tf.constant(M_gk, dtype=tf.float32)
        self.A = tf.constant(A_kk, dtype=tf.float32)
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.use_cell = bool(use_cell)
        self.shared = layers.Dense(int(hidden_dim), activation="relu")
        self.mod_lin = layers.Dense(int(hidden_dim), activation="relu")
        if self.use_cell:
            self.cell_emb = layers.Embedding(int(num_cells), int(hidden_dim))
        self.out = layers.Dense(1)

        deg = tf.reduce_sum(self.M, axis=0)
        self.mod_deg = tf.maximum(deg, 1.0)

    def call(self, inputs):
        ctl, drug_targets, cell_idx, drug_fp = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        if len(ctl.shape) == 3 and ctl.shape[-1] == 2:
            ctl_base = ctl[..., 0]
        elif len(ctl.shape) == 2:
            ctl_base = ctl
        else:
            ctl_base = tf.squeeze(ctl, axis=-1)

        if len(drug_targets.shape) == 2:
            t = drug_targets
        else:
            t = tf.squeeze(drug_targets, axis=-1)

        x = tf.stack([ctl_base, t], axis=-1)
        h_gene = self.shared(x)

        h_mod = tf.einsum("gk,bgh->bkh", self.M, h_gene) / self.mod_deg[None, :, None]
        S = self.mod_lin(h_mod)
        Z = tf.identity(S)
        for _ in range(self.steps):
            Z = (1.0 - self.alpha) * tf.einsum("bkh,kl->blh", Z, self.A) + self.alpha * S

        h_back = tf.einsum("gk,bkh->bgh", self.M, Z)
        h = h_gene + h_back
        if self.use_cell:
            c = self.cell_emb(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(h)[1], 1])
            h = h + c
        y = tf.squeeze(self.out(h), axis=-1)
        return y


class SchemeC(keras.Model):
    def __init__(self, M_gk, A_kk, num_cells, hidden_dim=64, steps=10, alpha=0.1, use_cell=True):
        super().__init__()
        self.M = tf.constant(M_gk, dtype=tf.float32)
        self.Mt = tf.transpose(self.M)
        self.A = tf.constant(A_kk, dtype=tf.float32)
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.use_cell = bool(use_cell)
        self.mod_in = layers.Dense(int(hidden_dim), activation="relu")
        if self.use_cell:
            self.cell_emb = layers.Embedding(int(num_cells), int(hidden_dim))
        self.out = layers.Dense(1)
        deg = tf.reduce_sum(self.M, axis=0)
        self.mod_deg = tf.maximum(deg, 1.0)

    def call(self, inputs):
        ctl, drug_targets, cell_idx, drug_fp = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        if len(ctl.shape) == 3 and ctl.shape[-1] == 2:
            ctl_base = ctl[..., 0]
        elif len(ctl.shape) == 2:
            ctl_base = ctl
        else:
            ctl_base = tf.squeeze(ctl, axis=-1)

        if len(drug_targets.shape) == 2:
            t = drug_targets
        else:
            t = tf.squeeze(drug_targets, axis=-1)

        x_gene = tf.stack([ctl_base, t], axis=-1)
        x_mod = tf.einsum("gk,bgh->bkh", self.M, x_gene) / self.mod_deg[None, :, None]
        S = self.mod_in(x_mod)
        Z = tf.identity(S)
        for _ in range(self.steps):
            Z = (1.0 - self.alpha) * tf.einsum("bkh,kl->blh", Z, self.A) + self.alpha * S

        h_gene = tf.einsum("gk,bkh->bgh", self.M, Z)
        if self.use_cell:
            c = self.cell_emb(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(h_gene)[1], 1])
            h_gene = h_gene + c
        y = tf.squeeze(self.out(h_gene), axis=-1)
        return y


def load_module_graph_np(root, use_landmark, k_modules, consensus_only, is_directed_only):
    from module_graph_recipe import (
        build_gene_module_matrix,
        build_tf_regulon_modules,
        coarsen_edges_to_modules,
        load_omnipath_ppi_edges,
        load_omnipath_tf_edges,
        load_symbol_to_entrez,
        load_target_genes,
        select_modules_greedy,
    )

    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    landmark_json = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")

    symbol_to_entrez = load_symbol_to_entrez(full_gene_path)
    target_genes = load_target_genes(landmark_json, bool(use_landmark), full_gene_path)
    node_list = list(target_genes)
    gene2idx = {g: i for i, g in enumerate(node_list)}
    target_set = set(node_list)

    tf_edges = load_omnipath_tf_edges(tf_path, symbol_to_entrez, target_set, consensus_only=bool(consensus_only))
    ppi_dir, ppi_undir = load_omnipath_ppi_edges(
        ppi_path,
        symbol_to_entrez,
        target_set,
        consensus_only=bool(consensus_only),
        is_directed_only=bool(is_directed_only),
    )
    edges_directed_signed = tf_edges + ppi_dir
    edges_undirected = [] if (bool(consensus_only) or bool(is_directed_only)) else ppi_undir

    base_modules = build_tf_regulon_modules(tf_edges, min_targets=5, max_targets=200)
    modules = select_modules_greedy(
        base_modules,
        node_list,
        k=int(k_modules),
        min_cover_frac=0.9,
        max_overlap_jaccard=0.9,
    )
    M = build_gene_module_matrix(gene2idx, modules).astype(np.float32)
    M_gk = M.toarray().astype(np.float32)
    A_pos, A_neg, A_undir = coarsen_edges_to_modules(
        gene2idx,
        modules,
        edges_directed_signed=edges_directed_signed,
        edges_undirected=edges_undirected,
        distribute="uniform",
        keep_top_per_src=0,
    )
    A = build_signed_module_adj(A_pos, A_neg, A_undir)
    return node_list, M_gk, A


def cold_split(drug_ids, cell_idx, mode, test_frac, seed):
    rng = np.random.default_rng(int(seed))
    if mode == "cold_cell":
        uniq = np.unique(cell_idx)
        if len(uniq) < 2:
            raise ValueError("cold_cell needs >=2 cells")
        n_test = max(1, int(len(uniq) * float(test_frac)))
        n_test = min(n_test, len(uniq) - 1)
        held = rng.choice(uniq, size=n_test, replace=False)
        test_mask = np.isin(cell_idx, held)
        return ~test_mask, test_mask
    uniq = np.unique(drug_ids)
    if len(uniq) < 2:
        raise ValueError("cold_drug needs >=2 drugs")
    n_test = max(1, int(len(uniq) * float(test_frac)))
    n_test = min(n_test, len(uniq) - 1)
    held = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(drug_ids, held)
    return ~test_mask, test_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    p.add_argument("--cell_line", default="ALL")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--k_modules", type=int, default=200)
    p.add_argument("--split_mode", choices=["cold_drug", "cold_cell"], default="cold_drug")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--consensus_only", action="store_true", default=False)
    p.add_argument("--is_directed_only", action="store_true", default=False)
    p.add_argument("--no_cell", action="store_true", default=False)
    p.add_argument("--residualize_by_cell", action="store_true", default=True)
    args = p.parse_args()

    root = args.root
    if not os.path.exists(root):
        root = "/local/data1/liume102/rfa"
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from data_loader import load_rfa_data

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
    fp = np.asarray(data["X_fingerprint"], dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names = np.asarray(data["cell_names"], dtype=str)
    loss_mask = np.asarray(data["loss_mask"], dtype=np.float32)

    if int(args.max_samples) > 0 and len(y) > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(y), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y = y[idx]
        X_drug = X_drug[idx]
        fp = fp[idx]
        drug_ids = drug_ids[idx]
        cell_names = cell_names[idx]

    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names)
    num_cells = int(len(le.classes_))

    train_mask, test_mask = cold_split(drug_ids, cell_idx, str(args.split_mode), float(args.test_frac), int(args.seed))
    tr = np.where(train_mask)[0]
    te = np.where(test_mask)[0]
    print(f"Train samples: {len(tr)} | Test samples: {len(te)}")

    node_list, M_gk, A_kk = load_module_graph_np(
        root,
        use_landmark=True,
        k_modules=int(args.k_modules),
        consensus_only=bool(args.consensus_only),
        is_directed_only=bool(args.is_directed_only),
    )
    print(f"Modules: {int(M_gk.shape[1])} | Module edges: {int(np.count_nonzero(A_kk))}")

    x_train = [X_ctl[tr], X_drug[tr], cell_idx[tr], fp[tr]]
    y_train_full = y[tr]
    x_test = [X_ctl[te], X_drug[te], cell_idx[te], fp[te]]
    y_test_full = y[te]

    if bool(args.residualize_by_cell):
        sums = np.zeros((num_cells, y.shape[1]), dtype=np.float32)
        cnts = np.zeros((num_cells,), dtype=np.int64)
        np.add.at(sums, cell_idx[tr], y_train_full)
        np.add.at(cnts, cell_idx[tr], 1)
        cell_mean = sums / np.maximum(cnts[:, None], 1)
        y_train = y_train_full - cell_mean[cell_idx[tr]]
        y_test = y_test_full - cell_mean[cell_idx[te]]
    else:
        cell_mean = None
        y_train = y_train_full
        y_test = y_test_full

    schemes = [
        ("A", SchemeA(M_gk, A_kk, num_cells=num_cells, hidden_dim=int(args.hidden_dim), use_cell=not bool(args.no_cell))),
        ("B", SchemeB(M_gk, A_kk, num_cells=num_cells, hidden_dim=int(args.hidden_dim), use_cell=not bool(args.no_cell))),
        ("C", SchemeC(M_gk, A_kk, num_cells=num_cells, hidden_dim=int(args.hidden_dim), use_cell=not bool(args.no_cell))),
    ]

    for name, model in schemes:
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        best = -1e9
        for ep in range(int(args.epochs)):
            model.fit(x_train, y_train, batch_size=int(args.batch_size), epochs=1, verbose=0)
            pred = model.predict(x_test, batch_size=int(args.batch_size), verbose=0)
            if cell_mean is not None:
                pred_eval = pred + cell_mean[cell_idx[te]]
                y_eval = y_test_full
            else:
                pred_eval = pred
                y_eval = y_test
            pcc = samplewise_pcc(y_eval, pred_eval, loss_mask)
            mse = samplewise_mse(y_eval, pred_eval, loss_mask)
            best = max(best, pcc)
            print(f"Scheme {name} | epoch {ep+1} | val_pcc={pcc:.4f} | val_mse={mse:.4f}")
        print(f"Scheme {name} best_val_pcc={best:.4f}")


if __name__ == "__main__":
    main()
