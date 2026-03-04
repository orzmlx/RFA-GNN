import argparse
import os
import sys

import numpy as np
import pandas as pd
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


def load_prior_edges(tf_path, ppi_path, symbol_to_entrez, target_set, consensus_only=True):
    tf_df = pd.read_csv(tf_path, dtype=str, low_memory=False)
    for c in ["consensus_direction", "consensus_stimulation", "consensus_inhibition"]:
        if c not in tf_df.columns:
            tf_df[c] = False
    tf_df["consensus_direction"] = tf_df["consensus_direction"].fillna(False).astype(bool)
    tf_df["consensus_stimulation"] = tf_df["consensus_stimulation"].fillna(False).astype(bool)
    tf_df["consensus_inhibition"] = tf_df["consensus_inhibition"].fillna(False).astype(bool)

    def _sign(stim, inhib):
        return -1.0 if (bool(inhib) and not bool(stim)) else 1.0

    directed_signed = []
    for _, r in tf_df.iterrows():
        if bool(consensus_only) and not bool(r["consensus_direction"]):
            continue
        s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
        t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
        if not s or not t:
            continue
        s = _norm_entrez(s)
        t = _norm_entrez(t)
        if s in target_set and t in target_set:
            directed_signed.append((s, t, _sign(r["consensus_stimulation"], r["consensus_inhibition"])))

    ppi_df = pd.read_csv(ppi_path, dtype=str, low_memory=False)
    for c in ["consensus_direction", "is_directed", "consensus_stimulation", "consensus_inhibition"]:
        if c not in ppi_df.columns:
            ppi_df[c] = False
    ppi_df["consensus_direction"] = ppi_df["consensus_direction"].fillna(False).astype(bool)
    ppi_df["is_directed"] = ppi_df["is_directed"].fillna(False).astype(bool)
    ppi_df["consensus_stimulation"] = ppi_df["consensus_stimulation"].fillna(False).astype(bool)
    ppi_df["consensus_inhibition"] = ppi_df["consensus_inhibition"].fillna(False).astype(bool)

    undirected = []
    for _, r in ppi_df.iterrows():
        if bool(consensus_only) and not bool(r["consensus_direction"]):
            continue
        s = symbol_to_entrez.get(_norm_symbol(r["source_genesymbol"]))
        t = symbol_to_entrez.get(_norm_symbol(r["target_genesymbol"]))
        if not s or not t:
            continue
        s = _norm_entrez(s)
        t = _norm_entrez(t)
        if s in target_set and t in target_set and s != t:
            if bool(r["consensus_direction"]) or bool(r["is_directed"]):
                directed_signed.append((s, t, _sign(r["consensus_stimulation"], r["consensus_inhibition"])))
            else:
                undirected.append((s, t))

    return directed_signed, undirected


def build_correlation_graph_modules(y_train, gene_ids, top_k=30, min_corr=0.2, min_size=8, max_size=200, resolution=2.0):
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    y = np.asarray(y_train, dtype=np.float32)
    if y.shape[0] < 50:
        raise ValueError("y_train 太小，不适合做共响应模块")
    corr = np.corrcoef(y, rowvar=False).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    G = corr.shape[0]

    graph = nx.Graph()
    graph.add_nodes_from(range(G))
    for i in range(G):
        row = corr[i]
        idx = np.argpartition(-row, kth=min(int(top_k), G - 1) - 1)[: int(top_k)]
        for j in idx:
            if j == i:
                continue
            w = float(row[j])
            if w < float(min_corr):
                continue
            graph.add_edge(i, int(j), weight=w)

    def _split_community(nodes, base_res=1.0):
        nodes = list(nodes)
        if len(nodes) <= int(max_size):
            return [nodes]
        sub = graph.subgraph(nodes).copy()
        res = float(base_res)
        out = []
        queue = [(sub, res)]
        while queue:
            sg, r = queue.pop()
            ns = list(sg.nodes())
            if len(ns) <= int(max_size):
                out.append(ns)
                continue
            comms2 = louvain_communities(sg, resolution=float(r), weight="weight", seed=42)
            if len(comms2) <= 1:
                if r >= 5.0:
                    out.append(ns[: int(max_size)])
                    for k in range(int(max_size), len(ns), int(max_size)):
                        out.append(ns[k : k + int(max_size)])
                    continue
                queue.append((sg, float(r) * 1.5))
                continue
            for c2 in comms2:
                c2 = list(c2)
                if len(c2) <= int(max_size):
                    out.append(c2)
                else:
                    queue.append((sg.subgraph(c2).copy(), float(r) * 1.2))
        return out

    comms = louvain_communities(graph, resolution=float(resolution), weight="weight", seed=42)
    modules = []
    cid = 0
    for c in comms:
        for cc in _split_community(c, base_res=float(resolution)):
            if len(cc) < int(min_size):
                continue
            cid += 1
            genes = {gene_ids[i] for i in cc}
            modules.append({"name": f"CORR_COMM:{cid}", "genes": genes})
    modules.sort(key=lambda x: len(x["genes"]), reverse=True)
    return modules


def refine_modules_with_prior(modules, prior_undirected_edges, min_size=8, max_size=200, resolution=1.0):
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    edge_set = set()
    for u, v in prior_undirected_edges:
        if u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        edge_set.add((a, b))

    refined = []
    rid = 0
    for m in modules:
        genes = sorted(list(set(m["genes"])))
        if len(genes) < int(min_size):
            continue
        G = nx.Graph()
        G.add_nodes_from(genes)
        for u in genes:
            for v in genes:
                if u >= v:
                    continue
                if (u, v) in edge_set:
                    G.add_edge(u, v, weight=1.0)
        if G.number_of_edges() == 0:
            if len(genes) <= int(max_size):
                rid += 1
                refined.append({"name": f"PRIOR_KEEP:{rid}", "genes": set(genes)})
            continue
        comms = louvain_communities(G, resolution=float(resolution), weight="weight", seed=42)
        for c in comms:
            c = set(c)
            if len(c) < int(min_size) or len(c) > int(max_size):
                continue
            rid += 1
            refined.append({"name": f"PRIOR_REF:{rid}", "genes": c})
    refined.sort(key=lambda x: len(x["genes"]), reverse=True)
    return refined


def build_gene_module_matrix(gene2idx, modules):
    rows = []
    cols = []
    data = []
    for j, m in enumerate(modules):
        for g in m["genes"]:
            i = gene2idx.get(g)
            if i is None:
                continue
            rows.append(i)
            cols.append(j)
            data.append(1.0)
    G = len(gene2idx)
    K = len(modules)
    return rows, cols, data, G, K


def build_module_adj_from_prior(gene2idx, modules, directed_signed):
    idx2mods = {}
    for midx, m in enumerate(modules):
        for g in m["genes"]:
            gi = gene2idx.get(g)
            if gi is None:
                continue
            idx2mods.setdefault(gi, []).append(midx)

    K = len(modules)
    A = np.zeros((K, K), dtype=np.float32)
    for s, t, sign in directed_signed:
        si = gene2idx.get(s)
        ti = gene2idx.get(t)
        if si is None or ti is None:
            continue
        sm = idx2mods.get(si)
        tm = idx2mods.get(ti)
        if not sm or not tm:
            continue
        w = 1.0 / (len(sm) * len(tm))
        for u in sm:
            for v in tm:
                A[u, v] += w * float(sign)

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


class ModulePropagationModel(keras.Model):
    def __init__(self, M_gk, A_kk, num_cells, hidden_dim=64, steps=10, alpha=0.1, pool="max", use_cell=True):
        super().__init__()
        self.M = tf.constant(M_gk, dtype=tf.float32)
        self.Mt = tf.transpose(self.M)
        self.A = tf.constant(A_kk, dtype=tf.float32)
        self.pool = str(pool)
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
        else:
            ctl_base = ctl if len(ctl.shape) == 2 else tf.squeeze(ctl, axis=-1)

        if len(drug_targets.shape) == 2:
            t = drug_targets
        else:
            t = tf.squeeze(drug_targets, axis=-1)

        if self.pool == "max":
            X = tf.expand_dims(t, axis=-1) * self.M[None, :, :]
            t_mod = tf.reduce_max(X, axis=1)
        else:
            t_mod = tf.matmul(t, self.M)
            deg = tf.reduce_sum(self.M, axis=0)
            t_mod = t_mod / tf.maximum(deg[None, :], 1.0)

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
    p.add_argument("--module_max_samples", type=int, default=15000)
    p.add_argument("--corr_top_k", type=int, default=30)
    p.add_argument("--corr_min", type=float, default=0.2)
    p.add_argument("--corr_resolution", type=float, default=2.0)
    p.add_argument("--module_min_size", type=int, default=5)
    p.add_argument("--module_max_size", type=int, default=200)
    p.add_argument("--prior_refine", action="store_true", default=False)
    p.add_argument("--pool", choices=["max", "mean"], default="max")
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
    y_te_resid = y[te] - cell_mean[cell_idx[te]]

    y_mod = y_tr_resid
    if int(args.module_max_samples) > 0 and len(y_mod) > int(args.module_max_samples):
        rng = np.random.default_rng(int(args.seed))
        idxm = rng.choice(len(y_mod), size=int(args.module_max_samples), replace=False)
        y_mod = y_mod[idxm]

    node_list = [_norm_entrez(x) for x in data["target_genes"]]
    gene2idx = {g: i for i, g in enumerate(node_list)}

    symbol_to_entrez = build_symbol_to_entrez(full_gene_path)
    target_set = set(node_list)
    directed_signed, prior_undir = load_prior_edges(tf_path, ppi_path, symbol_to_entrez, target_set, consensus_only=True)

    base_modules = build_correlation_graph_modules(
        y_mod,
        gene_ids=node_list,
        top_k=int(args.corr_top_k),
        min_corr=float(args.corr_min),
        min_size=int(args.module_min_size),
        max_size=int(args.module_max_size),
        resolution=float(args.corr_resolution),
    )
    if bool(args.prior_refine):
        modules = refine_modules_with_prior(
            base_modules,
            prior_undir,
            min_size=int(args.module_min_size),
            max_size=int(args.module_max_size),
            resolution=1.0,
        )
    else:
        modules = base_modules
    if len(modules) == 0:
        raise RuntimeError("modules is empty")

    rows, cols, vals, G, K = build_gene_module_matrix(gene2idx, modules)
    M = np.zeros((G, K), dtype=np.float32)
    M[rows, cols] = 1.0
    A = build_module_adj_from_prior(gene2idx, modules, directed_signed)
    print(f"Modules: {K} | Module edges nnz: {int(np.count_nonzero(A))}")

    model = ModulePropagationModel(M, A, num_cells=num_cells, hidden_dim=int(args.hidden_dim), pool=str(args.pool), use_cell=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    x_train = [X_ctl[tr], X_drug[tr], cell_idx[tr], fp[tr]]
    x_test = [X_ctl[te], X_drug[te], cell_idx[te], fp[te]]

    best = -1e9
    for ep in range(int(args.epochs)):
        model.fit(x_train, y_tr_resid, batch_size=int(args.batch_size), epochs=1, verbose=0)
        pred_resid = model.predict(x_test, batch_size=int(args.batch_size), verbose=0)
        pred = pred_resid + cell_mean[cell_idx[te]]
        m = samplewise_pcc_mse(y[te], pred, loss_mask)
        best = max(best, m["pcc"])
        tag = "DualView" if bool(args.prior_refine) else "CoRespOnly"
        print(f"{tag} | epoch {ep+1} | val_pcc={m['pcc']:.4f} | val_mse={m['mse']:.4f}")
    print(f"Best val_pcc: {best:.4f}")


if __name__ == "__main__":
    main()
