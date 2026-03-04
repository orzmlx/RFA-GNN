import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _norm_symbol(s):
    return str(s).strip().upper()


def _norm_entrez(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_symbol_to_entrez(full_gene_path: str):
    df = pd.read_csv(full_gene_path, sep="\t", dtype=str, low_memory=False)
    if "pr_gene_symbol" not in df.columns or "pr_gene_id" not in df.columns:
        raise ValueError("full_gene_path 缺少 pr_gene_symbol/pr_gene_id 列")
    m = {}
    for sym, eid in zip(df["pr_gene_symbol"].astype(str), df["pr_gene_id"].astype(str)):
        s = _norm_symbol(sym)
        e = _norm_entrez(eid)
        if s and e:
            m[s] = e
    return m


def load_landmark_entrez(landmark_json_path: str):
    import json

    with open(landmark_json_path, "r") as f:
        genes_meta = json.load(f)
    target_genes = [str(g["entrez_id"]) for g in genes_meta if "entrez_id" in g]
    target_genes = [_norm_entrez(x) for x in target_genes if str(x).strip() != ""]
    target_genes = list(dict.fromkeys(target_genes))
    return target_genes


def load_omnipath_tf_edges(tf_path: str, symbol_to_entrez: dict, target_set: set, consensus_only: bool):
    df = pd.read_csv(tf_path, dtype=str, low_memory=False)
    for c in ["source_genesymbol", "target_genesymbol"]:
        if c not in df.columns:
            raise ValueError("TF 文件缺少 source_genesymbol/target_genesymbol")
    for c in ["consensus_direction", "consensus_stimulation", "consensus_inhibition"]:
        if c not in df.columns:
            df[c] = False
    df["consensus_direction"] = df["consensus_direction"].fillna(False).astype(bool)
    df["consensus_stimulation"] = df["consensus_stimulation"].fillna(False).astype(bool)
    df["consensus_inhibition"] = df["consensus_inhibition"].fillna(False).astype(bool)

    edges = []
    for _, row in df.iterrows():
        if consensus_only and not bool(row["consensus_direction"]):
            continue
        src = symbol_to_entrez.get(_norm_symbol(row["source_genesymbol"]))
        tgt = symbol_to_entrez.get(_norm_symbol(row["target_genesymbol"]))
        if not src or not tgt:
            continue
        src = _norm_entrez(src)
        tgt = _norm_entrez(tgt)
        if src not in target_set or tgt not in target_set:
            continue
        stim = bool(row["consensus_stimulation"])
        inhib = bool(row["consensus_inhibition"])
        sign = -1.0 if (inhib and not stim) else 1.0
        edges.append((src, tgt, sign))
    return edges


def load_omnipath_ppi_edges(ppi_path: str, symbol_to_entrez: dict, target_set: set, consensus_only: bool, is_directed_only: bool):
    df = pd.read_csv(ppi_path, dtype=str, low_memory=False)
    for c in ["source_genesymbol", "target_genesymbol"]:
        if c not in df.columns:
            raise ValueError("PPI 文件缺少 source_genesymbol/target_genesymbol")
    for c in ["consensus_direction", "is_directed", "consensus_stimulation", "consensus_inhibition"]:
        if c not in df.columns:
            df[c] = False
    df["consensus_direction"] = df["consensus_direction"].fillna(False).astype(bool)
    df["is_directed"] = df["is_directed"].fillna(False).astype(bool)
    df["consensus_stimulation"] = df["consensus_stimulation"].fillna(False).astype(bool)
    df["consensus_inhibition"] = df["consensus_inhibition"].fillna(False).astype(bool)

    directed_edges = []
    undirected_edges = []
    for _, row in df.iterrows():
        if consensus_only and not bool(row["consensus_direction"]):
            continue
        if is_directed_only and not bool(row["is_directed"]):
            continue
        src = symbol_to_entrez.get(_norm_symbol(row["source_genesymbol"]))
        tgt = symbol_to_entrez.get(_norm_symbol(row["target_genesymbol"]))
        if not src or not tgt:
            continue
        src = _norm_entrez(src)
        tgt = _norm_entrez(tgt)
        if src not in target_set or tgt not in target_set or src == tgt:
            continue
        stim = bool(row["consensus_stimulation"])
        inhib = bool(row["consensus_inhibition"])
        sign = -1.0 if (inhib and not stim) else 1.0
        if bool(row["consensus_direction"]) or bool(row["is_directed"]):
            directed_edges.append((src, tgt, sign))
        else:
            undirected_edges.append((src, tgt, 1.0))
    return directed_edges, undirected_edges


def build_tf_regulon_modules(tf_edges, min_targets=5, max_targets=200):
    tf2t = {}
    for s, t, _ in tf_edges:
        tf2t.setdefault(s, set()).add(t)
    mods = []
    for tf, targets in tf2t.items():
        n = len(targets)
        if n < int(min_targets) or n > int(max_targets):
            continue
        mods.append({"name": f"TF:{tf}", "genes": set(targets)})
    mods.sort(key=lambda x: len(x["genes"]), reverse=True)
    return mods


def build_ppi_community_modules(node_list, undirected_edges, resolution=1.0, min_size=8, max_size=200):
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    G = nx.Graph()
    G.add_nodes_from(node_list)
    for u, v, _ in undirected_edges:
        if u != v:
            G.add_edge(u, v, weight=1.0)
    comms = louvain_communities(G, resolution=float(resolution), weight="weight", seed=42)
    mods = []
    idx = 0
    for c in comms:
        c = set(c)
        n = len(c)
        if n < int(min_size) or n > int(max_size):
            continue
        idx += 1
        mods.append({"name": f"PPI_COMM:{idx}", "genes": c})
    mods.sort(key=lambda x: len(x["genes"]), reverse=True)
    return mods


def build_nmf_expression_modules(y_train, gene_ids, n_components=64, top_genes=25, max_iter=200, seed=42):
    from sklearn.decomposition import NMF

    y = np.asarray(y_train, dtype=np.float32)
    y = y - np.min(y)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    model = NMF(
        n_components=int(n_components),
        init="nndsvda",
        random_state=int(seed),
        max_iter=int(max_iter),
        tol=1e-4,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H=0.0,
    )
    W = model.fit_transform(y)
    H = model.components_
    mods = []
    for k in range(H.shape[0]):
        row = H[k]
        idx = np.argpartition(-row, kth=min(int(top_genes), len(row) - 1) - 1)[: int(top_genes)]
        idx = idx[np.argsort(-row[idx])]
        genes = {gene_ids[i] for i in idx}
        mods.append({"name": f"NMF:{k+1}", "genes": genes})
    return mods


def dedup_modules(modules, max_overlap_jaccard=0.95):
    selected = []
    for m in modules:
        g = set(m["genes"])
        if not g:
            continue
        ok = True
        for s in selected:
            a = g
            b = s["genes"]
            inter = len(a & b)
            uni = len(a | b)
            j = (inter / uni) if uni > 0 else 0.0
            if j >= float(max_overlap_jaccard):
                ok = False
                break
        if ok:
            selected.append({"name": m["name"], "genes": g})
    return selected


def select_modules_cover_greedy(modules, target_genes, k_total=200, min_cover_frac=0.9, max_overlap_jaccard=0.95):
    target_genes = list(target_genes)
    target_set = set(target_genes)
    covered = set()
    selected = []
    for m in modules:
        if len(selected) >= int(k_total):
            break
        g = set(m["genes"]) & target_set
        if len(g) == 0:
            continue
        ok = True
        for s in selected:
            a = g
            b = s["genes"]
            inter = len(a & b)
            uni = len(a | b)
            j = (inter / uni) if uni > 0 else 0.0
            if j >= float(max_overlap_jaccard):
                ok = False
                break
        if not ok:
            continue
        selected.append({"name": m["name"], "genes": g})
        covered |= g
        if len(covered) / max(1, len(target_set)) >= float(min_cover_frac) and len(selected) >= min(50, int(k_total)):
            break
    return selected


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
    return sp.csr_matrix((data, (rows, cols)), shape=(G, K), dtype=np.float32)


def coarsen_edges_to_modules(gene2idx, modules, edges_directed_signed, edges_undirected, keep_top_per_src=0):
    idx2mods = {}
    for midx, m in enumerate(modules):
        for g in m["genes"]:
            gi = gene2idx.get(g)
            if gi is None:
                continue
            idx2mods.setdefault(gi, []).append(midx)

    K = len(modules)
    A_pos = np.zeros((K, K), dtype=np.float32)
    A_neg = np.zeros((K, K), dtype=np.float32)
    A_undir = np.zeros((K, K), dtype=np.float32)

    for s, t, sign in edges_directed_signed:
        si = gene2idx.get(s)
        ti = gene2idx.get(t)
        if si is None or ti is None:
            continue
        smods = idx2mods.get(si)
        tmods = idx2mods.get(ti)
        if not smods or not tmods:
            continue
        w = 1.0 / (len(smods) * len(tmods))
        if float(sign) >= 0:
            for u in smods:
                for v in tmods:
                    A_pos[u, v] += w
        else:
            for u in smods:
                for v in tmods:
                    A_neg[u, v] += w

    for s, t, _ in edges_undirected:
        si = gene2idx.get(s)
        ti = gene2idx.get(t)
        if si is None or ti is None:
            continue
        smods = idx2mods.get(si)
        tmods = idx2mods.get(ti)
        if not smods or not tmods:
            continue
        w = 1.0 / (len(smods) * len(tmods))
        for u in smods:
            for v in tmods:
                A_undir[u, v] += w
                A_undir[v, u] += w

    if int(keep_top_per_src) > 0:
        k = int(keep_top_per_src)
        for mat in (A_pos, A_neg, A_undir):
            for i in range(K):
                row = mat[i]
                if np.count_nonzero(row) <= k:
                    continue
                idx = np.argpartition(-row, kth=k - 1)[:k]
                keep = np.zeros((K,), dtype=bool)
                keep[idx] = True
                row[~keep] = 0.0

    return A_pos, A_neg, A_undir


def build_signed_module_adj(A_pos, A_neg, A_undir):
    A = A_pos - A_neg + A_undir
    A = np.asarray(A, dtype=np.float32)
    row_norm = np.sum(np.abs(A), axis=1, keepdims=True)
    row_norm = np.maximum(row_norm, 1.0)
    return A / row_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    p.add_argument("--max_samples_for_nmf", type=int, default=8000)
    p.add_argument("--nmf_components", type=int, default=80)
    p.add_argument("--nmf_top_genes", type=int, default=25)
    p.add_argument("--ppi_resolution", type=float, default=1.0)
    p.add_argument("--k_total", type=int, default=200)
    p.add_argument("--consensus_only", action="store_true", default=False)
    p.add_argument("--is_directed_only", action="store_true", default=False)
    p.add_argument("--out", default="data/module_graph_plus.npz")
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
    landmark_json = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    symbol_to_entrez = load_symbol_to_entrez(full_gene_path)
    node_list = load_landmark_entrez(landmark_json)
    gene2idx = {g: i for i, g in enumerate(node_list)}
    target_set = set(node_list)

    tf_edges = load_omnipath_tf_edges(tf_path, symbol_to_entrez, target_set, consensus_only=bool(args.consensus_only))
    ppi_dir, ppi_undir = load_omnipath_ppi_edges(
        ppi_path,
        symbol_to_entrez,
        target_set,
        consensus_only=bool(args.consensus_only),
        is_directed_only=bool(args.is_directed_only),
    )

    tf_mods = build_tf_regulon_modules(tf_edges, min_targets=5, max_targets=200)
    ppi_mods = build_ppi_community_modules(node_list, ppi_undir, resolution=float(args.ppi_resolution), min_size=8, max_size=200)

    data = load_rfa_data(
        ctl_path,
        trt_path,
        drug_target_path=drug_target_path,
        landmark_path=landmark_json,
        siginfo_path=siginfo_path,
        fingerprint_path=fingerprint_path,
        use_landmark_genes=True,
        full_gene_path=full_gene_path,
        cell_lines=None,
        ctl_residual_pool_size=0,
    )
    y = np.asarray(data["y_delta"], dtype=np.float32)
    if int(args.max_samples_for_nmf) > 0 and len(y) > int(args.max_samples_for_nmf):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), size=int(args.max_samples_for_nmf), replace=False)
        y = y[idx]
    nmf_mods = build_nmf_expression_modules(
        y,
        gene_ids=node_list,
        n_components=int(args.nmf_components),
        top_genes=int(args.nmf_top_genes),
        max_iter=200,
        seed=42,
    )

    all_mods = tf_mods + ppi_mods + nmf_mods
    all_mods = dedup_modules(all_mods, max_overlap_jaccard=0.95)
    modules = select_modules_cover_greedy(all_mods, node_list, k_total=int(args.k_total), min_cover_frac=0.95, max_overlap_jaccard=0.95)
    M = build_gene_module_matrix(gene2idx, modules)

    edges_directed_signed = tf_edges + ppi_dir
    edges_undirected = [] if (bool(args.consensus_only) or bool(args.is_directed_only)) else ppi_undir
    A_pos, A_neg, A_undir = coarsen_edges_to_modules(
        gene2idx,
        modules,
        edges_directed_signed=edges_directed_signed,
        edges_undirected=edges_undirected,
        keep_top_per_src=50,
    )
    A = build_signed_module_adj(A_pos, A_neg, A_undir)

    out_path = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sp.save_npz(out_path.replace(".npz", ".M.npz"), M)
    sp.save_npz(out_path.replace(".npz", ".A_pos.npz"), sp.csr_matrix(A_pos))
    sp.save_npz(out_path.replace(".npz", ".A_neg.npz"), sp.csr_matrix(A_neg))
    sp.save_npz(out_path.replace(".npz", ".A_undir.npz"), sp.csr_matrix(A_undir))
    np.savez_compressed(out_path, node_list=np.asarray(node_list, dtype=object), module_names=np.asarray([m["name"] for m in modules], dtype=object), A=A)

    sizes = np.array([len(m["genes"]) for m in modules], dtype=np.int32)
    covered = set()
    for m in modules:
        covered |= set(m["genes"])
    print(f"Modules={len(modules)} cover={len(covered)}/{len(node_list)} median_size={float(np.median(sizes)):.1f} max_size={int(sizes.max())}")
    print(f"Module A nnz={int(np.count_nonzero(A))}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

