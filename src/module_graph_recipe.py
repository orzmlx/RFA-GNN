import argparse
import json
import os

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
    m = {}
    if "pr_gene_symbol" not in df.columns or "pr_gene_id" not in df.columns:
        raise ValueError("full_gene_path 缺少 pr_gene_symbol/pr_gene_id 列")
    for sym, eid in zip(df["pr_gene_symbol"].astype(str), df["pr_gene_id"].astype(str)):
        s = _norm_symbol(sym)
        e = _norm_entrez(eid)
        if s and e:
            m[s] = e
    return m


def load_target_genes(landmark_json_path: str, use_landmark: bool, full_gene_path: str):
    if use_landmark:
        with open(landmark_json_path, "r") as f:
            genes_meta = json.load(f)
        target_genes = [str(g["entrez_id"]) for g in genes_meta if "entrez_id" in g]
        target_genes = [_norm_entrez(x) for x in target_genes if str(x).strip() != ""]
        target_genes = list(dict.fromkeys(target_genes))
        return target_genes
    df = pd.read_csv(full_gene_path, sep="\t", dtype=str, low_memory=False)
    if "pr_gene_id" not in df.columns:
        raise ValueError("full_gene_path 缺少 pr_gene_id 列")
    target_genes = [_norm_entrez(x) for x in df["pr_gene_id"].astype(str).tolist()]
    target_genes = [x for x in target_genes if x]
    target_genes = list(dict.fromkeys(target_genes))
    return target_genes


def _sign_from_row(row):
    stim = row.get("consensus_stimulation")
    inhib = row.get("consensus_inhibition")
    try:
        s = bool(stim)
        h = bool(inhib)
    except Exception:
        s = False
        h = False
    if h and not s:
        return -1.0
    if s and not h:
        return 1.0
    return 1.0


def load_omnipath_tf_edges(tf_path: str, symbol_to_entrez: dict, target_set: set, consensus_only: bool):
    df = pd.read_csv(tf_path, dtype=str, low_memory=False)
    required = {"source_genesymbol", "target_genesymbol"}
    if not required.issubset(set(df.columns)):
        raise ValueError("TF 文件缺少 source_genesymbol/target_genesymbol")
    if "consensus_direction" not in df.columns:
        df["consensus_direction"] = False
    if "consensus_stimulation" not in df.columns:
        df["consensus_stimulation"] = False
    if "consensus_inhibition" not in df.columns:
        df["consensus_inhibition"] = False

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
        if src in target_set and tgt in target_set:
            sign = _sign_from_row(row)
            edges.append((src, tgt, sign))
    return edges


def load_omnipath_ppi_edges(ppi_path: str, symbol_to_entrez: dict, target_set: set, consensus_only: bool, is_directed_only: bool):
    df = pd.read_csv(ppi_path, dtype=str, low_memory=False)
    required = {"source_genesymbol", "target_genesymbol"}
    if not required.issubset(set(df.columns)):
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
        sign = _sign_from_row(row)
        if bool(row["consensus_direction"]) or bool(row["is_directed"]):
            directed_edges.append((src, tgt, sign))
        else:
            undirected_edges.append((src, tgt, 1.0))
    return directed_edges, undirected_edges


def build_tf_regulon_modules(tf_edges, min_targets=5, max_targets=200):
    tf2t = {}
    tf2sign = {}
    for s, t, w in tf_edges:
        tf2t.setdefault(s, set()).add(t)
        tf2sign.setdefault(s, {}).setdefault(t, w)
    modules = []
    for tf, targets in tf2t.items():
        n = len(targets)
        if n < int(min_targets) or n > int(max_targets):
            continue
        modules.append({"name": f"TF:{tf}", "genes": set(targets)})
    modules.sort(key=lambda x: len(x["genes"]), reverse=True)
    return modules


def select_modules_greedy(modules, target_genes, k=200, min_cover_frac=0.9, max_overlap_jaccard=0.9):
    target_genes = list(target_genes)
    target_set = set(target_genes)
    covered = set()
    selected = []
    for m in modules:
        if len(selected) >= int(k):
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
        if len(covered) / max(1, len(target_set)) >= float(min_cover_frac):
            if len(selected) >= min(50, int(k)):
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
    M = sp.csr_matrix((data, (rows, cols)), shape=(G, K), dtype=np.float32)
    return M


def module_pool_target_signal(M_gene_module, s_gene, method="softmax", tau=5.0):
    s_gene = np.asarray(s_gene, dtype=np.float32).reshape(-1)
    if method == "max":
        X = M_gene_module.multiply(s_gene[:, None])
        s_mod = np.asarray(X.max(axis=0)).reshape(-1)
        return s_mod
    if method == "mean":
        sums = (M_gene_module.T @ s_gene).A.reshape(-1)
        cnt = np.asarray(M_gene_module.sum(axis=0)).reshape(-1)
        return sums / np.maximum(cnt, 1.0)
    if method == "softmax":
        X = M_gene_module.multiply(s_gene[:, None])
        raw = np.asarray(X.max(axis=0)).reshape(-1)
        w = np.exp(float(tau) * raw)
        return w / (np.sum(w) + 1e-9)
    raise ValueError(f"unknown method: {method}")


def coarsen_edges_to_modules(
    gene2idx,
    modules,
    edges_directed_signed,
    edges_undirected,
    distribute="uniform",
    keep_top_per_src=0,
):
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

    def _pairs(u_mods, v_mods):
        for uu in u_mods:
            for vv in v_mods:
                yield uu, vv

    for s, t, sign in edges_directed_signed:
        si = gene2idx.get(s)
        ti = gene2idx.get(t)
        if si is None or ti is None:
            continue
        smods = idx2mods.get(si)
        tmods = idx2mods.get(ti)
        if not smods or not tmods:
            continue
        if distribute == "uniform":
            w = 1.0 / (len(smods) * len(tmods))
        else:
            w = 1.0
        if float(sign) >= 0:
            for u, v in _pairs(smods, tmods):
                A_pos[u, v] += w
        else:
            for u, v in _pairs(smods, tmods):
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
        if distribute == "uniform":
            w = 1.0 / (len(smods) * len(tmods))
        else:
            w = 1.0
        for u, v in _pairs(smods, tmods):
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


def save_module_graph(out_path, node_list, modules, M_gene_module, A_pos, A_neg, A_undir):
    out = {
        "node_list": np.asarray(node_list, dtype=object),
        "module_names": np.asarray([m["name"] for m in modules], dtype=object),
        "M_gene_module": M_gene_module.tocsr(),
        "A_pos": sp.csr_matrix(A_pos),
        "A_neg": sp.csr_matrix(A_neg),
        "A_undir": sp.csr_matrix(A_undir),
    }
    sp.save_npz(out_path.replace(".npz", ".M_gene_module.npz"), out["M_gene_module"])
    sp.save_npz(out_path.replace(".npz", ".A_pos.npz"), out["A_pos"])
    sp.save_npz(out_path.replace(".npz", ".A_neg.npz"), out["A_neg"])
    sp.save_npz(out_path.replace(".npz", ".A_undir.npz"), out["A_undir"])
    np.savez_compressed(
        out_path,
        node_list=out["node_list"],
        module_names=out["module_names"],
    )


def print_stats(node_list, modules, A_pos, A_neg, A_undir):
    G = len(node_list)
    K = len(modules)
    sizes = np.array([len(m["genes"]) for m in modules], dtype=np.int32)
    covered = set()
    for m in modules:
        covered |= set(m["genes"])
    cover_frac = len(covered) / max(1, G)
    def _edge_count(A):
        return int(np.count_nonzero(A))
    print(f"Genes: {G} | Modules: {K}")
    if K > 0:
        print(f"Module size: min={int(sizes.min())} median={float(np.median(sizes)):.1f} max={int(sizes.max())}")
    print(f"Coverage: {len(covered)}/{G} ({cover_frac:.3f})")
    print(f"Module edges (pos/neg/undir): {_edge_count(A_pos)}/{_edge_count(A_neg)}/{_edge_count(A_undir)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    p.add_argument("--use_landmark", action="store_true", default=True)
    p.add_argument("--k_modules", type=int, default=200)
    p.add_argument("--min_module_targets", type=int, default=5)
    p.add_argument("--max_module_targets", type=int, default=200)
    p.add_argument("--min_cover_frac", type=float, default=0.9)
    p.add_argument("--max_overlap_jaccard", type=float, default=0.9)
    p.add_argument("--consensus_only", action="store_true", default=False)
    p.add_argument("--is_directed_only", action="store_true", default=False)
    p.add_argument("--keep_top_per_src", type=int, default=0)
    p.add_argument("--out", default="data/module_graph_out.npz")
    args = p.parse_args()

    root = args.root
    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    landmark_json = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")

    symbol_to_entrez = load_symbol_to_entrez(full_gene_path)
    target_genes = load_target_genes(landmark_json, bool(args.use_landmark), full_gene_path)
    node_list = list(target_genes)
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
    edges_directed_signed = tf_edges + ppi_dir
    edges_undirected = [] if (bool(args.consensus_only) or bool(args.is_directed_only)) else ppi_undir

    base_modules = build_tf_regulon_modules(
        tf_edges,
        min_targets=int(args.min_module_targets),
        max_targets=int(args.max_module_targets),
    )
    modules = select_modules_greedy(
        base_modules,
        node_list,
        k=int(args.k_modules),
        min_cover_frac=float(args.min_cover_frac),
        max_overlap_jaccard=float(args.max_overlap_jaccard),
    )
    M_gene_module = build_gene_module_matrix(gene2idx, modules)
    A_pos, A_neg, A_undir = coarsen_edges_to_modules(
        gene2idx,
        modules,
        edges_directed_signed=edges_directed_signed,
        edges_undirected=edges_undirected,
        distribute="uniform",
        keep_top_per_src=int(args.keep_top_per_src),
    )
    print_stats(node_list, modules, A_pos, A_neg, A_undir)
    out_path = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_module_graph(out_path, node_list, modules, M_gene_module, A_pos, A_neg, A_undir)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

