import os
import numpy as np
import pandas as pd


def build_relational_omnipath_graph(
    target_genes,
    symbol_to_entrez,
    tf_path,
    ppi_path,
    mirna_path=None,
    omnipath_consensus_only=False,
    omnipath_is_directed_only=False,
    include_ppi_undirected=True,
):
    def _norm_symbol(s):
        return str(s).strip().upper()

    def _norm_entrez(x):
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    target_entrez = [_norm_entrez(x) for x in target_genes]
    target_entrez = [x for x in target_entrez if x]
    target_entrez = list(dict.fromkeys(target_entrez))
    target_set = set(target_entrez)

    if not symbol_to_entrez:
        raise RuntimeError("symbol_to_entrez is required")

    def _map_pair(df, src_col, tgt_col):
        src_sym = df[src_col].astype(str).map(_norm_symbol)
        tgt_sym = df[tgt_col].astype(str).map(_norm_symbol)
        src_entrez = src_sym.map(symbol_to_entrez)
        tgt_entrez = tgt_sym.map(symbol_to_entrez)
        valid = src_entrez.notna() & tgt_entrez.notna()
        return src_entrez, tgt_entrez, valid

    rel_edges = {
        "TF_ACT": [],
        "TF_INH": [],
        "PPI_DIR_ACT": [],
        "PPI_DIR_INH": [],
        "PPI_UNDIR": [],
        "MIRNA_INH": [],
    }

    def _sign_from_consensus(df, idx):
        stim = df.get("consensus_stimulation")
        inhib = df.get("consensus_inhibition")
        if stim is not None and inhib is not None:
            s = bool(stim.iloc[idx])
            h = bool(inhib.iloc[idx])
            if h and not s:
                return -1.0
            if s and not h:
                return 1.0
        return 1.0

    def _dir_from_flags(df, idx):
        if bool(omnipath_consensus_only):
            return bool(df["consensus_direction"].iloc[idx])
        if bool(omnipath_is_directed_only):
            return bool(df["is_directed"].iloc[idx])
        cd = df.get("consensus_direction")
        if cd is not None and bool(cd.iloc[idx]):
            return True
        di = df.get("is_directed")
        if di is not None and bool(di.iloc[idx]):
            return True
        return False

    if tf_path and os.path.exists(tf_path):
        df_tf = pd.read_csv(tf_path)
        if "consensus_direction" not in df_tf.columns:
            df_tf["consensus_direction"] = False
        if "is_directed" not in df_tf.columns:
            df_tf["is_directed"] = True
        df_tf["consensus_stimulation"] = df_tf.get("consensus_stimulation", False).fillna(False).astype(bool)
        df_tf["consensus_inhibition"] = df_tf.get("consensus_inhibition", False).fillna(False).astype(bool)
        src_entrez, tgt_entrez, valid = _map_pair(df_tf, "source_genesymbol", "target_genesymbol")
        idxs = np.where(valid.values)[0]
        for i in idxs:
            s = str(src_entrez.iloc[i])
            t = str(tgt_entrez.iloc[i])
            if s not in target_set or t not in target_set:
                continue
            if bool(omnipath_consensus_only) and not bool(df_tf["consensus_direction"].iloc[i]):
                continue
            sign = _sign_from_consensus(df_tf, i)
            if sign > 0:
                rel_edges["TF_ACT"].append((s, t, 1.0))
            else:
                rel_edges["TF_INH"].append((s, t, 1.0))

    if ppi_path and os.path.exists(ppi_path):
        df_ppi = pd.read_csv(ppi_path)
        if "consensus_direction" not in df_ppi.columns:
            df_ppi["consensus_direction"] = False
        if "is_directed" not in df_ppi.columns:
            df_ppi["is_directed"] = False
        df_ppi["consensus_stimulation"] = df_ppi.get("consensus_stimulation", False).fillna(False).astype(bool)
        df_ppi["consensus_inhibition"] = df_ppi.get("consensus_inhibition", False).fillna(False).astype(bool)
        df_ppi["consensus_direction"] = df_ppi["consensus_direction"].fillna(False).astype(bool)
        df_ppi["is_directed"] = df_ppi["is_directed"].fillna(False).astype(bool)
        src_entrez, tgt_entrez, valid = _map_pair(df_ppi, "source_genesymbol", "target_genesymbol")
        idxs = np.where(valid.values)[0]
        for i in idxs:
            s = str(src_entrez.iloc[i])
            t = str(tgt_entrez.iloc[i])
            if s not in target_set or t not in target_set:
                continue
            is_dir = _dir_from_flags(df_ppi, i)
            sign = _sign_from_consensus(df_ppi, i)
            if is_dir:
                if sign > 0:
                    rel_edges["PPI_DIR_ACT"].append((s, t, 1.0))
                else:
                    rel_edges["PPI_DIR_INH"].append((s, t, 1.0))
            elif bool(include_ppi_undirected) and (not bool(omnipath_consensus_only)) and (not bool(omnipath_is_directed_only)):
                rel_edges["PPI_UNDIR"].append((s, t, 1.0))

    if mirna_path and os.path.exists(mirna_path):
        df_mi = pd.read_csv(mirna_path, usecols=["source_genesymbol", "target_genesymbol", "consensus_direction", "is_directed"], low_memory=False)
        df_mi["consensus_direction"] = df_mi.get("consensus_direction", False).fillna(False).astype(bool)
        df_mi["is_directed"] = df_mi.get("is_directed", True).fillna(False).astype(bool)
        src_entrez, tgt_entrez, valid = _map_pair(df_mi, "source_genesymbol", "target_genesymbol")
        idxs = np.where(valid.values)[0]
        for i in idxs:
            s = str(src_entrez.iloc[i])
            t = str(tgt_entrez.iloc[i])
            if s not in target_set or t not in target_set:
                continue
            is_dir = _dir_from_flags(df_mi, i)
            if not is_dir and (bool(omnipath_consensus_only) or bool(omnipath_is_directed_only)):
                continue
            rel_edges["MIRNA_INH"].append((s, t, 1.0))

    gene2idx = {g: i for i, g in enumerate(target_entrez)}
    rel_index = {}
    for rel, edges in rel_edges.items():
        if not edges:
            continue
        src_idx = []
        dst_idx = []
        w = []
        if rel == "PPI_UNDIR":
            for s, t, ww in edges:
                if s in gene2idx and t in gene2idx and s != t:
                    a = gene2idx[s]
                    b = gene2idx[t]
                    src_idx.extend([a, b])
                    dst_idx.extend([b, a])
                    w.extend([ww, ww])
        else:
            for s, t, ww in edges:
                if s in gene2idx and t in gene2idx:
                    src_idx.append(gene2idx[s])
                    dst_idx.append(gene2idx[t])
                    w.append(ww)
        if src_idx:
            rel_index[rel] = (np.asarray([src_idx, dst_idx], dtype=np.int32), np.asarray(w, dtype=np.float32))

    return {"node_list": target_entrez, "gene2idx": gene2idx, "relations": rel_index}
