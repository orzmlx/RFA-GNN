import heapq
import pandas as pd
import numpy as np
import os
import json
import h5py
from train_common import build_disjoint_target_split_masks


def build_combined_gnn(
    tf_path="data/omnipath/omnipath_tf_regulons.csv",
    ppi_path="data/omnipath/omnipath_interactions.csv",
    target_genes=None,
    #full_gene_path="data/GSE92742_Broad_LINCS_gene_info.txt",
    #string_path = "data/string_interactions_mapped.csv",
    # confid_threshold= 0.9,
    # directed=True,
    # omnipath_consensus_only=False,
    omnipath_is_directed_only=False,
    symbol_to_entrez=None,
    return_edge_weight=False,

):
    """
    合并 TF 调控网络和 PPI 互作网络，构建更密集的 GNN 图。
    
    Args:
        directed (bool):
            True: 将无向互作边(PPI/STRING)拆成双向边 (u->v, v->u)
            False: 将无向互作边(PPI/STRING)按无向去重 (u,v 与 v,u 视为同一条边)
        omnipath_is_directed_only (bool):
            True: 仅使用 OmniPath 中 is_directed=True 的边（无向互作边直接丢弃）。
    """
    print(">>> 正在构建 Combined GNN (TF + PPI) ...")
    if bool(omnipath_is_directed_only):
        print("OmniPath: 仅保留 is_directed=True 的边")
    
    # 1. 加载基因 ID 映射 (Symbol -> Entrez)
    def _norm_symbol(s):
        return str(s).strip().upper()

    def _norm_entrez(x):
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s
     

    if not symbol_to_entrez or symbol_to_entrez is None:
        raise RuntimeError("未能构建 symbol_to_entrez 映射：请检查 full_gene_path / landmark_path 文件。")

    if target_genes is None or len(target_genes) == 0:
        raise Exception("target_genes 不能为空")

    target_entrez = [_norm_entrez(x) for x in target_genes]
    target_entrez = [x for x in target_entrez if x]
    target_entrez = list(dict.fromkeys(target_entrez))

        
    target_genes_set = set(target_entrez)


    def process_omnipath_signed(
        df,
        src_symbol_col,
        tgt_symbol_col,
        consensus_direction_col="consensus_direction",
        consensus_stim_col="consensus_stimulation",
        consensus_inhib_col="consensus_inhibition",
        is_directed_col="is_directed",
        default_weight=1.0,
        directed_only=False,
        omnipath_consensus_only=False,
        omnipath_is_directed_only=False,
        require_sign_cols=False,
    ):
        if src_symbol_col not in df.columns or tgt_symbol_col not in df.columns:
            return [], []

        src_sym = df[src_symbol_col].astype(str).map(_norm_symbol)
        tgt_sym = df[tgt_symbol_col].astype(str).map(_norm_symbol)
        src_entrez = src_sym.map(symbol_to_entrez)
        tgt_entrez = tgt_sym.map(symbol_to_entrez)
        valid = src_entrez.notna() & tgt_entrez.notna()
        if not bool(valid.any()):
            raise Exception("OmniPath 数据中源或目标基因未能映射到 Entrez ID。")
        is_directed_mask = df[is_directed_col].fillna(False).astype(bool) if is_directed_col in df.columns else None

        if bool(omnipath_is_directed_only) and is_directed_mask is None:
            raise Exception("omnipath_is_directed_only=True 但缺少 is_directed 列")

        stim = None
        inhib = None
        if str(consensus_stim_col) in df.columns:
            stim = df[str(consensus_stim_col)].fillna(False).astype(bool)
        if str(consensus_inhib_col) in df.columns:
            inhib = df[str(consensus_inhib_col)].fillna(False).astype(bool)
        consensus_stim = df["consensus_stimulation"].fillna(False).astype(bool) if "consensus_stimulation" in df.columns else None
        consensus_inhib = df["consensus_inhibition"].fillna(False).astype(bool) if "consensus_inhibition" in df.columns else None
        if bool(require_sign_cols) and (stim is None or inhib is None):
            raise Exception(f"缺少 sign 列: stim={consensus_stim_col}, inhib={consensus_inhib_col}")
        directed_edges = []
        undirected_edges = []
        for idx in np.where(valid.values)[0]:
            # s_symbol = str(src_sym.iloc[idx])
            # t_symbol = str(tgt_sym.iloc[idx])
            s = str(src_entrez.iloc[idx])
            t = str(tgt_entrez.iloc[idx])
            
            if s not in target_genes_set or t not in target_genes_set:
                continue
            if bool(omnipath_is_directed_only) and is_directed_mask is not None and not bool(is_directed_mask.iloc[idx]):
                continue
            sign = 1.0
            stim_i = bool(stim.iloc[idx]) if stim is not None else False
            inhib_i = bool(inhib.iloc[idx]) if inhib is not None else False
            if consensus_stim is not None and consensus_inhib is not None:
                cs = bool(consensus_stim.iloc[idx])
                ci = bool(consensus_inhib.iloc[idx])
                if (stim_i and inhib_i) or ((not stim_i) and (not inhib_i)):
                    stim_i = cs
                    inhib_i = ci
            if inhib_i and not stim_i:
                sign = -1.0
            elif stim_i and not inhib_i:
                sign = 1.0
            else: # 各种数据库对方向没有共识，直接丢弃，防止引入噪音，或者有共识，但是共识也冲突
                continue
            w = float(default_weight) * float(sign)

            if bool(omnipath_is_directed_only):
                is_dir = bool(is_directed_mask.iloc[idx]) if is_directed_mask is not None else False
            else:
                is_dir = False
                if is_directed_mask is not None and bool(is_directed_mask.iloc[idx]):
                    is_dir = True

            if is_dir:
                directed_edges.append((s, t, w))
            elif not bool(directed_only):
                undirected_edges.append((s, t, abs(w)))
        print(f"OmniPath 有向边数: {len(directed_edges)}", f"OmniPath 无向边数: {len(undirected_edges)}")   
        return directed_edges, undirected_edges

    edges_directed = []
    edges_undirected = []
    
    # 2. 加载 TF Regulons
    if tf_path is not None and os.path.exists(tf_path):
        print(f"加载 TF 调控网络: {tf_path}")
        df_tf = pd.read_csv(tf_path)
        edges_tf_dir, _ = process_omnipath_signed(
            df_tf,
            "source_genesymbol",
            "target_genesymbol",
            directed_only=True,
            omnipath_is_directed_only=bool(omnipath_is_directed_only),
            consensus_stim_col="is_stimulation",
            consensus_inhib_col="is_inhibition",
            require_sign_cols=True,
        )
        edges_tf = edges_tf_dir
        print(f"TF 边数: {len(edges_tf)}")
        edges_tf_debug = pd.DataFrame(edges_tf, columns=["src_entrez", "dst_entrez", "sign"]).astype({"src_entrez": str, "dst_entrez": str})
        inv_symbol_to_entrez = {str(v): str(k) for k, v in symbol_to_entrez.items()}
        edges_tf_debug["src_symbol"] = edges_tf_debug["src_entrez"].map(inv_symbol_to_entrez).fillna(edges_tf_debug["src_entrez"])
        edges_tf_debug["dst_symbol"] = edges_tf_debug["dst_entrez"].map(inv_symbol_to_entrez).fillna(edges_tf_debug["dst_entrez"])
        edges_directed.extend(edges_tf)

    # 3. 加载 PPI
    if ppi_path is not None and os.path.exists(ppi_path):
        print(f"加载 PPI 网络: {ppi_path}")
        df_ppi = pd.read_csv(ppi_path)
        edges_ppi_dir, edges_ppi_undir = process_omnipath_signed(
            df_ppi,
            "source_genesymbol",
            "target_genesymbol",
            directed_only=False,
            omnipath_is_directed_only=bool(omnipath_is_directed_only),
            consensus_stim_col="consensus_stimulation",
            consensus_inhib_col="consensus_inhibition",
            require_sign_cols=True,
        )
        edges_directed.extend(edges_ppi_dir)
        edges_undirected.extend(edges_ppi_undir)
        
        edges_ppi_debug = pd.DataFrame(edges_ppi_dir, columns=["src_entrez", "dst_entrez", "sign"]).astype({"src_entrez": str, "dst_entrez": str})
        inv_symbol_to_entrez = {str(v): str(k) for k, v in symbol_to_entrez.items()}
        edges_ppi_debug["src_symbol"] = edges_ppi_debug["src_entrez"].map(inv_symbol_to_entrez).fillna(edges_ppi_debug["src_entrez"])
        edges_ppi_debug["dst_symbol"] = edges_ppi_debug["dst_entrez"].map(inv_symbol_to_entrez).fillna(edges_ppi_debug["dst_entrez"])
        print(f"PPI 边数: {len(edges_ppi_dir) + len(edges_ppi_undir)}")

    directed_map = {}
    for s, t, w in edges_directed:
        key = (str(s), str(t))
        prev = directed_map.get(key)
        if prev is None:
            directed_map[key] = w
        else:
            directed_map[key] = w if abs(w) > abs(prev) else prev
    edges_directed = [(s, t, w) for (s, t), w in directed_map.items()]

    node_list = target_entrez
        
    gene2idx = {g: i for i, g in enumerate(node_list)}
    N = len(node_list)

    if bool(return_edge_weight):
        sparse_edge_map = {}

        for u, v, w in edges_directed:
            if u in gene2idx and v in gene2idx:
                i, j = gene2idx[u], gene2idx[v]
                key = (i, j)
                prev = sparse_edge_map.get(key)
                if prev is None or abs(w) > abs(prev):
                    sparse_edge_map[key] = float(w)

        for u, v, w in edges_undirected:
            if u in gene2idx and v in gene2idx:
                i, j = gene2idx[u], gene2idx[v]
                for key in ((i, j), (j, i)):
                    prev = sparse_edge_map.get(key)
                    if prev is None:
                        sparse_edge_map[key] = float(w)
                    else:
                        sparse_edge_map[key] = float(max(prev, w))

        edge_items = sorted(sparse_edge_map.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        if edge_items:
            src = np.asarray([ij[0] for ij, _ in edge_items], dtype=np.int32)
            dst = np.asarray([ij[1] for ij, _ in edge_items], dtype=np.int32)
            edge_index = np.stack([src, dst], axis=0)
            edge_weight = np.asarray([w for _, w in edge_items], dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int32)
            edge_weight = np.zeros((0,), dtype=np.float32)

        print(f"Combined Graph 构建完成: {N} 节点, {edge_index.shape[1]} 稀疏边 (不含 self-loop)")
        return None, node_list, gene2idx, edge_index, edge_weight

    adj_matrix = np.zeros((N, N), dtype=np.float32)
    edge_index = [[], []]
    count = 0

    for u, v, w in edges_directed:
        if u in gene2idx and v in gene2idx:
            i, j = gene2idx[u], gene2idx[v]
            if abs(w) > abs(adj_matrix[j, i]):
                adj_matrix[j, i] = w
            edge_index[0].append(i)
            edge_index[1].append(j)
            count += 1

    for u, v, w in edges_undirected:
        if u in gene2idx and v in gene2idx:
            i, j = gene2idx[u], gene2idx[v]
            adj_matrix[i, j] = max(adj_matrix[i, j], w)
            adj_matrix[j, i] = max(adj_matrix[j, i], w)
            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_index[0].append(j)
            edge_index[1].append(i)
            count += 1

    # 5. Add Self-Loops (Weight=1.0)
    np.fill_diagonal(adj_matrix, 1.0)

    print(f"Combined Graph 构建完成: {N} 节点, {count} 边 (含权重)")
    return adj_matrix, node_list, gene2idx, np.array(edge_index)


def create_pair_split_masks(split_mode, drug_ids, cell_names, test_frac, seed=42, drug_target_matrix=None):
    """Create train/test masks for paired treatment anchors under different split modes."""
    split_mode = str(split_mode).strip()
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")
    rng = np.random.default_rng(int(seed))
    n = len(drug_ids)
    if split_mode == "warm":
        if n < 2:
            raise RuntimeError("warm 需要至少 2 个样本")
        n_test = max(1, int(n * test_frac))
        n_test = min(n_test, n - 1)
        held = rng.choice(np.arange(n), size=n_test, replace=False)
        test_mask = np.zeros((n,), dtype=bool)
        test_mask[held] = True
        train_mask = ~test_mask
        print(f"Split=warm | Held-out trt samples: {int(np.sum(test_mask))}/{n}")
        return train_mask, test_mask
    if split_mode == "cold_cell":
        unique_cells = np.unique(cell_names)
        if len(unique_cells) < 2:
            raise RuntimeError("cold_cell 需要至少 2 个细胞系")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        held = rng.choice(unique_cells, size=n_test, replace=False)
        test_mask = np.isin(cell_names, held)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(held)}/{len(unique_cells)}")
        return train_mask, test_mask
    if split_mode == "cold_drug":
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise RuntimeError("cold_drug 需要至少 2 个药物")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        held = rng.choice(unique_drugs, size=n_test, replace=False)
        test_mask = np.isin(drug_ids, held)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(held)}/{len(unique_drugs)}")
        return train_mask, test_mask
    if split_mode == "cold_target_pattern":
        if drug_target_matrix is None:
            raise RuntimeError("cold_target_pattern 需要提供 drug_target_matrix")
        train_mask, test_mask, stats = build_disjoint_target_split_masks(
            drug_target_matrix,
            test_frac,
            seed=seed,
        )
        print(
            "Split=cold_target_pattern | "
            f"Held-out overlap components: {stats['held_out_components']}/{stats['total_components']} | "
            f"train_n={stats['train_samples']} test_n={stats['test_samples']} | "
            f"pattern_overlap={stats['pattern_overlap_count']} "
            f"target_overlap={stats['target_overlap_count']}"
        )
        return train_mask, test_mask
    raise ValueError(f"未知 split_mode: {split_mode}")


def build_pair_split_masks(split_mode, drug_ids, cell_names, test_frac, seed=42, drug_target_matrix=None):
    """Backward-compatible alias for `create_pair_split_masks()`."""
    return create_pair_split_masks(
        split_mode,
        drug_ids,
        cell_names,
        test_frac,
        seed=seed,
        drug_target_matrix=drug_target_matrix,
    )


def slice_anchor_data(anchor_data, indices):
    """Slice anchor-aligned arrays/lists while preserving the rest of the data dict."""
    idx = np.asarray(indices)
    sliced_data = dict(anchor_data)
    array_keys = [
        "anchor_X_trt",
        "anchor_X_drug",
        "anchor_X_fingerprint",
        "anchor_drug_fp_idx",
        "anchor_drug_has_target",
        "anchor_drug_has_fp",
        "anchor_drug_ids",
        "anchor_trt_distil_ids",
        "anchor_cell_names",
        "anchor_batch_ids",
        "anchor_det_plate_ids",
    ]
    for key in array_keys:
        if key in sliced_data and sliced_data[key] is not None:
            sliced_data[key] = np.asarray(sliced_data[key])[idx]
    if "anchor_ctl_candidates" in sliced_data and sliced_data["anchor_ctl_candidates"] is not None:
        control_candidates = sliced_data["anchor_ctl_candidates"]
        sliced_data["anchor_ctl_candidates"] = [list(control_candidates[int(i)]) for i in idx.tolist()]
    return sliced_data


def subset_anchor_data(data, indices):
    """Backward-compatible alias for `slice_anchor_data()`."""
    return slice_anchor_data(data, indices)


def split_control_pools(train_candidates, test_candidates, seed=42):
    """Split shared control candidates into disjoint train/test control pools."""
    train_candidates = [[str(c) for c in cands] for cands in train_candidates]
    test_candidates = [[str(c) for c in cands] for cands in test_candidates]
    train_used = set()
    test_used = set()
    for cands in train_candidates:
        train_used.update(cands)
    for cands in test_candidates:
        test_used.update(cands)

    train_allowed = set(train_used - test_used)
    test_allowed = set(test_used - train_used)
    shared = set(train_used & test_used)
    rng = np.random.default_rng(int(seed))

    def _collect_missing_controls(side_candidates, allowed_controls):
        unmet = set()
        anchor_to_shared = {}
        shared_to_anchors = {}
        for anchor_idx, candidates in enumerate(side_candidates):
            has_unique = any(candidate in allowed_controls for candidate in candidates)
            shared_list = [candidate for candidate in candidates if candidate in shared]
            anchor_to_shared[anchor_idx] = shared_list
            if not has_unique:
                unmet.add(anchor_idx)
                for control_id in shared_list:
                    shared_to_anchors.setdefault(control_id, []).append(anchor_idx)
        return unmet, anchor_to_shared, shared_to_anchors

    train_unmet, train_anchor_to_shared, train_shared_to_anchors = _collect_missing_controls(train_candidates, train_allowed)
    test_unmet, test_anchor_to_shared, test_shared_to_anchors = _collect_missing_controls(test_candidates, test_allowed)

    train_cover = {ctl_id: len(anchors) for ctl_id, anchors in train_shared_to_anchors.items()}
    test_cover = {ctl_id: len(anchors) for ctl_id, anchors in test_shared_to_anchors.items()}
    assigned_side = {}
    heap = []
    for ctl_id in shared:
        score = max(train_cover.get(ctl_id, 0), test_cover.get(ctl_id, 0))
        heapq.heappush(heap, (-score, ctl_id))

    def _mark_anchor_done(side, anchor_idx):
        if side == "train":
            if anchor_idx not in train_unmet:
                return
            train_unmet.remove(anchor_idx)
            for other_ctl in train_anchor_to_shared.get(anchor_idx, []):
                if assigned_side.get(other_ctl) is None:
                    train_cover[other_ctl] = max(0, int(train_cover.get(other_ctl, 0)) - 1)
                    heapq.heappush(heap, (-max(train_cover.get(other_ctl, 0), test_cover.get(other_ctl, 0)), other_ctl))
            return
        if anchor_idx not in test_unmet:
            return
        test_unmet.remove(anchor_idx)
        for other_ctl in test_anchor_to_shared.get(anchor_idx, []):
            if assigned_side.get(other_ctl) is None:
                test_cover[other_ctl] = max(0, int(test_cover.get(other_ctl, 0)) - 1)
                heapq.heappush(heap, (-max(train_cover.get(other_ctl, 0), test_cover.get(other_ctl, 0)), other_ctl))

    while len(heap) > 0:
        neg_score, ctl_id = heapq.heappop(heap)
        if ctl_id in assigned_side:
            continue
        gain_train = int(train_cover.get(ctl_id, 0))
        gain_test = int(test_cover.get(ctl_id, 0))
        current_score = max(gain_train, gain_test)
        if -neg_score != current_score:
            heapq.heappush(heap, (-current_score, ctl_id))
            continue
        if current_score <= 0:
            break
        if gain_train > gain_test:
            side = "train"
        elif gain_test > gain_train:
            side = "test"
        elif len(train_allowed) < len(test_allowed):
            side = "train"
        elif len(test_allowed) < len(train_allowed):
            side = "test"
        else:
            side = "train" if rng.random() < 0.5 else "test"
        assigned_side[ctl_id] = side
        if side == "train":
            train_allowed.add(ctl_id)
            for anchor_idx in train_shared_to_anchors.get(ctl_id, []):
                _mark_anchor_done("train", anchor_idx)
        else:
            test_allowed.add(ctl_id)
            for anchor_idx in test_shared_to_anchors.get(ctl_id, []):
                _mark_anchor_done("test", anchor_idx)

    remaining = [ctl_id for ctl_id in shared if ctl_id not in assigned_side]
    rng.shuffle(remaining)
    for ctl_id in remaining:
        if len(train_allowed) <= len(test_allowed):
            train_allowed.add(ctl_id)
        else:
            test_allowed.add(ctl_id)

    train_filtered = [[c for c in cands if c in train_allowed] for cands in train_candidates]
    test_filtered = [[c for c in cands if c in test_allowed] for cands in test_candidates]
    train_empty = sum(1 for cands in train_filtered if len(cands) == 0)
    test_empty = sum(1 for cands in test_filtered if len(cands) == 0)
    stats = {
        "train_ctl_ids": len(train_allowed),
        "test_ctl_ids": len(test_allowed),
        "shared_ctl_ids": len(train_used & test_used),
        "train_empty_anchors": int(train_empty),
        "test_empty_anchors": int(test_empty),
    }
    return train_allowed, test_allowed, train_filtered, test_filtered, stats



def filter_control_pool(dataset, allowed_control_ids):
    """Filter the cached control pool down to a selected subset of control ids."""
    ctl_pool_ids = np.asarray(dataset["ctl_pool_ids"], dtype=object)
    ctl_pool_expr = np.asarray(dataset["ctl_pool_expr"], dtype=np.float32)
    allowed_control_ids = {str(control_id) for control_id in allowed_control_ids}
    keep_idx = [i for i, cid in enumerate(ctl_pool_ids.tolist()) if str(cid) in allowed_control_ids]
    if len(keep_idx) == 0:
        num_genes = int(ctl_pool_expr.shape[1]) if ctl_pool_expr.ndim == 2 and ctl_pool_expr.shape[1] > 0 else int(dataset["input_dim"])
        return np.asarray([], dtype=object), np.zeros((0, num_genes), dtype=np.float32)
    keep_idx = np.asarray(keep_idx, dtype=np.int32)
    return ctl_pool_ids[keep_idx], ctl_pool_expr[keep_idx]


def assemble_paired_dataset(anchor_data, pairing_mode="multi_trt_multi_ctl", ctl_residual_pool_size=3, seed=42):
    """Sample control pairs and assemble the concrete train/eval arrays."""
    valid_pairing_modes = {"multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"}
    if pairing_mode not in valid_pairing_modes:
        raise ValueError(f"未知 pairing_mode: {pairing_mode}")

    anchor_X_trt = np.asarray(anchor_data["anchor_X_trt"], dtype=np.float32)
    anchor_X_drug = np.asarray(anchor_data["anchor_X_drug"], dtype=np.float32)
    anchor_drug_ids = np.asarray(anchor_data["anchor_drug_ids"], dtype=str)
    anchor_trt_distil_ids = np.asarray(anchor_data["anchor_trt_distil_ids"], dtype=str)
    anchor_cell_names = np.asarray(anchor_data["anchor_cell_names"], dtype=str)
    anchor_batch_ids = np.asarray(anchor_data["anchor_batch_ids"], dtype=str)
    anchor_det_plate_ids = np.asarray(anchor_data["anchor_det_plate_ids"], dtype=str)
    anchor_drug_fp_idx = np.asarray(anchor_data["anchor_drug_fp_idx"], dtype=np.int32)
    anchor_drug_has_target = np.asarray(anchor_data["anchor_drug_has_target"], dtype=np.float32)
    anchor_drug_has_fp = np.asarray(anchor_data["anchor_drug_has_fp"], dtype=np.float32)
    anchor_ctl_candidates = anchor_data["anchor_ctl_candidates"]
    ctl_pool_ids = np.asarray(anchor_data["ctl_pool_ids"], dtype=str)
    ctl_pool_expr = np.asarray(anchor_data["ctl_pool_expr"], dtype=np.float32)
    ctl_id_to_idx = {str(cid): i for i, cid in enumerate(ctl_pool_ids.tolist())}
    drug_fp_table = anchor_data.get("drug_fp_table")
    drug_fp_table = None if drug_fp_table is None else np.asarray(drug_fp_table, dtype=np.float32)
    anchor_X_fingerprint = anchor_data.get("anchor_X_fingerprint")
    anchor_X_fingerprint = None if anchor_X_fingerprint is None else np.asarray(anchor_X_fingerprint, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    n_ctl_per_trt = int(ctl_residual_pool_size) if int(ctl_residual_pool_size) > 0 else 3

    pair_anchor_idx = []
    pair_ctl_ids = []

    if pairing_mode == "multi_trt_multi_ctl":
        for i, ctl_candidates in enumerate(anchor_ctl_candidates):
            if len(ctl_candidates) == 0:
                continue
            replace = len(ctl_candidates) < n_ctl_per_trt
            sampled_ctl = rng.choice(np.asarray(ctl_candidates, dtype=object), size=n_ctl_per_trt, replace=replace)
            for ctl_id in sampled_ctl.tolist():
                pair_anchor_idx.append(i)
                pair_ctl_ids.append(str(ctl_id))
    elif pairing_mode == "unique_trt_reuse_ctl":
        for i, ctl_candidates in enumerate(anchor_ctl_candidates):
            if len(ctl_candidates) == 0:
                continue
            ctl_id = str(rng.choice(np.asarray(ctl_candidates, dtype=object)))
            pair_anchor_idx.append(i)
            pair_ctl_ids.append(ctl_id)
    else:
        order = list(range(len(anchor_ctl_candidates)))
        rng.shuffle(order)
        order.sort(key=lambda i: len(anchor_ctl_candidates[i]))
        used_ctl_ids = set()
        skipped_due_to_unique_ctl = 0
        for i in order:
            available = [c for c in anchor_ctl_candidates[i] if c not in used_ctl_ids]
            if len(available) == 0:
                skipped_due_to_unique_ctl += 1
                continue
            ctl_id = str(rng.choice(np.asarray(available, dtype=object)))
            used_ctl_ids.add(ctl_id)
            pair_anchor_idx.append(i)
            pair_ctl_ids.append(ctl_id)
        print(f"unique_trt_unique_ctl: 因 ctl 不可重复而跳过 {skipped_due_to_unique_ctl} 个 trt")

    if len(pair_anchor_idx) == 0:
        num_genes = int(anchor_X_trt.shape[1])
        fp_width = 0 if drug_fp_table is None else int(drug_fp_table.shape[1])
        return {
            "X_ctl": np.zeros((0, num_genes), dtype=np.float32),
            "y_delta": np.zeros((0, num_genes), dtype=np.float32),
            "X_drug": np.zeros((0, anchor_X_drug.shape[1]), dtype=np.float32),
            "X_fingerprint": np.zeros((0, fp_width), dtype=np.float32) if fp_width > 0 else None,
            "drug_fp_idx": np.zeros((0,), dtype=np.int32),
            "drug_has_target": np.zeros((0,), dtype=np.float32),
            "drug_has_fp": np.zeros((0,), dtype=np.float32),
            "drug_ids": np.asarray([], dtype=object),
            "trt_distil_ids": np.asarray([], dtype=object),
            "ctl_distil_ids": np.asarray([], dtype=object),
            "cell_names": np.asarray([], dtype=object),
            "batch_ids": np.asarray([], dtype=object),
            "det_plate_ids": np.asarray([], dtype=object),
            "pairing_mode": pairing_mode,
        }

    pair_anchor_idx = np.asarray(pair_anchor_idx, dtype=np.int32)
    pair_ctl_ids = np.asarray(pair_ctl_ids, dtype=object)
    ctl_rows = [ctl_id_to_idx[str(cid)] for cid in pair_ctl_ids.tolist()]
    X_ctl = ctl_pool_expr[np.asarray(ctl_rows, dtype=np.int32)]
    X_trt = anchor_X_trt[pair_anchor_idx]
    y_delta = X_trt - X_ctl
    paired = {
        "X_ctl": X_ctl.astype(np.float32),
        "y_delta": y_delta.astype(np.float32),
        "X_drug": anchor_X_drug[pair_anchor_idx].astype(np.float32),
        "X_fingerprint": None,
        "drug_fp_idx": anchor_drug_fp_idx[pair_anchor_idx].astype(np.int32),
        "drug_has_target": anchor_drug_has_target[pair_anchor_idx].astype(np.float32),
        "drug_has_fp": anchor_drug_has_fp[pair_anchor_idx].astype(np.float32),
        "drug_ids": anchor_drug_ids[pair_anchor_idx].astype(object),
        "trt_distil_ids": anchor_trt_distil_ids[pair_anchor_idx].astype(object),
        "ctl_distil_ids": pair_ctl_ids.astype(object),
        "cell_names": anchor_cell_names[pair_anchor_idx].astype(object),
        "batch_ids": anchor_batch_ids[pair_anchor_idx].astype(object),
        "det_plate_ids": anchor_det_plate_ids[pair_anchor_idx].astype(object),
        "pairing_mode": pairing_mode,
    }
    if anchor_X_fingerprint is not None:
        paired["X_fingerprint"] = anchor_X_fingerprint[pair_anchor_idx].astype(np.float32)
    elif drug_fp_table is not None:
        fp_idx = paired["drug_fp_idx"]
        materialize_limit_mb = 512
        need_bytes = int(len(fp_idx)) * int(drug_fp_table.shape[1]) * 4
        if need_bytes <= int(materialize_limit_mb) * 1024 * 1024:
            paired["X_fingerprint"] = drug_fp_table[fp_idx]
    return paired


def prepare_split_data(
    data,
    split_mode,
    test_frac,
    seed=42,
    train_pairing_mode="multi_trt_multi_ctl",
    train_ctl_pair_k=3,
    test_pairing_mode="unique_trt_reuse_ctl",
):
    """Build split-specific paired datasets while isolating train/test control pools."""
    anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
    anchor_cell_names = np.asarray(data["anchor_cell_names"], dtype=str)
    anchor_X_drug = np.asarray(data["anchor_X_drug"], dtype=np.float32)
    train_mask, test_mask = create_pair_split_masks(
        split_mode,
        anchor_drug_ids,
        anchor_cell_names,
        test_frac,
        seed=seed,
        drug_target_matrix=anchor_X_drug,
    )
    train_anchor_data = slice_anchor_data(data, np.where(train_mask)[0])
    test_anchor_data = slice_anchor_data(data, np.where(test_mask)[0])
    train_allowed_ctl, test_allowed_ctl, train_candidates, test_candidates, ctl_stats = split_control_pools(
        train_anchor_data["anchor_ctl_candidates"],
        test_anchor_data["anchor_ctl_candidates"],
        seed=seed,
    )
    train_anchor_data["anchor_ctl_candidates"] = train_candidates
    test_anchor_data["anchor_ctl_candidates"] = test_candidates
    train_anchor_data["ctl_pool_ids"], train_anchor_data["ctl_pool_expr"] = filter_control_pool(data, train_allowed_ctl)
    test_anchor_data["ctl_pool_ids"], test_anchor_data["ctl_pool_expr"] = filter_control_pool(data, test_allowed_ctl)
    print(
        "Isolated ctl pools | "
        f"shared_before={ctl_stats['shared_ctl_ids']} "
        f"train_ctl={ctl_stats['train_ctl_ids']} "
        f"test_ctl={ctl_stats['test_ctl_ids']} "
        f"train_empty={ctl_stats['train_empty_anchors']} "
        f"test_empty={ctl_stats['test_empty_anchors']}"
    )
    train_data = assemble_paired_dataset(
        train_anchor_data,
        pairing_mode=str(train_pairing_mode),
        ctl_residual_pool_size=int(train_ctl_pair_k),
        seed=int(seed),
    )
    test_data = assemble_paired_dataset(
        test_anchor_data,
        pairing_mode=str(test_pairing_mode),
        ctl_residual_pool_size=1,
        seed=int(seed) + 1,
    )
    return train_data, test_data, train_mask, test_mask


def _build_symbol_entrez_map(full_gene_path):
    """从全基因文件生成 Symbol->Entrez 映射"""
    symbol_to_entrez = {}
    try:
        df_genes = pd.read_csv(full_gene_path, sep='\t', dtype=str)
        for _, row in df_genes.iterrows():
            symbol_to_entrez[row['pr_gene_symbol']] = row['pr_gene_id']
    except Exception as e:
        print(f"生成 Symbol->Entrez 映射失败: {e}")
    return symbol_to_entrez

def _load_expression_data(path, target_samples, target_genes):
    """
    加载表达数据，支持 H5 (优先) 和 CSV。
    """
    print(f"正在加载数据文件: {path} ...")
    
    df = None
    # 1. H5 优化读取 (使用 h5py 直接读取索引)
    if path.endswith('.h5'):
        try:
            with h5py.File(path, 'r') as f:
                if 'data' in f and 'block0_values' in f['data']:
                    d = f['data']
                    axis0 = d['axis0'][:].astype(str) # Index
                    axis1 = d['axis1'][:].astype(str) # Columns
                    target_set = set(target_samples)
                    
                    # 匹配 Index (Samples)
                    matches = [i for i, x in enumerate(axis0) if x in target_set]
                    if len(matches) > 0:
                        print(f"  H5 (h5py): 匹配到 {len(matches)} 个样本 (axis0/Index). 读取中...")
                        matches.sort()
                        
                        # block0_values shape (N_col, N_row) -> (Genes, Samples)
                        # 优化: 分批读取
                        batch_size = 5000
                        vals_list = []
                        
                        import time
                        t0 = time.time()
                        
                        for i in range(0, len(matches), batch_size):
                            batch_idx = matches[i:i+batch_size]
                            batch_vals = d['block0_values'][:, batch_idx]
                            vals_list.append(batch_vals)
                            
                        print(f"  读取耗时: {time.time()-t0:.2f}s")
                        vals = np.hstack(vals_list)
                        df = pd.DataFrame(vals.T, index=axis0[matches], columns=axis1)
                    else:
                        print("  H5 (h5py): 未在 Index 中找到样本，将尝试全量读取。")
        except Exception as e:
            print(f"  H5 h5py 读取尝试失败: {e}，转为常规读取。")
    
    # 2. Fallback / 常规读取 (如果分块失败或未执行)
    if df is None:
        try:
            df = pd.read_hdf(path, key='data')
            print(f"  H5 一次性加载成功，原始形状: {df.shape}")
        except Exception as e:
            raise Exception(f"无法读取文件 {path}: {e}")

    # 智能检测方向 & 转置
    # 1. 基于内容匹配 (Robust)
    target_set_genes = set(str(g) for g in target_genes)
    index_overlap = len(set(df.index.astype(str)).intersection(target_set_genes))
    columns_overlap = len(set(df.columns.astype(str)).intersection(target_set_genes))
    
    print(f"  Gene ID 匹配: Index={index_overlap}, Columns={columns_overlap}")
    print(f"  形状检查: Rows={df.shape[0]}, Target={len(target_genes)}")
    
    # 情况 A: 行是基因 (Genes x Samples) -> 需要转置
    need_transpose = False
    
    if index_overlap > columns_overlap and index_overlap > len(target_genes) * 0.1:
        print("  根据 Gene ID 匹配，判定为 (Genes, Samples)，正在转置...")
        need_transpose = True
    elif df.shape[0] == len(target_genes) and columns_overlap < len(target_genes) * 0.1:
            # 行数匹配且列名不匹配 -> 可能是转置
            print("  根据维度匹配，判定为 (Genes, Samples)，正在转置...")
            need_transpose = True
            
    if need_transpose:
        df = df.T
        
    # 检查是否需要筛选列 (Genes)
    common_genes = [g for g in target_genes if g in df.columns]
    if len(common_genes) != len(target_genes):
            print(f"  警告: 仅找到 {len(common_genes)} / {len(target_genes)} 个目标基因。")
    
    # 执行筛选 (Reindex 会自动处理缺失值为 NaN，填 0)
    df = df.reindex(columns=target_genes, fill_value=0.0)
    print(f"  已筛选并对齐到 {len(target_genes)} 个 Genes。")
    
    # 过滤样本 (Re-verify)
    valid_samples = [s for s in target_samples if s in df.index]
    missing = len(target_samples) - len(valid_samples)
    if missing > 0:
        print(f"  警告: {missing} 个样本在加载的数据中未找到。")
    
    if not valid_samples:
        raise Exception("  未找到任何匹配样本！")
    
    df_subset = df.loc[valid_samples]
    return df_subset, valid_samples


def _load_drug_target(drug_target_path,full_symbol_to_entrez,gene_to_idx):
    drug_to_targets = {} 

    df_targets = pd.read_csv(drug_target_path, sep='\t', dtype=str)
    for _, row in df_targets.iterrows():
        pert_id = row['pert_id']
        target_str = row['target']
        if pd.isna(target_str) or target_str == "" or target_str == '""': continue
        
        targets = [t.strip() for t in str(target_str).split(',')]
        target_indices = []
        for t_sym in targets:
            if t_sym in full_symbol_to_entrez:
                entrez = full_symbol_to_entrez[t_sym]
                if entrez in gene_to_idx:
                    target_indices.append(gene_to_idx[entrez])
        if target_indices:
            if pert_id not in drug_to_targets:
                drug_to_targets[pert_id] = set()
            drug_to_targets[pert_id].update(target_indices)
    return drug_to_targets

def _load_drug_fingerprint(fingerprint_path):
    drug_to_fp = {}
    fp_dim = 0
    if fingerprint_path and os.path.exists(fingerprint_path):
        print(f"正在加载药物指纹: {fingerprint_path}")
     
        df_fp = pd.read_csv(fingerprint_path)
        fp_cols = [c for c in df_fp.columns if c.startswith('fp')]
        if not fp_cols:
            raise Exception("药物指纹文件中未找到以 'fp' 开头的列。")
        fp_dim = len(fp_cols)        
        for _, row in df_fp.iterrows():
            d_id = str(row.iloc[0]) # Assuming first column is ID
            fp_vec = row[fp_cols].values.astype(np.float32)
            drug_to_fp[d_id] = fp_vec
    return drug_to_fp, fp_dim


def load_rfa_data(
    ctl_path, 
    trt_path, 
    landmark_path="data/landmark_genes.json", 
    drug_target_path="data/compound_targets.txt",
    siginfo_path="data/siginfo_beta.txt",
    fingerprint_path=None,
    use_landmark_genes=False,
    full_gene_path="data/GSE92742_Broad_LINCS_gene_info.txt",
    filter_time=24, # default 24h
    filter_dose=10,  # default 10uM
    cell_lines=None,
    ctl_residual_pool_size=0,
    pairing_mode="multi_trt_multi_ctl",
):
    """
    加载 RFA-GNN 专用数据 (直接从 Level 3 CSV 加载并对齐):
    1. 读取 siginfo 进行样本配对 (Trt vs Ctl based on cell, time, batch)
    2. 从原始 CSV (ctl_path, trt_path) 中只读取选定的样本列
    3. 构建 Drug-Target 特征
    4. 返回对齐后的 numpy array
    
    Args:
        cell_lines: 指定要保留的细胞系 (cell_iname)。None 表示不过滤；传入 str 或 list[str]。
        ctl_residual_pool_size: 每个 trt 随机配对的 ctl 数量（优先同 cell/batch）。<=0 时默认 3。
        pairing_mode:
            - "multi_trt_multi_ctl": 旧逻辑，一个 trt 可配多个 ctl，ctl 也可重复
            - "unique_trt_reuse_ctl": 每个 trt 仅配 1 个 ctl，ctl 可重复
            - "unique_trt_unique_ctl": 每个 trt 仅配 1 个 ctl，且 ctl 全局不重复
    """
    
    print(f"正在加载 RFA 数据 (Landmark Mode: {use_landmark_genes})...")
    print(f"CSV路径: CTL={ctl_path}, TRT={trt_path}")
    print(f"配对模式: {pairing_mode}")
    
    # 1. 加载基因列表
    target_genes = [] 

    full_symbol_to_entrez = _build_symbol_entrez_map(full_gene_path)


    # Check landmark availability (handle None)
    landmark_available = landmark_path and os.path.exists(landmark_path)

  
    cell_lines_set = None
    if cell_lines is not None and isinstance(cell_lines, str):
        s = str(cell_lines).strip()
        if s == "" or s.upper() in {"ALL", "NONE", "NULL"}:
            cell_lines_set = None
        else:
            cell_lines_set = {str(c).strip().upper() for c in s.split(",") if str(c).strip() != ""}
            if len(cell_lines_set) == 0:
                cell_lines_set = None
    elif cell_lines is not None and isinstance(cell_lines, (list, tuple, set)):
        cell_lines_set = {str(c).strip().upper() for c in cell_lines if str(c).strip() != ""}
        if len(cell_lines_set) == 0:
            cell_lines_set = None

    if cell_lines_set is not None:
        print(f"指定细胞系数量: {len(cell_lines_set)}")

    if use_landmark_genes and landmark_available:   
        try:
            with open(landmark_path, 'r') as f:
                genes_meta = json.load(f)
            # Update mapping just in case
            for g in genes_meta:
                if 'gene_symbol' in g and 'entrez_id' in g:
                    full_symbol_to_entrez[g['gene_symbol']] = str(g['entrez_id'])
            target_genes = [str(g['entrez_id']) for g in genes_meta]
        except Exception as e:
            print(f"加载 Landmark JSON 失败: {e}")
            return None
    elif use_landmark_genes and landmark_path and os.path.exists(landmark_path.replace('.json', '.txt')):
         # Fallback to txt
         txt_path = landmark_path.replace('.json', '.txt')
         with open(txt_path, 'r') as f:
            target_genes = [line.strip() for line in f if line.strip()]
    elif not use_landmark_genes and full_gene_path and os.path.exists(full_gene_path):
         # 加载 12328 全基因 (从 txt)
         # target_genes 已经在上面 map 加载时可以获取? 
         # 显式获取
         df_genes = pd.read_csv(full_gene_path, sep='\t', dtype=str)
         target_genes = df_genes['pr_gene_id'].tolist()
    else:
        # 自动检测 (当 landmark_path=None 或 文件不存在时)
        #print("未指定有效基因列表文件，尝试从数据文件自动识别...")
        raise Exception("未指定有效基因列表文件，无法自动识别。请检查数据文件格式或指定 landmark_path。")


    if not target_genes:
        raise Exception("目标基因列表为空")
        
    gene_to_idx = {gid: i for i, gid in enumerate(target_genes)}
    num_genes = len(target_genes)
    print(f"目标基因数: {num_genes}")

    # 2. 加载 SigInfo 并构建样本对齐映射
    print(f"正在读取元数据: {siginfo_path}")
    siginfo = pd.read_csv(siginfo_path, sep='\t', low_memory=False)
    
    # 过滤条件: trt_cp 或 ctl_vehicle
    siginfo = siginfo[siginfo['pert_type'].isin(['trt_cp', 'ctl_vehicle'])]
    
    # 过滤时间和剂量 (仅对 trt_cp 生效，ctl 需要匹配 trt 的时间)
    # 注意: CTL 通常也是有时间和批次的
    if filter_time:
        siginfo = siginfo[siginfo['pert_time'] == filter_time]
    
    # 剂量过滤比较模糊，通常只过滤 trt 的剂量
    # ctl 的 pert_dose 通常是 0 或 -666
    if filter_dose:
         # 仅保留 trt_cp 满足剂量 或 ctl_vehicle
        cond = (siginfo['pert_type'] == 'ctl_vehicle') | \
            ((siginfo['pert_type'] == 'trt_cp') & (np.abs(siginfo['pert_dose'] - filter_dose) < 0.1))
        siginfo = siginfo[cond]
    #如果cell_line_set不为None,说明前面有过滤逻辑
    if cell_lines_set is not None:
        before = len(siginfo)
        siginfo = siginfo[siginfo["cell_iname"].astype(str).str.upper().isin(cell_lines_set)]
        after = len(siginfo)
        uniq_cells = int(siginfo["cell_iname"].nunique()) if after > 0 else 0
        print(f"细胞系过滤: {before} -> {after}, 剩余细胞系数量(cells={uniq_cells})")
        if after == 0:
            raise Exception("细胞系过滤后 siginfo 为空，请检查 cell_lines 与 siginfo 的 cell_iname 是否一致。")
    
    print(f"过滤后元数据记录数: {len(siginfo)}")

    # --- Helper Function: Data Loader (H5/CSV) ---
   
    
    
    # 3.1 识别所有 Control 样本
    ctl_rows = siginfo[siginfo['pert_type'] == 'ctl_vehicle']
    ctl_id_to_cell = {}
    ctl_id_to_batch = {}
    ctl_id_to_det_plate = {}
    
    all_ctl_ids = []
    for _, row in ctl_rows.iterrows():
        d_ids = str(row['distil_ids']).split('|')
        c_name = row['cell_iname']
        b_name = str(row.get('bead_batch', 'Unknown'))
        plates_raw = row.get('det_plates', 'Unknown')
        plates = str(plates_raw).split('|')
        for i, d_id in enumerate(d_ids):
            if len(plates) == len(d_ids):
                p_name = str(plates[i])
            else:
                p_name = str(d_id).split(':')[0] if ':' in str(d_id) else str(plates_raw)
            ctl_id_to_cell[d_id] = c_name
            ctl_id_to_batch[d_id] = b_name
            ctl_id_to_det_plate[d_id] = p_name
            all_ctl_ids.append(d_id)
            
    all_ctl_ids = sorted(list(set(all_ctl_ids)))
    print(f"发现 {len(all_ctl_ids)} 个 Control 样本，涉及 {len(set(ctl_id_to_cell.values()))} 个细胞系。")
    
    # 3.2 加载所有 Control 数据
    print("正在加载所有 Control 数据以计算均值...")
    # 使用新的通用加载函数
    df_ctl_all, valid_ctl_ids = _load_expression_data(ctl_path, all_ctl_ids, target_genes)
    if df_ctl_all is None:
        raise Exception("  加载 Control 数据失败！")
        
    # 重索引确保基因顺序一致 (Samples x Genes) -> 已经是 Samples x Genes 了
    # 但需要确保列是 target_genes
    # 如果 H5 已经是 aligned landmark，那么列名应该是 Entrez ID
    # 我们检查一下列名覆盖率
    # common_genes = [g for g in target_genes if g in df_ctl_all.columns]
    # if len(common_genes) < len(target_genes) * 0.9:
    #     print(f"警告: 基因匹配率低 ({len(common_genes)}/{len(target_genes)})，尝试转置或检查 ID...")
    #     # 有可能 H5 列名是 Symbol 或者其他，这里假设已经是 Entrez
    
    # df_ctl_all = df_ctl_all.reindex(columns=target_genes, fill_value=0.0)
    
    # 3.3 计算每种细胞系的 Mean Vector
    #cell_mean_dict = {}
    
    # df_ctl_all 是 (Samples, Genes)
    # 添加 cell_iname 列
    df_ctl_temp = df_ctl_all.copy()
    df_ctl_temp['cell_iname'] = [ctl_id_to_cell.get(idx, 'Unknown') for idx in df_ctl_temp.index]
    df_ctl_temp['bead_batch'] = [ctl_id_to_batch.get(idx, 'Unknown') for idx in df_ctl_temp.index]
    


    # 4. 处理 Treatment 样本
    trt_rows = siginfo[(siginfo['pert_type'] == 'trt_cp') & (siginfo["is_hiq"] == 1)]

    trt_samples = []
    for _, row in trt_rows.iterrows():
        d_ids = str(row['distil_ids']).split('|')
        c_name = row['cell_iname']
        p_id = row['pert_id']
        b_name = str(row.get('bead_batch', 'Unknown'))
        plates_raw = row.get('det_plates', 'Unknown')
        plates = str(plates_raw).split('|')
        for i, d_id in enumerate(d_ids):
            if len(plates) == len(d_ids):
                p_name = str(plates[i])
            else:
                p_name = str(d_id).split(':')[0] if ':' in str(d_id) else str(plates_raw)
            trt_samples.append({
                'distil_id': d_id,
                'cell_iname': c_name,
                'pert_id': p_id,
                'bead_batch': b_name,
                'det_plate': p_name,
            })
    print(f"发现 {len(trt_samples)} 个高质量 Treatment 样本。")

    # 5. 加载 Treatment 数据
    target_trt_ids = [s['distil_id'] for s in trt_samples]
    print("正在加载 Treatment 数据...")
    df_trt_data, valid_trt_ids = _load_expression_data(trt_path, target_trt_ids, target_genes)
    if df_trt_data is None:
        raise Exception("  加载 Treatment 数据失败！")

    # 6. 构建可 split 的 trt anchor，并保留 ctl 候选
    ctl_index = pd.DataFrame({
        "distil_id": df_ctl_all.index.astype(str),
        "cell_iname": [ctl_id_to_cell.get(idx, "Unknown") for idx in df_ctl_all.index.astype(str)],
        "bead_batch": [ctl_id_to_batch.get(idx, "Unknown") for idx in df_ctl_all.index.astype(str)],
        "det_plate": [ctl_id_to_det_plate.get(idx, "Unknown") for idx in df_ctl_all.index.astype(str)],
    }).drop_duplicates()

    valid_pairing_modes = {"multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"}
    if pairing_mode not in valid_pairing_modes:
        raise ValueError(f"未知 pairing_mode: {pairing_mode}")

    trt_info_map = {s['distil_id']: s for s in trt_samples}
    trt_entries = []
    for trt_id, row in df_trt_data.iterrows():
        if trt_id not in trt_info_map:
            continue
        info = trt_info_map[trt_id]
        cell = info['cell_iname']
        batch = str(info.get('bead_batch', 'Unknown'))
        det_plate = str(info.get('det_plate', 'Unknown'))
        if det_plate == 'Unknown' or batch == 'Unknown':
            continue
        ctl_candidates = ctl_index[
            (ctl_index['cell_iname'] == cell) &
            (ctl_index['bead_batch'] == batch) &
            (ctl_index['det_plate'] == det_plate)
        ]['distil_id'].astype(str).tolist()
        if len(ctl_candidates) == 0:
            ctl_candidates = ctl_index[
                (ctl_index['cell_iname'] == cell) &
                (ctl_index['det_plate'] == det_plate)
            ]['distil_id'].astype(str).tolist()
        if len(ctl_candidates) == 0:
            continue
        trt_entries.append({
            "trt_id": str(trt_id),
            "x_trt": row.values.astype(np.float32),
            "info": info,
            "cell": cell,
            "batch": batch,
            "det_plate": det_plate,
            "ctl_candidates": ctl_candidates,
        })
    print(f"可用于 split-before-match 的 trt anchors: {len(trt_entries)}")

    # 7. 加载药物靶点 (Target Encoding) & 指纹 (Fingerprints)
    print(f"正在加载药物靶点: {drug_target_path}")
    drug_to_targets = _load_drug_target(drug_target_path,full_symbol_to_entrez,gene_to_idx)
    print(f"已加载 {len(drug_to_targets)} 个药物靶点。")

    drug_to_fp, fp_dim = _load_drug_fingerprint(fingerprint_path)
    print(f"已加载 {len(drug_to_fp)} 个药物指纹，维度: {fp_dim}。")

    num_anchor = len(trt_entries)
    if num_anchor > 0:
        anchor_X_trt = np.stack([e["x_trt"] for e in trt_entries], axis=0).astype(np.float32)
    else:
        anchor_X_trt = np.zeros((0, num_genes), dtype=np.float32)
    anchor_drug_ids = np.asarray([str(e["info"]["pert_id"]) for e in trt_entries], dtype=object)
    anchor_trt_distil_ids = np.asarray([str(e["trt_id"]) for e in trt_entries], dtype=object)
    anchor_cell_names = np.asarray([str(e["cell"]) for e in trt_entries], dtype=object)
    anchor_batch_ids = np.asarray([str(e["batch"]) for e in trt_entries], dtype=object)
    anchor_det_plate_ids = np.asarray([str(e["det_plate"]) for e in trt_entries], dtype=object)
    anchor_ctl_candidates = [list(e["ctl_candidates"]) for e in trt_entries]

    anchor_X_drug = np.zeros((num_anchor, num_genes), dtype=np.float32)
    anchor_has_target = np.zeros((num_anchor,), dtype=bool)
    anchor_has_fp = np.zeros((num_anchor,), dtype=bool)
    for i, pid in enumerate(anchor_drug_ids.tolist()):
        if pid in drug_to_targets:
            indices = list(drug_to_targets[pid])
            anchor_X_drug[i, indices] = 1.0
            anchor_has_target[i] = True
        if fp_dim > 0 and pid in drug_to_fp:
            anchor_has_fp[i] = True

    keep_mask = anchor_has_target
    print(f"  保留 {np.sum(keep_mask)} 个 trt anchors (required: targets)")
    print(f"  有 targets 的 anchors: {np.sum(anchor_has_target)}")
    print(f"  有 fingerprints 的 anchors: {np.sum(anchor_has_fp)}")

    if not np.all(keep_mask):
        print(f"过滤掉 {np.sum(~keep_mask)} 个药物特征不满足要求的 trt anchors...")
        keep_idx = np.where(keep_mask)[0]
        anchor_X_trt = anchor_X_trt[keep_idx]
        anchor_X_drug = anchor_X_drug[keep_idx]
        anchor_drug_ids = anchor_drug_ids[keep_idx]
        anchor_trt_distil_ids = anchor_trt_distil_ids[keep_idx]
        anchor_cell_names = anchor_cell_names[keep_idx]
        anchor_batch_ids = anchor_batch_ids[keep_idx]
        anchor_det_plate_ids = anchor_det_plate_ids[keep_idx]
        anchor_has_target = anchor_has_target[keep_idx]
        anchor_has_fp = anchor_has_fp[keep_idx]
        anchor_ctl_candidates = [anchor_ctl_candidates[int(i)] for i in keep_idx.tolist()]

    print(f"最终有效 trt anchors: {anchor_X_trt.shape}")

    uniq_drugs = list(dict.fromkeys([str(x) for x in anchor_drug_ids.tolist()]))
    drug2row = {d: i for i, d in enumerate(uniq_drugs)}
    fp_width = int(fp_dim) + 2
    drug_fp_table = np.zeros((len(uniq_drugs), fp_width), dtype=np.float32)
    for d, r in drug2row.items():
        vec = drug_to_fp.get(d)
        if vec is not None:
            drug_fp_table[r, :fp_dim] = vec
            drug_fp_table[r, fp_dim] = 1.0
        if d in drug_to_targets:
            drug_fp_table[r, fp_dim + 1] = 1.0
    anchor_drug_fp_idx = np.asarray([drug2row[str(pid)] for pid in anchor_drug_ids.tolist()], dtype=np.int32) if len(anchor_drug_ids) > 0 else np.zeros((0,), dtype=np.int32)

    anchor_X_fingerprint = None
    materialize_limit_mb = 512
    need_bytes_anchor = int(len(anchor_drug_ids)) * int(fp_width) * 4
    if len(anchor_drug_ids) > 0 and need_bytes_anchor <= int(materialize_limit_mb) * 1024 * 1024:
        anchor_X_fingerprint = drug_fp_table[anchor_drug_fp_idx]

    used_ctl_ids = []
    seen_ctl = set()
    for candidates in anchor_ctl_candidates:
        for ctl_id in candidates:
            ctl_id = str(ctl_id)
            if ctl_id in seen_ctl or ctl_id not in df_ctl_all.index:
                continue
            seen_ctl.add(ctl_id)
            used_ctl_ids.append(ctl_id)
    if len(used_ctl_ids) > 0:
        ctl_pool_expr = df_ctl_all.loc[used_ctl_ids].values.astype(np.float32)
        ctl_pool_ids = np.asarray(used_ctl_ids, dtype=object)
    else:
        ctl_pool_expr = np.zeros((0, num_genes), dtype=np.float32)
        ctl_pool_ids = np.asarray([], dtype=object)

    base_data = {
        "anchor_X_trt": anchor_X_trt,
        "anchor_X_drug": anchor_X_drug,
        "anchor_X_fingerprint": anchor_X_fingerprint,
        "anchor_drug_fp_idx": anchor_drug_fp_idx,
        "anchor_drug_has_target": anchor_has_target.astype(np.float32),
        "anchor_drug_has_fp": anchor_has_fp.astype(np.float32),
        "anchor_drug_ids": anchor_drug_ids,
        "anchor_trt_distil_ids": anchor_trt_distil_ids,
        "anchor_cell_names": anchor_cell_names,
        "anchor_batch_ids": anchor_batch_ids,
        "anchor_det_plate_ids": anchor_det_plate_ids,
        "anchor_ctl_candidates": anchor_ctl_candidates,
        "ctl_pool_ids": ctl_pool_ids,
        "ctl_pool_expr": ctl_pool_expr,
        "drug_fp_table": drug_fp_table,
        "input_dim": num_genes,
        "node_feature_dim": 2,
        "target_genes": target_genes,
        "loss_mask": np.ones((1, num_genes), dtype=np.float32),
        "symbol_to_entrez": full_symbol_to_entrez,
        "pairing_mode": pairing_mode,
    }

    paired = assemble_paired_dataset(
        base_data,
        pairing_mode=pairing_mode,
        ctl_residual_pool_size=int(ctl_residual_pool_size),
        seed=42,
    )
    print(f"数据组装完成: {paired['X_ctl'].shape}")

    data = dict(base_data)
    data.update(
        {
            "X_ctl": paired["X_ctl"],
            "y_delta": paired["y_delta"],
            "X_drug": paired["X_drug"],
            "X_fingerprint": paired["X_fingerprint"],
            "drug_fp_idx": paired["drug_fp_idx"],
            "drug_has_target": paired["drug_has_target"],
            "drug_has_fp": paired["drug_has_fp"],
            "drug_ids": paired["drug_ids"],
            "trt_distil_ids": paired["trt_distil_ids"],
            "ctl_distil_ids": paired["ctl_distil_ids"],
            "cell_names": paired["cell_names"],
            "batch_ids": paired["batch_ids"],
            "det_plate_ids": paired["det_plate_ids"],
        }
    )
    return data
