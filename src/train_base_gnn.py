import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_gnn import BaseLineGAT
from data_loader import load_rfa_data, build_combined_gnn
from train_deepcop import cold_split
from xpert_preprocess import generate_cleaned_gctx_pairs

# Loss functions (copied from train_rfa.py)
def pcc_loss(y_true, y_pred):
    mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
    my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    xm = y_true - mx
    ym = y_pred - my
    r_num = tf.reduce_sum(xm * ym, axis=1)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8)
    r = r_num / r_den
    return 1.0 - tf.reduce_mean(r)

def combined_loss(y_true, y_pred):
    mse = keras.losses.MeanSquaredError()(y_true, y_pred)
    pcc = pcc_loss(y_true, y_pred)
    return mse + 5.0 * pcc

def _load_landmark_genes(landmark_genes_path: str) -> List[str]:
    with open(landmark_genes_path, "r") as f:
        genes_meta = json.load(f)
    return [str(g["entrez_id"]) for g in genes_meta if "entrez_id" in g]


def _build_symbol_to_entrez(full_gene_path: str) -> Dict[str, str]:
    symbol_to_entrez: Dict[str, str] = {}
    if not full_gene_path or not os.path.exists(full_gene_path):
        return symbol_to_entrez
    df_genes = pd.read_csv(full_gene_path, sep="\t", dtype=str)
    for _, row in df_genes.iterrows():
        symbol_to_entrez[row["pr_gene_symbol"]] = row["pr_gene_id"]
    return symbol_to_entrez


def _build_drug_target_map(
    target_path: str,
    gene_to_idx: Dict[str, int],
    symbol_to_entrez: Dict[str, str],
) -> Dict[str, List[int]]:
    drug_to_targets: Dict[str, List[int]] = {}
    df_targets = pd.read_csv(target_path, sep="\t", dtype=str)
    for _, row in df_targets.iterrows():
        pert_id = row["pert_id"]
        target_str = row["target"]
        if pd.isna(target_str) or target_str == "" or target_str == '""':
            continue
        targets = [t.strip() for t in str(target_str).split(",") if t.strip()]
        target_indices = []
        for t_sym in targets:
            if t_sym in symbol_to_entrez:
                entrez = symbol_to_entrez[t_sym]
                if entrez in gene_to_idx:
                    target_indices.append(gene_to_idx[entrez])
        if target_indices:
            drug_to_targets[pert_id] = target_indices
    return drug_to_targets


def _load_xpert_augmented_data(
    siginfo_path: str,
    trt_gctx_path: str,
    ctl_gctx_path: str,
    landmark_genes_path: str,
    target_path: str,
    full_gene_path: str,
    seeds: Sequence[int],
    output_dir: str,
    output_suffix_base: str,
    plate_col: str = "det_plates",
    filter_time: int = 24,
    filter_dose: float = 10.0,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if not seeds:
        raise ValueError("seeds 不能为空")

    landmark_genes = _load_landmark_genes(landmark_genes_path)
    gene_to_idx = {gid: i for i, gid in enumerate(landmark_genes)}
    symbol_to_entrez = _build_symbol_to_entrez(full_gene_path)
    drug_to_targets = _build_drug_target_map(target_path, gene_to_idx, symbol_to_entrez)

    siginfo = pd.read_csv(siginfo_path, sep="\t", low_memory=False)
    siginfo = siginfo[siginfo["pert_type"].isin(["trt_cp", "ctl_vehicle"])].copy()
    siginfo = siginfo.dropna(subset=["pert_id", "cell_iname", "pert_time", "distil_ids"])
    if filter_time is not None:
        siginfo = siginfo[siginfo["pert_time"] == filter_time]
    if filter_dose is not None and "pert_dose" in siginfo.columns:
        dose_mask = (siginfo["pert_type"] == "ctl_vehicle") | (
            (siginfo["pert_type"] == "trt_cp") & (np.abs(siginfo["pert_dose"] - filter_dose) < 0.1)
        )
        siginfo = siginfo[dose_mask]
    if "is_hiq" in siginfo.columns:
        siginfo = siginfo[(siginfo["pert_type"] == "ctl_vehicle") | (siginfo["is_hiq"] == 1)]

    siginfo_exp = siginfo.copy()
    siginfo_exp["distil_ids"] = siginfo_exp["distil_ids"].astype(str).str.split("|")
    siginfo_exp = siginfo_exp.explode("distil_ids").rename(columns={"distil_ids": "distil_id"})

    trt_meta = siginfo_exp[siginfo_exp["pert_type"] == "trt_cp"][
        ["distil_id", "pert_id", "cell_iname"]
    ].dropna()
    distil_to_meta = trt_meta.set_index("distil_id").to_dict("index")

    all_trt = []
    all_ctl = []
    all_targets = []
    all_drug_ids = []
    all_cells = []

    for seed in seeds:
        output_suffix = f"{output_suffix_base}_seed{seed}"
        trt_paths, ctl_paths = generate_cleaned_gctx_pairs(
            siginfo_path=siginfo_path,
            trt_gctx_path=trt_gctx_path,
            ctl_gctx_path=ctl_gctx_path,
            landmark_genes_path=landmark_genes_path,
            output_dir=output_dir,
            output_suffix=output_suffix,
            plate_col=plate_col,
            filter_time=filter_time,
            filter_dose=filter_dose,
            seed=seed,
        )

        trt_df = pd.read_hdf(trt_paths[0], key="data")
        ctl_df = pd.read_hdf(ctl_paths[0], key="data")

        trt_df = trt_df.reindex(columns=landmark_genes, fill_value=0.0)
        ctl_df = ctl_df.reindex(columns=landmark_genes, fill_value=0.0)

        # Aggregate duplicated distil_id rows to avoid reindex errors
        if trt_df.index.has_duplicates:
            trt_df = trt_df.groupby(level=0).mean()
        if ctl_df.index.has_duplicates:
            ctl_df = ctl_df.groupby(level=0).mean()

        if not trt_df.index.equals(ctl_df.index):
            ctl_df = ctl_df.reindex(trt_df.index)

        trt_vals = trt_df.values.astype(np.float32)
        ctl_vals = ctl_df.values.astype(np.float32)


        drug_ids = []
        cell_names = []
        keep_mask = []
        for distil_id in trt_df.index.astype(str):
            meta = distil_to_meta.get(distil_id)
            if not meta:
                keep_mask.append(False)
                drug_ids.append("")
                cell_names.append("Unknown")
                continue
            drug_ids.append(meta["pert_id"])
            cell_names.append(meta["cell_iname"])
            keep_mask.append(True)

        keep_mask = np.array(keep_mask)
        print(
            f"[Xpert seed {seed}] trt_rows={len(trt_df)} ctl_rows={len(ctl_df)} kept={int(keep_mask.sum())}"
        )
        if not np.any(keep_mask):
            continue

        trt_vals = trt_vals[keep_mask]
        ctl_vals = ctl_vals[keep_mask]
        drug_ids = [pid for pid, k in zip(drug_ids, keep_mask) if k]
        cell_names = [c for c, k in zip(cell_names, keep_mask) if k]

        x_target = np.zeros((len(drug_ids), len(landmark_genes)), dtype=np.float32)
        for i, pid in enumerate(drug_ids):
            for idx in drug_to_targets.get(pid, []):
                x_target[i, idx] = 1.0

        all_trt.append(trt_vals)
        all_ctl.append(ctl_vals)
        all_targets.append(x_target)
        all_drug_ids.extend(drug_ids)
        all_cells.extend(cell_names)

    if not all_trt:
        raise ValueError("没有生成有效的配对数据。")

    X_trt_arr = np.vstack(all_trt)
    X_ctl_arr = np.vstack(all_ctl)
    X_target = np.vstack(all_targets)

    if max_samples is not None and X_trt_arr.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_trt_arr.shape[0], size=max_samples, replace=False)
        X_trt_arr = X_trt_arr[idx]
        X_ctl_arr = X_ctl_arr[idx]
        X_target = X_target[idx]
        all_drug_ids = [all_drug_ids[i] for i in idx]
        all_cells = [all_cells[i] for i in idx]

    X_node = np.stack([X_ctl_arr, X_target], axis=-1).astype(np.float32)
    y_delta = X_trt_arr - X_ctl_arr

    return {
        "X_ctl": X_node,
        "y_delta": y_delta,
        "X_drug": X_target,
        "drug_ids": all_drug_ids,
        "cell_names": all_cells,
        "input_dim": len(landmark_genes),
        "node_feature_dim": 2,
        "target_genes": landmark_genes,
        "loss_mask": np.ones((1, len(landmark_genes)), dtype=np.float32),
    }


def train_base_gnn(
    debug: bool = False,
    use_landmark: bool = False,
    limit: int = 20000,
    use_xpert_aug: bool = False,
    xpert_seeds: Optional[Sequence[int]] = None,
):
    # 1. Load Data
    landmark_path = "data/landmark.txt"
    landmark_genes_path = "data/landmark_genes.json"
    siginfo_path = "data/siginfo_beta.txt"
    ctl_csv_path = "data/cmap/level3_beta_ctl_n188708x12328.h5"
    trt_csv_path = "data/cmap/level3_beta_trt_cp_n1805898x12328.h5"
    full_gene_path = "data/GSE92742_Broad_LINCS_gene_info.txt"
    
    print(f"\n>>> Phase 1: 加载表达数据和药物靶点 (Landmark: {use_landmark}) <<<")
    
    if use_xpert_aug:
        if not use_landmark:
            raise ValueError("Xpert 数据增强仅支持 use_landmark=True")
        if not xpert_seeds:
            raise ValueError("use_xpert_aug=True 时必须提供 xpert_seeds")
        data = _load_xpert_augmented_data(
            siginfo_path=siginfo_path,
            trt_gctx_path="data/cmap/level3_beta_trt_cp_n1805898x12328.gctx",
            ctl_gctx_path="data/cmap/level3_beta_ctl_n188708x12328.gctx",
            landmark_genes_path=landmark_genes_path,
            target_path="data/compound_targets.txt",
            full_gene_path=full_gene_path,
            seeds=xpert_seeds,
            output_dir="data/cmap",
            output_suffix_base="landmark_cleaned",
            max_samples=limit if debug else None,
        )
    else:
        data = load_rfa_data(
            ctl_csv_path, trt_csv_path,
            landmark_path=landmark_genes_path,
            siginfo_path=siginfo_path,
            use_landmark_genes=use_landmark,
            full_gene_path=full_gene_path,
            max_samples=limit if debug else 20000 # Use limit for loading if debug
        )
    if data is None: raise ValueError("加载数据失败")
    
    if debug:
        print("\n[DEBUG MODE] Truncating data...")
        data["X_ctl"] = data["X_ctl"][:limit]
        data["y_delta"] = data["y_delta"][:limit]
        data["X_drug"] = data["X_drug"][:limit]
        data["drug_ids"] = data["drug_ids"][:limit]

    # 2. Build Graph
    print("\n>>> Phase 2: 构建生物学图结构 (OmniPath TF + PPI) <<<")
    target_genes = data["target_genes"]
        
    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        landmark_genes=target_genes, 
        landmark_path=landmark_genes_path, 
        tf_path="data/omnipath/omnipath_tf_regulons.csv",
        ppi_path="data/omnipath/omnipath_interactions.csv" # Changed from None to load PPI
    )

    if len(node_list) != len(target_genes) or node_list[:50] != target_genes[:50]:
        raise ValueError(
            "Graph node_list 与表达 target_genes 顺序/长度不一致，训练会发生基因错位。"
        )
    
    # 3. Data Split
    train_data, test_data = cold_split(data)
    train_ctl, train_trt, train_drug = train_data
    test_ctl, test_trt, test_drug = test_data
    
    # 提取 Cell 信息
    # cold_split 默认返回 (ctl, trt, drug)，我们需要修改它或者手动提取
    # 简单起见，我们重新从 data 中提取并分割，因为 cold_split 逻辑比较固定
    # 实际上 cold_split 内部使用了 mask。我们可以复用 mask。
    # 为了不修改 train_deepcop.py，我们这里手动做一次 split (hacky but safe)
    
    # 获取 mask (hack: cold_split 返回的是 array，我们反推 mask? 不太行)
    # 更好的方法: 我们自己实现简单的 cold split 或者修改 cold_split 返回 mask
    
    # 让我们看看 train_deepcop.py 的 cold_split
    # 算了，直接在这里实现 split 逻辑，或者假设我们能访问 mask
    # 为了方便，我们这里直接使用 data['drug_ids'] 来分割
    
    # Encode Cells First
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    all_cells = data['cell_names']
    cell_idx = le.fit_transform(all_cells)
    num_cells = len(le.classes_)
    print(f"检测到 {num_cells} 个细胞系。")
    
    # Split
    unique_drugs = np.unique(data['drug_ids'])
    np.random.seed(42)
    test_drugs = np.random.choice(unique_drugs, int(len(unique_drugs) * 0.2), replace=False)
    
    test_mask = np.isin(data['drug_ids'], test_drugs)
    train_mask = ~test_mask
    
    train_ctl = data['X_ctl'][train_mask]
    train_trt = data['y_delta'][train_mask]
    train_drug = data['X_drug'][train_mask]
    train_cells = cell_idx[train_mask]
    
    test_ctl = data['X_ctl'][test_mask]
    test_trt = data['y_delta'][test_mask]
    test_drug = data['X_drug'][test_mask]
    test_cells = cell_idx[test_mask]
    
    # 4. Model
    print(f"\n>>> Phase 3: 初始化 BaseLineGAT 模型 (Genes: {len(target_genes)}, Cells: {num_cells}) <<<")
    model = BaseLineGAT(
        num_genes=len(target_genes),
        num_cells=num_cells,
        hidden_dim=64,
        num_heads=4, # Increased heads
        dropout=0.2,
        use_residual=False
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=5e-4) # Increased LR
    
    # 自定义 Masked Loss (仅对 Landmark 基因计算 Loss)
    loss_mask = tf.constant(data['loss_mask'], dtype=tf.float32)
    
    # def masked_combined_loss(y_true, y_pred):
    #     # 1. Masking for MSE
    #     y_true_m = y_true * loss_mask
    #     y_pred_m = y_pred * loss_mask
        
    #     # Sum of squared errors / Number of valid genes
    #     valid_count = tf.reduce_sum(loss_mask)
    #     mse = tf.reduce_sum(tf.square(y_true_m - y_pred_m)) / (valid_count * tf.cast(tf.shape(y_true)[0], tf.float32))
        
    #     # 2. Masking for PCC (Extract valid columns)
    #     # tf.boolean_mask returns flattened tensor if mask is 1D? 
    #     # We need to gather columns.
    #     valid_indices = tf.where(loss_mask[0] > 0)[:, 0]
    #     y_t_valid = tf.gather(y_true, valid_indices, axis=1)
    #     y_p_valid = tf.gather(y_pred, valid_indices, axis=1)
        
    #     pcc = pcc_loss(y_t_valid, y_p_valid)
        
    #     return mse + 5.0 * pcc

    # Wrapper for Adj
    class GAT_Wrapper(keras.Model):
        def __init__(self, gat_model, adj_matrix):
            super().__init__()
            self.gat = gat_model
            self.adj = tf.constant(adj_matrix, dtype=tf.float32)
            
        def call(self, inputs):
            ctl, drug_targets, cell_idx = inputs
            cell_idx = tf.cast(cell_idx, tf.int32)
            return self.gat([self.adj, ctl, drug_targets, cell_idx])
            
    wrapped_model = GAT_Wrapper(model, adj_matrix)
    wrapped_model.compile(optimizer=optimizer, loss=masked_combined_loss, metrics=[keras.metrics.MeanSquaredError()])
    
    # 5. Train
    print("\n>>> Phase 4: 开始训练 <<<")
    wrapped_model.fit(
        [train_ctl, train_drug, train_cells],
        train_trt,
        epochs=20, # Increased epochs
        batch_size=32, 
        validation_data=([test_ctl, test_drug, test_cells], test_trt)
    )
    
    # 6. Eval (Simplified)
    print("评估中 (只评估 Landmark 基因)...")
    pred = wrapped_model.predict([test_ctl, test_drug, test_cells], batch_size=32)
    
    # 提取 Landmark 部分进行评估
    valid_indices = np.where(data['loss_mask'][0] > 0)[0]
    test_trt_valid = test_trt[:, valid_indices]
    pred_valid = pred[:, valid_indices]
    
    # Calculate Sample-wise PCC (consistent with loss)
    # Row-wise correlation
    pcc_list = []
    for i in range(len(test_trt_valid)):
        if np.std(test_trt_valid[i]) > 1e-6 and np.std(pred_valid[i]) > 1e-6:
            p, _ = pearsonr(test_trt_valid[i], pred_valid[i])
            pcc_list.append(p)
    
    avg_pcc = np.mean(pcc_list) if pcc_list else 0.0
    
    mse = mean_squared_error(test_trt_valid, pred_valid)
    # pcc = pearsonr(test_trt_valid.flatten(), pred_valid.flatten())[0] # Global PCC (deprecated)
    
    print(f"BaseLineGAT | MSE: {mse:.4f} | Sample-wise PCC: {avg_pcc:.4f}")

if __name__ == "__main__":
    # 可以修改这里来切换模式
    # use_landmark=True: 只用 978 基因 (快速，稀疏图)
    # use_landmark=False: 用 12328 基因 (慢，全图，Zero Padding)
    
    # 全量数据跑，只跑 Landmark 基因
    # 关闭 debug (读取所有数据)
    # 开启 use_landmark (只用 978 基因)
    train_base_gnn(
        debug=False,
        use_landmark=True,
        limit=20000,
        use_xpert_aug=True,
        xpert_seeds=[42, 43, 44],
    )
