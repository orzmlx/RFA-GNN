from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from data_loader import load_rfa_data, build_combined_gnn
# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rfa_gcn_drug import RFAGCN_Drug
from train_deepcop import load_deepcop_data, cold_split

def build_adjacency_matrix(landmark_genes, interactions):
    """
    基于固定的 Landmark Genes 列表构建邻接矩阵。
    确保矩阵的行/列顺序与表达数据的列顺序严格一致。
    """
    print(f"正在构建 {len(landmark_genes)} x {len(landmark_genes)} 的邻接矩阵...")
    
    # 1. 建立基因到索引的映射
    gene_to_idx = {gene: i for i, gene in enumerate(landmark_genes)}
    num_genes = len(landmark_genes)
    
    # 2. 初始化全零矩阵
    adj = np.zeros((num_genes, num_genes), dtype=np.float32)
    
    # 3. 填充 OmniPath 相互作用
    count = 0
    # 过滤出 source 和 target 都在 landmark_genes 中的边
    valid_interactions = interactions[
        interactions["source_genesymbol"].isin(landmark_genes) & 
        interactions["target_genesymbol"].isin(landmark_genes)
    ]
    
    for _, row in valid_interactions.iterrows():
        src = row["source_genesymbol"]
        tgt = row["target_genesymbol"]
        weight = row["weight"] # +1 或 -1
        
        if src in gene_to_idx and tgt in gene_to_idx:
            i = gene_to_idx[src]
            j = gene_to_idx[tgt]
            adj[i, j] = weight
            count += 1
            
    print(f"邻接矩阵构建完成。包含 {count} 条边 (Coverage: {count / (num_genes*num_genes):.4%})")
    return adj

def pcc_loss(y_true, y_pred):
    """Pearson Correlation Loss"""
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

def train_rfa(debug=False):
    # 1. 路径配置
    landmark_path = "data/landmark.txt"
    landmark_genes_path = "data/landmark_genes.json"
    morgan_path = "data/new_morgan_fingerprints.csv"
    siginfo_path = "data/siginfo_beta.txt"
    # go_path = "DeepCOP/Data/go_fingerprints.csv"
    
    # 使用全量基因数据 (12328 genes)
    ctl_csv_path = "data/cmap/level3_beta_ctl_n188708x12328.h5"
    trt_csv_path = "data/cmap/level3_beta_trt_cp_n1805898x12328.h5"
    
    # 2. 加载 RFA 数据 (替代 DeepCOP loader)
    print(f"正在加载 RFA 数据 (Full Genes Mode)...")
    print(f"H5路径: CTL={ctl_csv_path}, TRT={trt_csv_path}")
    
    # 不再传入 landmark_path，表示使用全部基因
    # 注意: max_samples 依然重要，防止爆内存
    data = load_rfa_data(ctl_csv_path, trt_csv_path, landmark_path=None, siginfo_path=siginfo_path, max_samples=5000)
    if data is None: return
    
    # DEBUG: 截断数据以快速验证流程
    if debug:
        print("\n[DEBUG MODE] 截断数据以快速验证流程...")
        limit = 1000
        data["X_ctl"] = data["X_ctl"][:limit]
        data["y_trt"] = data["y_trt"][:limit]
        data["X_drug"] = data["X_drug"][:limit]
        data["drug_ids"] = data["drug_ids"][:limit]


    # 3. 加载 OmniPath 并构建图
    print("\n>>> Phase 2: 构建生物学图结构 (OmniPath) <<<")
    
    # 使用 data 中加载的 target_genes (可能是 978 也可能是 12328)
    target_genes = data["target_genes"]
    
    # 使用 build_combined_gnn 构建更密集的图 (TF + PPI)
    print("正在构建 Combined OmniPath 图 (TF + PPI)...")
    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        landmark_genes=target_genes, 
        landmark_path=None, # 避免内部重新加载 978 映射
        tf_path="data/omnipath/omnipath_tf_regulons.csv",
        ppi_path="data/omnipath/omnipath_interactions.csv",
    )
    
    # --- DEBUG: 检查图的稀疏度和覆盖率 ---
    num_edges = np.sum(adj_matrix)
    num_nodes = adj_matrix.shape[0]
    sparsity = num_edges / (num_nodes * num_nodes)
    print(f"\n[GRAPH DEBUG]")
    print(f"节点数: {num_nodes}")
    print(f"边数 (非零元素): {int(num_edges)}")
    print(f"稀疏度: {sparsity:.6%}")
    
    if num_edges == 0:
        raise ValueError("警告: 这是一个空图！GNN 将无法学习拓扑结构。请检查 OmniPath 数据与 Landmark 基因的匹配情况。")


    # 4. 数据划分
    train_data, test_data = cold_split(data)
    
    # Unpack (Note: cold_split returns tuples, need to check if it handles cell_names)
    # cold_split in train_deepcop.py:
    # return (X_ctl_train, y_trt_train, X_drug_train), (X_ctl_test, y_trt_test, X_drug_test)
    # It doesn't handle cell_names yet! We need to manually split cell_names or update cold_split.
    # Or simpler: let's do manual splitting here using the same mask logic.
    
    drug_ids = data["drug_ids"]
    unique_drugs = np.unique(drug_ids)
    np.random.seed(42)
    np.random.shuffle(unique_drugs)
    split_idx = int(len(unique_drugs) * 0.8)
    train_drugs = set(unique_drugs[:split_idx])
    
    train_mask = np.array([d in train_drugs for d in drug_ids])
    test_mask = ~train_mask
    
    train_ctl = data["X_ctl"][train_mask]
    train_trt = data["y_trt"][train_mask]
    train_drug = data["X_drug"][train_mask]
    train_drug_ids_raw = [data["drug_ids"][i] for i in range(len(data["drug_ids"])) if train_mask[i]]
    train_cells_raw = [data["cell_names"][i] for i in range(len(data["cell_names"])) if train_mask[i]]
    
    test_ctl = data["X_ctl"][test_mask]
    test_trt = data["y_trt"][test_mask]
    test_drug = data["X_drug"][test_mask]
    test_drug_ids_raw = [data["drug_ids"][i] for i in range(len(data["drug_ids"])) if test_mask[i]]
    test_cells_raw = [data["cell_names"][i] for i in range(len(data["cell_names"])) if test_mask[i]]
    
    print(f"训练集: {len(train_ctl)}, 测试集: {len(test_ctl)}")
    
    # Encode Cell Names
    from sklearn.preprocessing import LabelEncoder
    le_cell = LabelEncoder()
    # Fit on all possible cells
    all_cells = data["cell_names"]
    le_cell.fit(all_cells)
    train_cells = le_cell.transform(train_cells_raw)
    test_cells = le_cell.transform(test_cells_raw)
    num_cells = len(le_cell.classes_)
    print(f"检测到 {num_cells} 个细胞系。")
    
    # Encode Drug IDs
    le_drug = LabelEncoder()
    all_drugs = data["drug_ids"]
    le_drug.fit(all_drugs)
    train_drug_idx = le_drug.transform(train_drug_ids_raw)
    test_drug_idx = le_drug.transform(test_drug_ids_raw)
    num_drugs = len(le_drug.classes_)
    print(f"检测到 {num_drugs} 种药物。")

    # 5. 准备模型输入
    # RFA-GNN 需要: [adj, ctl_expr, drug_target, drug_idx, cell_idx]
    
    print("\n>>> Phase 3: 初始化 RFA-GNN 模型 <<<")
    
    model = RFAGCN_Drug(
        num_genes=978,
        num_cells=num_cells, 
        num_drugs=num_drugs, 
        hidden_dim=256, 
        alpha=0.3,
        use_residual=False
    )
    
    # 编译模型
    optimizer = keras.optimizers.Adam(learning_rate=1e-3) 
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[keras.metrics.MeanSquaredError(), pcc_loss])
    
    # 6. 训练
    print("\n>>> Phase 4: 开始训练 <<<")
    
    class RFA_Wrapper(keras.Model):
        def __init__(self, rfa_model, adj_matrix):
            super().__init__()
            self.rfa = rfa_model
            self.adj = tf.constant(adj_matrix, dtype=tf.float32)
            
        def call(self, inputs):
            # inputs: [ctl, drug_targets, drug_idx, cell_idx]
            ctl, drug_targets, drug_idx, cell_idx = inputs 
            cell_idx = tf.cast(cell_idx, tf.int32)
            drug_idx = tf.cast(drug_idx, tf.int32)
            return self.rfa([self.adj, ctl, drug_targets, drug_idx, cell_idx])
            
    wrapped_model = RFA_Wrapper(model, adj_matrix)
    wrapped_model.compile(optimizer=optimizer, loss=combined_loss, metrics=[keras.metrics.MeanSquaredError(), pcc_loss])
    
    print("开始训练 Wrapper 模型...")
    wrapped_model.fit(
        [train_ctl, train_drug, train_drug_idx, train_cells],
        train_trt,
        epochs=15, 
        batch_size=1, # Reduced to 1 to avoid OOM (12k^2 Attention Matrix)
        validation_data=([test_ctl, test_drug, test_drug_idx, test_cells], test_trt)
    )
    
    # 保存
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    wrapped_model.save_weights("saved_models/rfa_gnn.weights.h5")
    print("模型已保存。")

    # 7. 最终评估
    print(f"\n>>> 最终评估 (Cold Split Test Set) <<<")
    
    # 预测
    pred_rfa = wrapped_model.predict([test_ctl, test_drug, test_drug_idx, test_cells])
    
    # --- 评估指标 (复用 DeepCOP 逻辑) ---
    mse_rfa = mean_squared_error(test_trt, pred_rfa)
    pcc_rfa = pearsonr(test_trt.flatten(), pred_rfa.flatten())[0]
    
    # 计算 Baseline (Identity) - 即假设药物无效 (Output = Input) 时的指标
    # Input Ctl Expression is test_ctl[:, :, 0]
    # 在 Delta 模式下，Baseline 是 "预测 Delta 为 0"
    # 即 y_pred = 0. 
    # y_true 是 test_trt (即 Delta)
    
    mse_baseline = mean_squared_error(test_trt, np.zeros_like(test_trt))
    # PCC Baseline: Correlation(Delta, 0) 是未定义的 (NaN)，因为 0 没有方差
    # 但我们可以计算 Correlation(Delta_True, Random_Noise) 作为底线，或者是 Correlation(Delta_True, Mean_Delta_Train)
    # DeepCOP 通常不计算 Delta 模式下的 Baseline PCC，因为显然是 0。
    # 这里我们只打印 MSE Baseline
    
    print(f"\n[Baseline Comparison] Predict Delta = 0 (No Effect):")
    print(f"MSE: {mse_baseline:.4f} (如果模型 MSE 低于此值，说明学到了有效信息)")

    # PCC (Sample-wise)
    pcc_sample = []
    pcc_top100_change = [] # 基于高变化基因 (Diff Exp)
    pcc_top50_change = []
    
    for i in range(test_trt.shape[0]):
        y_true_delta = test_trt[i, :] # This is Delta
        y_pred_delta = pred_rfa[i, :] # This is Pred Delta
        
        if np.std(y_true_delta) > 1e-6 and np.std(y_pred_delta) > 1e-6:
            # 1. All Genes PCC (on Delta)
            p, _ = pearsonr(y_true_delta, y_pred_delta)
            pcc_sample.append(p)
            
            # 2. Top-100 High Change (Diff Exp)
            # 在 Delta 模式下，这就是绝对值最大的 Delta
            top100_change_idx = np.argsort(np.abs(y_true_delta))[-100:]
            
            if np.std(y_true_delta[top100_change_idx]) > 1e-6 and np.std(y_pred_delta[top100_change_idx]) > 1e-6:
                p100_c, _ = pearsonr(y_true_delta[top100_change_idx], y_pred_delta[top100_change_idx])
                pcc_top100_change.append(p100_c)

            # 3. Top-50 High Change
            top50_change_idx = np.argsort(np.abs(y_true_delta))[-50:]
            if np.std(y_true_delta[top50_change_idx]) > 1e-6 and np.std(y_pred_delta[top50_change_idx]) > 1e-6:
                p50_c, _ = pearsonr(y_true_delta[top50_change_idx], y_pred_delta[top50_change_idx])
                pcc_top50_change.append(p50_c)
                
    avg_pcc_sample = np.mean(pcc_sample) if pcc_sample else 0.0
    avg_pcc_top100_change = np.mean(pcc_top100_change) if pcc_top100_change else 0.0
    avg_pcc_top50_change = np.mean(pcc_top50_change) if pcc_top50_change else 0.0
    
    print(f"{'Model':<15} | {'MSE':<10} | {'PCC(All Delta)':<15} | {'PCC(Top100 Delta)':<20} | {'PCC(Top50 Delta)':<20}")
    print("-" * 100)
    print(f"{'RFA-GNN':<15} | {mse_rfa:.4f}     | {avg_pcc_sample:.4f}            | {avg_pcc_top100_change:.4f}               | {avg_pcc_top50_change:.4f}")



if __name__ == "__main__":
    # 设置为 True 进行快速测试，False 进行全量训练
    train_rfa(debug=False)
