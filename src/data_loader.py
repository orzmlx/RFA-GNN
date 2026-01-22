import pandas as pd
import numpy as np
import omnipath as op
import tensorflow as tf
from cmapPy.pandasGEXpress.parse import parse

def load_omnipath_network(genes_list=None):
    """
    使用 OmniPath API 下载并处理调控网络。
    """
    print("正在从 OmniPath 获取调控网络数据...")
    
    # 1. 下载转录调控网络 (Transcriptional) 和 信号传导网络 (Post-transcriptional/PPI)
    # 默认获取所有有向交互
    interactions = op.interactions.import_intercell_network(
        transmitter_params={"categories": "ligand"},
        receiver_params={"categories": "receptor"}
    )
    
    # 或者获取通用的 directed interactions (更常用)
    # 包含 Dorothea (TF-target), Kinase-substrate 等
    interactions = op.interactions.OmniPath.get(
        genesymbols=True,
        fields=["is_stimulation", "is_inhibition", "consensus_direction"]
    )
    
    # 2. 过滤交互：保留有明确方向且有激活/抑制符号的边
    # 1 表示激活, -1 表示抑制, 0 表示未知或冲突
    interactions["weight"] = 0
    interactions.loc[interactions["is_stimulation"] == 1, "weight"] = 1
    interactions.loc[interactions["is_inhibition"] == 1, "weight"] = -1
    
    # 去除矛盾或无符号的边
    interactions = interactions[interactions["weight"] != 0]
    
    # 仅保留 source 和 target 都在我们的基因列表中的边 (如果提供了列表)
    if genes_list is not None:
        interactions = interactions[
            interactions["source_genesymbol"].isin(genes_list) & 
            interactions["target_genesymbol"].isin(genes_list)
        ]
        
    print(f"获取到 {len(interactions)} 条交互记录")
    return interactions

def load_cmap_data(gctx_path, gene_ids=None):
    """
    加载 CMAP (L1000) 数据。
    """
    print(f"正在加载 CMAP 数据: {gctx_path}...")
    
    # 使用 cmapPy 解析 GCTX 文件
    # rid 是行ID (基因ID), cid 是列ID (样本ID)
    if gene_ids:
        # 如果指定了基因，只加载这些基因的数据
        gctx_data = parse(gctx_path, rid=gene_ids)
    else:
        # 加载所有数据 (注意内存！)
        gctx_data = parse(gctx_path)
        
    data_df = gctx_data.data_df
    row_meta = gctx_data.row_metadata_df
    col_meta = gctx_data.col_metadata_df
    
    print(f"加载了 {data_df.shape[0]} 个基因, {data_df.shape[1]} 个样本")
    return data_df, row_meta, col_meta

def align_data(interactions, expression_df):
    """
    对齐 OmniPath 网络和 CMAP 表达数据，构建邻接矩阵。
    """
    # 1. 获取共同基因
    network_genes = set(interactions["source_genesymbol"]).union(set(interactions["target_genesymbol"]))
    expr_genes = set(expression_df.index)
    
    common_genes = sorted(list(network_genes.intersection(expr_genes)))
    print(f"对齐后共有 {len(common_genes)} 个共同基因")
    
    # 2. 过滤数据
    # 过滤表达矩阵
    expr_aligned = expression_df.loc[common_genes]
    
    # 过滤交互网络
    interactions_aligned = interactions[
        interactions["source_genesymbol"].isin(common_genes) & 
        interactions["target_genesymbol"].isin(common_genes)
    ]
    
    # 3. 构建邻接矩阵
    gene_to_idx = {gene: i for i, gene in enumerate(common_genes)}
    num_nodes = len(common_genes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for _, row in interactions_aligned.iterrows():
        src_idx = gene_to_idx[row["source_genesymbol"]]
        tgt_idx = gene_to_idx[row["target_genesymbol"]]
        weight = row["weight"]
        adj_matrix[src_idx, tgt_idx] = weight
        
    return adj_matrix, expr_aligned, common_genes

# 示例使用 (如果作为脚本运行)
if __name__ == "__main__":
    print("--- 演示模式：使用真实 OmniPath 网络 + 模拟表达数据 ---")
    
    # 1. 加载真实 OmniPath 数据
    # 这会从网络下载数据，可能需要一点时间
    real_interactions = load_omnipath_network()
    
    # 为了演示，我们随机选取网络中的部分基因来生成模拟表达数据
    # 在实际应用中，这里应该加载真实的 CMAP 数据
    unique_genes = list(set(real_interactions["source_genesymbol"]) | set(real_interactions["target_genesymbol"]))
    print(f"OmniPath 网络中包含 {len(unique_genes)} 个唯一基因")
    
    # 随机选择 100 个基因用于演示
    sample_genes = np.random.choice(unique_genes, 100, replace=False)
    
    # 2. 模拟 CMAP 表达数据 (使用上面选出的真实基因名)
    print("生成模拟表达数据 (替代真实的 CMAP .gctx 文件)...")
    mock_expr = pd.DataFrame(
        np.random.randn(100, 10), # 100个基因，10个样本
        index=sample_genes,
        columns=[f"SAMPLE_{i}" for i in range(10)]
    )
    
    # 3. 对齐
    adj, expr, genes = align_data(real_interactions, mock_expr)
    
    print("\n--- 结果统计 ---")
    print("对齐后的基因数量:", len(genes))
    print("邻接矩阵形状:", adj.shape)
    print("表达矩阵形状:", expr.shape)
    print(f"邻接矩阵非零边数量: {np.sum(adj != 0)}")
    
    print("\n提示：已成功加载真实 OmniPath 网络并与模拟数据对齐。")
    print("下一步：请下载 CMAP .gctx 文件，并使用 load_cmap_data() 替换上面的模拟数据生成部分。")
