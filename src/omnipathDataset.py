
import pandas as pd
import numpy as np
import json
import os
class OmniapthDataset:


    def __init__(self,interactions_path,mirna_path,tf_regulous_path):
        self.name = "Omnipath Dataset"
        self.interactions_path = interactions_path
        self.mirna_path = mirna_path
        self.tf_regulous_path = tf_regulous_path
        self.landmark_path = "data/landmark_genes.json"

    def build_gnn_from_tf_network(
        self,
        landmark_path="data/landmark_genes.json",
        landmark_genes=None,
        use_landmark_filter=True,
        mapping_path=None,
        include_non_landmark_nodes=False
     ):
        """
        读取 TF 调控网络，构建 GNN 邻接矩阵和节点索引。
        Args:
            tf_path: TF调控网络csv文件路径，需包含 source_genesymbol, target_genesymbol 两列。
            landmark_path: landmark 基因列表路径（json 或 txt）。
            landmark_genes: 预加载的 landmark 基因列表（保持顺序）。
        Returns:
            adj_matrix: (N, N) numpy array，邻接矩阵（有向，1表示有调控）
            node_list: 节点（基因）名称列表
            gene2idx: 节点名称到索引的映射
            edge_index: (2, E) numpy array，GNN常用边索引格式
        """
        df = pd.read_csv(self.tf_regulous_path)
        # 选择 genesymbol 列（优先）
        source_col = 'source_genesymbol' if 'source_genesymbol' in df.columns else 'source'
        target_col = 'target_genesymbol' if 'target_genesymbol' in df.columns else 'target'

        # 过滤 landmark 基因
        landmark_set = None
        symbol_to_entrez = {}

        # 如需要且未传入 landmark_genes，则从文件加载
        if use_landmark_filter and landmark_genes is None and os.path.exists(landmark_path):
            try:
                if self.landmark_path.endswith('.json'):
                    with open(landmark_path, 'r') as f:
                        genes_meta = json.load(f)
                    landmark_genes = [g['gene_symbol'] for g in genes_meta if 'gene_symbol' in g]
                else:
                    with open(landmark_path, 'r') as f:
                        landmark_genes = [line.strip() for line in f if line.strip()]
                print(f"已加载 landmark 基因列表: {len(landmark_genes)} (来自 {landmark_path})")
            except Exception as e:
                print(f"加载 landmark 基因列表失败: {e}")
        
        # 尝试加载映射表 (Symbol -> Entrez)
        mapping_source = mapping_path or landmark_path
        if os.path.exists(mapping_source):
            try:
                if mapping_source.endswith('.json'):
                    with open(mapping_source, 'r') as f:
                        genes_meta = json.load(f)
                    # 建立 Symbol -> Entrez ID (str) 映射
                    for g in genes_meta:
                        if 'gene_symbol' in g and 'entrez_id' in g:
                            symbol_to_entrez[g['gene_symbol']] = str(g['entrez_id'])
                else:
                    # 如果是 txt，无法建立映射，只能假设输入就是 Symbol
                    pass
            except Exception as e:
                print(f"加载映射失败: {e}")

        # 如果传入的 landmark_genes 是 Entrez ID (数字字符串)，我们需要将 OmniPath 的 Symbol 转为 Entrez
        # 判断 landmark_genes 的第一个元素是否是数字
        is_entrez = False
        if landmark_genes and len(landmark_genes) > 0:
            first_gene = str(landmark_genes[0])
            if first_gene.isdigit():
                is_entrez = True
                print(f"检测到 Landmark Genes 为 Entrez ID (e.g., {first_gene})，将尝试映射 OmniPath Symbol...")
        
        if is_entrez and not symbol_to_entrez:
            # 若 landmark 为 Entrez，但未加载到映射，尝试默认 JSON
            fallback_json = "data/landmark_genes.json"
            if os.path.exists(fallback_json):
                try:
                    with open(fallback_json, 'r') as f:
                        genes_meta = json.load(f)
                    for g in genes_meta:
                        if 'gene_symbol' in g and 'entrez_id' in g:
                            symbol_to_entrez[g['gene_symbol']] = str(g['entrez_id'])
                    print(f"使用备用映射: {fallback_json}")
                except Exception as e:
                    print(f"备用映射加载失败: {e}")

        if is_entrez and symbol_to_entrez:
            if include_non_landmark_nodes:
                # 允许非 landmark 节点：无法映射的保留原 symbol
                df['source_mapped'] = df[source_col].map(symbol_to_entrez).fillna(df[source_col].astype(str))
                df['target_mapped'] = df[target_col].map(symbol_to_entrez).fillna(df[target_col].astype(str))
                source_col = 'source_mapped'
                target_col = 'target_mapped'
                print(f"映射后 OmniPath 剩余记录: {len(df)}")
            else:
                # 只保留能映射到 Entrez 的边（等价于两端均在 landmark 映射中）
                df['source_entrez'] = df[source_col].map(symbol_to_entrez)
                df['target_entrez'] = df[target_col].map(symbol_to_entrez)
                df = df.dropna(subset=['source_entrez', 'target_entrez'])
                source_col = 'source_entrez'
                target_col = 'target_entrez'
                print(f"映射后 OmniPath 剩余记录: {len(df)}")
        
        if use_landmark_filter and landmark_genes:
            landmark_set = set(landmark_genes)
            # 只要 source 或 target 在 landmark 中即可保留
            df = df[df[source_col].isin(landmark_set) | df[target_col].isin(landmark_set)]

        # 只保留有向调控关系
        sources = df[source_col].astype(str)
        targets = df[target_col].astype(str)
        # 使用过滤后的图构建节点列表
        # 保留全部 landmark 节点，并加入与其相连的非 landmark 节点
        if use_landmark_filter and landmark_genes:
            if include_non_landmark_nodes:
                node_list = sorted(set(landmark_genes) | set(sources) | set(targets))
            else:
                node_list = list(landmark_genes)
        else:
            node_list = sorted(set(sources) | set(targets))
        gene2idx = {g: i for i, g in enumerate(node_list)}
        N = len(node_list)
        adj_matrix = np.zeros((N, N), dtype=np.float32)
        edge_list = []
        for src, tgt in zip(sources, targets):
            if src in gene2idx and tgt in gene2idx:
                i, j = gene2idx[src], gene2idx[tgt]
                adj_matrix[i, j] = 1
                edge_list.append((i, j))
        # GNN常用edge_index格式
        if not edge_list:
            raise ValueError("未找到有效的TF调控边，请检查数据和过滤条件。")
        edge_index = np.array(edge_list).T  # shape (2, E)
        print(f"TF网络节点数: {N}，边数: {len(edge_list)}")
        total_possible = N * N
        density = len(edge_list) / total_possible if total_possible > 0 else 0.0
        sparsity = 1.0 - density
        print(f"稀疏度(无自环): {sparsity:.6f} (密度: {density:.6f})")
        
        # --- 关键修复: 添加自环 (Self-Loops) ---
        # 保证每个节点至少与自己相连，避免 GNN 在孤立节点上聚合噪声
        np.fill_diagonal(adj_matrix, 1.0)
        print("已添加自环 (Self-Loops) 以增强图连通性。")
        density_with_loops = (len(edge_list) + N) / total_possible if total_possible > 0 else 0.0
        sparsity_with_loops = 1.0 - density_with_loops
        print(f"稀疏度(含自环): {sparsity_with_loops:.6f} (密度: {density_with_loops:.6f})")
        
        return adj_matrix, node_list, gene2idx, edge_index
        
        # 没有有效边时，返回空边索引但保持尺寸一致
        #print("警告: 未找到有效的TF调控边，返回空图结构。")
        # total_possible = N * N
        # density = 0.0
        # sparsity = 1.0 - density
        # print(f"稀疏度(无自环): {sparsity:.6f} (密度: {density:.6f})")
        # # 即使是空图，也应该加自环，退化为 MLP
        # np.fill_diagonal(adj_matrix, 1.0)
        # print("已添加自环 (Self-Loops) 以防止计算崩溃。")
        # density_with_loops = N / total_possible if total_possible > 0 else 0.0
        # sparsity_with_loops = 1.0 - density_with_loops
        # print(f"稀疏度(含自环): {sparsity_with_loops:.6f} (密度: {density_with_loops:.6f})")
        
        # edge_index = np.zeros((2, 0), dtype=int)
        # return adj_matrix, node_list, gene2idx, edge_index
        