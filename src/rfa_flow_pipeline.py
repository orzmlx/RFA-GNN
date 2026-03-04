from typing import Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import json
import pickle
import time

# ==========================================
# 模块一：数据预处理 & 图构建
# ==========================================

class RFADataManager:
    def __init__(self, omnipath_path, string_path, gene_info_path, landmark_path="data/landmark_genes.json", ppi_path=None, string_undirected_path=None):
        self.omnipath_path = omnipath_path
        self.string_path = string_path
        self.gene_info_path = gene_info_path
        self.landmark_path = landmark_path
        self.ppi_path = ppi_path
        self.string_undirected_path = string_undirected_path
        
        self.genes = []
        self.gene2idx = {}
        self.omni_edges = []   # (u, v, sign)
        self.string_edges = [] # (u, v, score)
        self.all_edges = []
        
    def load_data(self):
        print(">>> [RFADataManager] Loading Data...")
        
        # 1. Load Genes
        if os.path.exists(self.landmark_path):
            with open(self.landmark_path, 'r') as f:
                genes_meta = json.load(f)
            self.genes = [str(g['entrez_id']) for g in genes_meta]
            print(f"  Loaded {len(self.genes)} Landmark Genes.")
        elif os.path.exists(self.gene_info_path):
            df_genes = pd.read_csv(self.gene_info_path, sep='\t', dtype=str)
            self.genes = df_genes['pr_gene_id'].tolist()
            print(f"  Loaded {len(self.genes)} Full Genes.")
        else:
            print("  Warning: No gene list found.")
            
        self.gene2idx = {g: i for i, g in enumerate(self.genes)}
        self.num_genes = len(self.genes)
        target_set = set(self.genes)
        
        # Load Symbol -> Entrez Mapping
        symbol_to_entrez = {}
        if os.path.exists(self.landmark_path):
             with open(self.landmark_path, 'r') as f:
                genes_meta = json.load(f)
                for g in genes_meta:
                    if 'gene_symbol' in g and 'entrez_id' in g:
                        symbol_to_entrez[g['gene_symbol']] = str(g['entrez_id'])
        def process_omnipath_edge(path):
        # 2. Load OmniPath TF (Ground Truth Directions)
            if os.path.exists(path):
                df_omni = pd.read_csv(path)
                src_col = 'source_entrez' if 'source_entrez' in df_omni.columns else 'source_genesymbol'
                tgt_col = 'target_entrez' if 'target_entrez' in df_omni.columns else 'target_genesymbol'
                
                count = 0
                for _, row in df_omni.iterrows():
                    consensus_direction = row.get('consensus_direction', 0)
                    if consensus_direction != 1:
                        continue
                    u, v = str(row[src_col]), str(row[tgt_col])
                    if u in symbol_to_entrez: u = symbol_to_entrez[u]
                    if v in symbol_to_entrez: v = symbol_to_entrez[v]
                    
                    if u not in target_set or v not in target_set: continue
                    
                    sign = 0
                    if row.get('consensus_stimulation', 0) == 1: sign = 1
                    elif row.get('consensus_inhibition', 0) == 1: sign = -1
                # if sign == 0: sign = 1 
                    
                    self.omni_edges.append((u, v, sign))
                    count += 1
                print(f"  Loaded {count} OmniPath TF edges.")
        process_omnipath_edge(self.omnipath_path)    
        process_omnipath_edge(self.ppi_path)


        # 4. Load STRING (Undirected)
        if os.path.exists(self.string_path):
            df_string = pd.read_csv(self.string_path)
            src_col = 'source_entrez' if 'source_entrez' in df_string.columns else 'source'
            tgt_col = 'target_entrez' if 'target_entrez' in df_string.columns else 'target'
            score_col = 'score' if 'score' in df_string.columns else None
            weight_col = 'weight' if 'weight' in df_string.columns else None
            count = 0
            for _, row in df_string.iterrows():
                raw_score = row[score_col] 
                weight = row[weight_col] 
                if raw_score < 150: continue # Lowered threshold
                u, v = str(row[src_col]), str(row[tgt_col])
                if u not in target_set or v not in target_set: continue            
                w = raw_score / 1000.0 * weight
                
                self.string_edges.append((u, v, w))
                count += 1
            print(f"  Loaded {count} STRING edges (Score >= 150).")
            
        # 5. Load Extra STRING Undirected (As Bidirectional)
        if self.string_undirected_path and os.path.exists(self.string_undirected_path):
            df_extra = pd.read_csv(self.string_undirected_path)
            # Assuming columns like source_protein, target_protein, score
            
            # We need a map from ENSP -> Entrez
            # Let's build it from df_string if available, or gene_info?
            # df_string (string_interactions_mapped.csv) has 'source' (ENSP) and 'source_entrez'
            ensp2entrez = {}
            if os.path.exists(self.string_path):
                df_map = pd.read_csv(self.string_path)
                # Map source
                if 'source' in df_map.columns and 'source_entrez' in df_map.columns:
                    for _, row in df_map.iterrows():
                        ensp2entrez[str(row['source'])] = str(row['source_entrez'])
                        ensp2entrez[str(row['target'])] = str(row['target_entrez'])
            
            count_extra = 0
            for _, row in df_extra.iterrows():
                u_ensp = str(row['source_protein'])
                v_ensp = str(row['target_protein'])
                
                if u_ensp in ensp2entrez: u = ensp2entrez[u_ensp]
                else: continue
                
                if v_ensp in ensp2entrez: v = ensp2entrez[v_ensp]
                else: continue
                
                if u not in target_set or v not in target_set: continue
                
                raw_score = row['score']
                if raw_score < 150: continue # Same threshold
                w = raw_score / 1000.0
                
                # Add as bidirectional edge
                self.string_edges.append((u, v, w))
                self.string_edges.append((v, u, w))
                count_extra += 1
            print(f"  Loaded {count_extra} Extra Undirected edges (as bidirectional) from {self.string_undirected_path}.")

        #只按source和target去重，分成两部分分别去重，一部分是score是负数的，一部分是正数的，保留绝对值分数最大的意向如果重复

        self.all_edges = self.omni_edges + self.string_edges
        pos_edge = [e for e in self.all_edges if e[2] > 0]
        neg_edge = [e for e in self.all_edges if e[2] < 0]
        pos_edge = sorted(pos_edge, key=lambda x: (x[0], x[1]), reverse=True)
        neg_edge = sorted(neg_edge, key=lambda x: (x[0], x[1]), reverse=False)
        # 使用字典去重，保留权重绝对值最大的边
        # pos_edge (weight > 0): 保留 max(weight)
        pos_dict = {}
        for u, v, w in pos_edge:
            if (u, v) not in pos_dict or w > pos_dict[(u, v)]:
                pos_dict[(u, v)] = w
        pos_edge = [(u, v, w) for (u, v), w in pos_dict.items()]

        # neg_edge (weight < 0): 保留 min(weight) 即绝对值最大
        neg_dict = {}
        for u, v, w in neg_edge:
            if (u, v) not in neg_dict or w < neg_dict[(u, v)]:
                neg_dict[(u, v)] = w
        neg_edge = [(u, v, w) for (u, v), w in neg_dict.items()]
        self.all_edges = sorted(pos_edge + neg_edge, key=lambda x: (x[0], x[1]), reverse=True)
        print(f"  Total {len(self.all_edges)} edges after merging.")
        


    def get_hetero_graph_data(self):
        omni_src, omni_tgt = [], []
        for u, v, _ in self.omni_edges:
            omni_src.append(self.gene2idx[u])
            omni_tgt.append(self.gene2idx[v])
            
        string_src, string_tgt = [], []
        for u, v, _ in self.string_edges:
            string_src.append(self.gene2idx[u])
            string_tgt.append(self.gene2idx[v])
            string_src.append(self.gene2idx[v])
            string_tgt.append(self.gene2idx[u])
            
        return (np.array(omni_src), np.array(omni_tgt)), (np.array(string_src), np.array(string_tgt))

# ==========================================
# 模块二：GNN 方向预测模型 (Heterogeneous)
# ==========================================

class HeteroDirectionPredictor(keras.Model):
    def __init__(self, num_nodes, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.node_emb = layers.Embedding(num_nodes, hidden_dim)
        self.W_self = layers.Dense(hidden_dim, use_bias=False)
        self.W_omni = layers.Dense(hidden_dim, use_bias=False)
        self.W_string = layers.Dense(hidden_dim, use_bias=False)
        self.act = layers.Activation('relu')
        self.dropout = layers.Dropout(dropout)
        self.classifier = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(4, activation='softmax')
        ])
        
    def call(self, inputs):
        node_ids, omni_indices, string_indices, query_edges = inputs
        h = self.node_emb(node_ids)
        
        adj_omni = omni_indices
        adj_string = string_indices
        
        h_self = self.W_self(h)
        h_omni = self.W_omni(h)
        h_string = self.W_string(h)
        
        agg_omni = tf.sparse.sparse_dense_matmul(adj_omni, h_omni)
        agg_string = tf.sparse.sparse_dense_matmul(adj_string, h_string)
        
        h_next = self.act(h_self + agg_omni + agg_string)
        h_next = self.dropout(h_next)
        
        u_idx = query_edges[:, 0]
        v_idx = query_edges[:, 1]
        h_u = tf.gather(h_next, u_idx)
        h_v = tf.gather(h_next, v_idx)
        
        return self.classifier(tf.concat([h_u, h_v], axis=-1))

def train_predictor(data_manager, epochs=10):
    print(">>> [Train] Training Direction Predictor...")
    omni_idx, string_idx = data_manager.get_hetero_graph_data()
    N = data_manager.num_genes
    
    def make_sparse(idxs, n):
        if len(idxs[0]) == 0:
            return tf.sparse.SparseTensor([[0,0]], [0.0], [n, n])
        indices = np.stack(idxs, axis=1).astype(np.int64)
        values = np.ones(len(idxs[0]), dtype=np.float32)
        return tf.sparse.reorder(tf.sparse.SparseTensor(indices, values, [n, n]))

    sp_omni = make_sparse(omni_idx, N)
    sp_string = make_sparse(string_idx, N)
    
    train_pairs, train_labels = [], []
    for u, v, sign in data_manager.omni_edges:
        if u not in data_manager.gene2idx or v not in data_manager.gene2idx: continue
        uid, vid = data_manager.gene2idx[u], data_manager.gene2idx[v]
        if sign == 1:
            train_pairs.append([uid, vid]); train_labels.append(0)
            train_pairs.append([vid, uid]); train_labels.append(2)
        else:
            train_pairs.append([uid, vid]); train_labels.append(1)
            train_pairs.append([vid, uid]); train_labels.append(3)
            
    if not train_pairs:
        print("Warning: No training pairs for direction predictor!")
        return None, sp_omni, sp_string
        
    train_pairs = np.array(train_pairs, dtype=np.int32)
    train_labels = np.array(train_labels, dtype=np.int32)
    
    model = HeteroDirectionPredictor(N)
    optimizer = keras.optimizers.Adam(1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    ds = tf.data.Dataset.from_tensor_slices((train_pairs, train_labels)).shuffle(10000).batch(256)
    all_nodes = tf.range(N, dtype=tf.int32)
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for batch_pairs, batch_y in ds:
            with tf.GradientTape() as tape:
                logits = model([all_nodes, sp_omni, sp_string, batch_pairs])
                loss = loss_fn(batch_y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss
            steps += 1
        print(f"  Epoch {epoch+1}: Loss = {total_loss/steps:.4f}")
        
    return model, sp_omni, sp_string

# ==========================================
# 模块三 & 四：RFA Solver & Subgraph
# ==========================================

class RFASolver:
    def __init__(self, nodes, edges, alpha=0.85):
        self.nodes = nodes
        self.node2idx = {n: i for i, n in enumerate(nodes)}
        self.edges = edges 
        self.alpha = alpha
        self.F = None
        
    def solve(self):
        print(">>> [RFA] Solving Flow Matrix...")
        N = len(self.nodes)
        rows, cols, data = [], [], []
        for u, v, sign in self.edges:
            if u in self.node2idx and v in self.node2idx:
                rows.append(self.node2idx[u])
                cols.append(self.node2idx[v])
                data.append(float(sign))
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        
        abs_A = np.abs(A)
        out_degree = np.array(abs_A.sum(axis=1)).flatten()
        out_degree[out_degree == 0] = 1.0
        U = sp.diags(1.0 / out_degree).dot(A)
        S = self.alpha * U
        
        print("  Approximating Inverse with Power Series (k=8)...")
        F = sp.eye(N, format='csr')
        S_pow = S.copy()
        for k in range(8):
            F = F + S_pow
            S_pow = S_pow.dot(S)
        self.F = F
        print("  Flow Matrix F computed.")
        return F

def extract_subgraph(rfa_solver, source, target, max_nodes=30):
    if source not in rfa_solver.node2idx or target not in rfa_solver.node2idx: return None
    src_idx, tgt_idx = rfa_solver.node2idx[source], rfa_solver.node2idx[target]
    F = rfa_solver.F
    
    total_flow = F[src_idx, tgt_idx]
    if abs(total_flow) < 1e-9: return None
    
    current_nodes = {src_idx, tgt_idx}
    F_csc = F.tocsc()
    row_AC = F[src_idx, :].toarray().flatten()
    col_CB = F_csc[:, tgt_idx].toarray().flatten()
    scores = np.abs(row_AC * col_CB) / (abs(total_flow) + 1e-9)
    candidate_indices = np.argsort(scores)[::-1]
    
    added = 0
    for idx in candidate_indices:
        if idx in current_nodes: continue
        if added >= max_nodes: break
        current_nodes.add(idx)
        added += 1
        
    final_node_list = [rfa_solver.nodes[i] for i in current_nodes]
    return {"source": source, "target": target, "nodes": final_node_list, "flow_ratio": 1.0}

def run_pipeline():
    OMNI_PATH = "data/omnipath/omnipath_tf_regulons.csv"
    PPI_PATH = "data/omnipath/omnipath_interactions.csv"
    STRING_PATH = "data/string_interactions_mapped.csv"
    STRING_UNDIRECTED = "data/string_undirected.csv"
    GENE_INFO = "data/GSE92742_Broad_LINCS_gene_info.txt"
    LANDMARK = "data/landmark_genes.json"
    
    manager = RFADataManager(OMNI_PATH, STRING_PATH, GENE_INFO, LANDMARK, PPI_PATH, STRING_UNDIRECTED)
    manager.load_data()
    
    if manager.num_genes == 0:
        raise Exception("No genes loaded.")
        
    model, sp_omni, sp_string = train_predictor(manager, epochs=5)
    
    if model is None:
        raise Exception("No training data for direction predictor.")
   
    print(">>> Predicting STRING/Undirected PPI directions...")
    string_src, string_tgt = [], []
    valid_edges = []
    for u, v, w in manager.string_edges:
        if u in manager.gene2idx and v in manager.gene2idx:
            string_src.append(manager.gene2idx[u])
            string_tgt.append(manager.gene2idx[v])
            valid_edges.append((u, v, w))
    
    batch_size = 1024
    num_edges = len(string_src)
    all_nodes = tf.range(manager.num_genes, dtype=tf.int32)
    predicted_edges = []
    
    print(f"  Total STRING edges to predict: {num_edges}")
    
    if num_edges > 0:
        for i in range(0, num_edges, batch_size):
            end = min(i+batch_size, num_edges)
            batch_pairs = np.stack([string_src[i:end], string_tgt[i:end]], axis=1)
            logits = model([all_nodes, sp_omni, sp_string, batch_pairs])
            probs = tf.nn.softmax(logits).numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            
            for j, pred in enumerate(preds):
                # if confs[j] < 0.6: continue 
                # 暂时注释掉置信度过滤，看看有没有边
                # if confs[j] < 0.5: continue
                u, v, w = valid_edges[i+j]
            
                # Check 900
                # if w < 0.9: continue
                if w < 0.7: continue
                
                # 放宽置信度要求
                # if confs[j] < 0.5: continue
                # 如果还是 0，那就不过滤置信度了，直接相信模型预测
                
                if pred == 0: predicted_edges.append((u, v, w)) # u->v
                elif pred == 1: predicted_edges.append((u, v, w)) # u->v (ignore sign for weight)
                elif pred == 2: predicted_edges.append((v, u, w)) # v->u
                elif pred == 3: predicted_edges.append((v, u, w)) # v->u
    
    print(f"  Predicted {len(predicted_edges)} directed edges.")
    final_edges = manager.omni_edges + predicted_edges

    solver = RFASolver(manager.genes, final_edges)
    F = solver.solve()
    
    print(">>> Saving Directed Edges for GNN...")
    with open("data/rfa_directed_edges.csv", "w") as f:
        f.write("source,target,weight\n")
        for u, v, w in final_edges:
            f.write(f"{u},{v},{w}\n")
    print("Done.")

if __name__ == "__main__":
    run_pipeline()
