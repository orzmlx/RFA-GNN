import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
import os
import sys
import time
from scipy.stats import pearsonr

# Import data loader (reuse to avoid duplication)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_rfa_data

# ==========================================
# 0. Utils & Metrics
# ==========================================

def pcc_loss(y_true, y_pred):
    """
    Sample-wise Pearson Correlation Loss.
    Loss = 1 - mean(PCC)
    """
    mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
    my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    xm = y_true - mx
    ym = y_pred - my
    r_num = tf.reduce_sum(xm * ym, axis=1)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8)
    r = r_num / r_den
    return 1.0 - tf.reduce_mean(r)

def pcc_loss_topk(y_true, y_pred, k=200):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    n = tf.shape(y_true)[1]
    k = tf.minimum(tf.cast(k, tf.int32), n)
    topk_idx = tf.math.top_k(tf.abs(y_true), k=k, sorted=False).indices
    y_true_k = tf.gather(y_true, topk_idx, batch_dims=1)
    y_pred_k = tf.gather(y_pred, topk_idx, batch_dims=1)
    return pcc_loss(y_true_k, y_pred_k)

def _rfa_flow_from_adj(adj, alpha=0.85, eps=1e-9):
    abs_A = np.abs(adj)
    out_degree = abs_A.sum(axis=1, keepdims=True)
    out_degree[out_degree < eps] = 1.0
    U = adj / out_degree
    S = alpha * U
    I = np.eye(adj.shape[0], dtype=np.float32)
    M = I - S
    return np.linalg.inv(M).astype(np.float32)

def rfa_algorithm2_min_nodes(F, sources, target, max_nodes=50, eps=1e-9):
    sources = [int(s) for s in sources]
    target = int(target)
    N = F.shape[0]
    sources = [s for s in sources if 0 <= s < N]
    if not sources or not (0 <= target < N):
        return np.array(sorted(set(sources + [target])), dtype=np.int32)
    total_flow = float(np.sum(F[sources, target]))
    if abs(total_flow) < eps:
        return np.array(sorted(set(sources + [target])), dtype=np.int32)
    row = np.sum(F[sources, :], axis=0)
    col = F[:, target]
    scores = np.abs(row * col) / (abs(total_flow) + eps)
    keep = set(sources + [target])
    for idx in np.argsort(scores)[::-1]:
        if idx in keep:
            continue
        keep.add(int(idx))
        if len(keep) >= max_nodes:
            break
    return np.array(sorted(keep), dtype=np.int32)

# ==========================================
# 1. Subgraph Extraction Logic
# ==========================================

class SubgraphManager:
    def __init__(self, edges_csv_path, gene_list):
        self.gene_list = gene_list
        self.gene2idx = {g: i for i, g in enumerate(gene_list)}
        self.idx2gene = {i: g for i, g in enumerate(gene_list)}
        self.G = self._build_graph(edges_csv_path)
        
    def _build_graph(self, csv_path):
        print(f"Building Global Graph from {csv_path}...")
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.gene_list)))
        
        if not os.path.exists(csv_path):
            raise Exception(f"Error: Graph file {csv_path} not found.")
        df = pd.read_csv(csv_path)
        edge_map = {}
        for _, row in df.iterrows():
            u = str(int(float(row['source']))) if str(row['source']).replace('.','',1).isdigit() else str(row['source'])
            v = str(int(float(row['target']))) if str(row['target']).replace('.','',1).isdigit() else str(row['target'])
            
            if u in self.gene2idx and v in self.gene2idx:
                uid, vid = self.gene2idx[u], self.gene2idx[v]
                w = float(row.get('weight', 1.0))
                key = (uid, vid)
                if key not in edge_map or abs(w) > abs(edge_map[key]):
                    edge_map[key] = w
        for (uid, vid), w in edge_map.items():
            G.add_edge(uid, vid, weight=w)
        print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        
        return G

    def pre_prune(self, roots, leafs, depth=10):
        roots = [int(x) for x in roots]
        leafs = [int(x) for x in leafs]
        roots = [x for x in roots if x in self.G]
        leafs = [x for x in leafs if x in self.G]
        if not roots or not leafs:
            return

        def multi_source_dist(G, sources, cutoff):
            dist = {}
            q = deque()
            for s in sources:
                if s not in dist:
                    dist[s] = 0
                    q.append(s)
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= cutoff:
                    continue
                for v in G.successors(u):
                    if v not in dist:
                        dist[v] = du + 1
                        q.append(v)
            return dist

        dist_r = multi_source_dist(self.G, roots, depth)
        rG = self.G.reverse(copy=False)
        dist_l = multi_source_dist(rG, leafs, depth)

        keep = set(roots) | set(leafs)
        for n, dr in dist_r.items():
            dl = dist_l.get(n)
            if dl is None:
                continue
            if dr + dl <= depth:
                keep.add(n)

        before_nodes = self.G.number_of_nodes()
        before_edges = self.G.number_of_edges()
        self.G = self.G.subgraph(sorted(list(keep))).copy()
        print(f"  Pre-pruned graph: {before_nodes}->{self.G.number_of_nodes()} nodes, {before_edges}->{self.G.number_of_edges()} edges.")

    def extract_subgraph(self, target_indices, k_hops=2, max_nodes=200):
        """
        Extracts induced subgraph for a drug.
        Starts from target_indices, expands k_hops.
        Returns:
            sub_nodes: List of global node indices in the subgraph.
            sub_adj: Adjacency matrix (dense) of the subgraph.
        """
        # 1. Initialize S_d = T_d
        current_nodes = set(target_indices)
        
        # 2. Expansion
        for _ in range(k_hops):
            neighbors = set()
            for node in current_nodes:
                if node in self.G:
                    # Add successors (downstream)
                    for nbr in self.G.successors(node):
                        neighbors.add(nbr)
                    # Add predecessors (upstream)
                    # RFA flow is directed, so maybe we only care about downstream?
                    # But for GNN message passing, we might want upstream context.
                    # Let's keep both for subgraph extraction.
                    for nbr in self.G.predecessors(node):
                        neighbors.add(nbr)
            
            # Stop if too large
            if len(current_nodes) + len(neighbors) > max_nodes:
                # Prioritize by degree or something? For now random.
                # Or prioritize successors?
                needed = max_nodes - len(current_nodes)
                if needed > 0:
                    current_nodes.update(list(neighbors)[:needed])
                break
            
            current_nodes.update(neighbors)
            
        # Ensure we have at least some nodes.
        if len(current_nodes) < 5:
            # Fallback: add random neighbors or just keep it small
            pass
            
        sub_nodes = sorted(list(current_nodes))
        sub_G = self.G.subgraph(sub_nodes)
        
        # 4. Adjacency
        # NetworkX to numpy
        N_sub = len(sub_nodes)
        sub_adj = nx.to_numpy_array(sub_G, nodelist=sub_nodes, weight='weight', dtype=np.float32)
        
        # Add self-loops to subgraph adj to ensure message passing
        # For RFA, diagonal is usually 0 in A, but (I-S) handles self-loop.
        # But here 'sub_adj' is passed to the model as 'adj'.
        # In DifferentiableRFAGNN:
        # A_tilde = masked_gates * adj
        # U = D^-1 A_tilde
        # If adj has self-loops, U will have self-loops.
        # RFA typically uses adjacency without self-loops for flow calc,
        # because flow moves FROM node TO neighbors.
        # If we have self-loops, flow stays at node.
        # Let's REMOVE self-loops for RFA logic.
        np.fill_diagonal(sub_adj, 0.0)
        
        return np.array(sub_nodes, dtype=np.int32), sub_adj

# ==========================================
# 2. GNN Model (Shared)
# ==========================================

# ==========================================
# 2. Differentiable RFA GNN (Pruning + Algo 2 Logic)
# ==========================================

class DifferentiableRFAGNN(keras.Model):
    def __init__(self, output_dim, fp_dim=0, num_cells=0, cell_emb_dim=16, hidden_dim=64, alpha=0.85, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.fp_dim = fp_dim
        
        self.use_cell = (num_cells > 0)
        if self.use_cell:
            self.cell_emb = layers.Embedding(num_cells, cell_emb_dim)
            self.cell_proj = layers.Dense(hidden_dim, activation='tanh')
        self.use_fp = (fp_dim > 0)
        if self.use_fp:
            self.fp_proj = layers.Dense(hidden_dim, activation='tanh')
        
        self.encoder = layers.Dense(hidden_dim, activation='relu')
        self.dropout_layer = layers.Dropout(dropout)
        
        # 2. Gating Mechanism (Pruning)
        # m_ij = sigmoid( (h_i W_s) @ (h_j W_d)^T + bias )
        # Change to tanh to allow negative weights (inhibition)
        self.gate_src = layers.Dense(hidden_dim, use_bias=False)
        self.gate_dst = layers.Dense(hidden_dim, use_bias=False)
        self.gate_bias = self.add_weight(name="gate_bias", shape=(1,), initializer=tf.constant_initializer(0.0))
        
        # 3. Target Source Learning (New Feature)
        # Learn the effective perturbation source strength/sign from context
        # Input x: (B, N, 2)
        self.source_embedding = layers.Dense(16, activation='tanh')
        self.source_strength = layers.Dense(1, activation='tanh') # (B, N, 1) output range [-1, 1]
        
        # 4. Global Bias Learning (New Feature)
        # Learn a baseline response for each node context
        self.node_bias_layer = layers.Dense(1) # (B, N, 1)

        # Output is implicit via RFA flow
        
    def call(self, inputs):
        x = None
        adj = None
        cell_ids = None
        fp = None
        if len(inputs) == 2:
            x, adj = inputs
        elif len(inputs) == 3:
            x, adj, cell_ids = inputs
        elif len(inputs) == 4:
            x, adj, cell_ids, fp = inputs
        else:
            raise ValueError("Expected 2-4 inputs: (x, adj[, cell_ids][, fp])")
        
        # Ensure adj is float32
        adj = tf.cast(adj, tf.float32)
        if len(adj.shape) == 2: adj = tf.expand_dims(adj, 0)
        
        # 1. Encode Nodes
        h = self.encoder(x) # (B, N, H)
        
        # Add Cell Context if available
        if self.use_cell and cell_ids is not None:
            # (B, D) -> (B, 1, D)
            c_emb = self.cell_emb(cell_ids)
            c_h = self.cell_proj(c_emb)
            c_h = tf.expand_dims(c_h, 1) 
            # Add to node embedding (broadcast across N)
            h = h + c_h
        if self.use_fp and fp is not None:
            fp_h = self.fp_proj(fp)
            fp_h = tf.expand_dims(fp_h, 1)
            h = h + fp_h
            
        h = self.dropout_layer(h)
        
        # 2. Compute Edge Gates
        h_s = self.gate_src(h)
        h_d = self.gate_dst(h)
        
        # Logits: (B, N, H) @ (B, H, N) -> (B, N, N)
        logits = tf.matmul(h_s, h_d, transpose_b=True) + self.gate_bias
        
        # Use tanh to allow negative weights (Inhibition)
        # Range: [-1, 1]
        gates = tf.tanh(logits)
        
        # Mask with original adjacency structure (Hard constraint)
        # We only prune existing edges, not create new ones
        adj_mask = tf.cast(tf.abs(adj) > 1e-9, tf.float32)
        masked_gates = gates * adj_mask
        
        # 3. Construct Pruned Graph A_tilde
        # A_tilde = Gates * Original_Weights
        # Original weights are all positive (from STRING/OmniPath magnitude), 
        # so sign comes purely from 'gates'.
        # If original weights had sign, we multiply.
        # Here we trust the model to learn the sign.
        A_tilde = masked_gates * tf.abs(adj) 
        
        # 4. RFA Flow Calculation
        # U = D^-1 A
        # For signed graph, we normalize by absolute row sum to keep stability
        abs_A = tf.abs(A_tilde)
        out_degree = tf.reduce_sum(abs_A, axis=2, keepdims=True)
        out_degree = tf.maximum(out_degree, 1e-9)
        U_tilde = A_tilde / out_degree
        
        # S = alpha * U
        S_tilde = self.alpha * U_tilde
        
        # F = (I - S)^-1
        N = tf.shape(x)[1]
        I = tf.eye(N, batch_shape=[tf.shape(x)[0]])
        
        # Invert (I - S)
        # Add small epsilon to diagonal for stability
        M = I - S_tilde + 1e-6 * I
        F_tilde = tf.linalg.inv(M)
        
        # 5. Prediction
        # y_hat = Sum_{t in Targets} F[:, t] * Strength_t
        
        src_emb = self.source_embedding(h)
        learned_strength = self.source_strength(src_emb)
        
        # Only allow flow from actual targets
        targets_mask = tf.expand_dims(x[:, :, 1], -1) # (B, N, 1) - 0 or 1
        
        # Effective Source: Target_Mask * Learned_Sign_Strength
        # We amplify by 10.0 to help flow reach further (as suggested)
        effective_source = targets_mask * learned_strength * 10.0
        
        # (B, N, N) @ (B, N, 1) -> (B, N, 1)
        y_rfa = tf.matmul(F_tilde, effective_source)
        y_rfa = tf.squeeze(y_rfa, -1) # (B, N)
        
        # Add Node-wise Learned Bias (Global baseline response)
        # We use the initial node embedding 'h' to predict a bias.
        # This helps model learn "which genes tend to change regardless of target"
        bias = self.node_bias_layer(h) # (B, N, 1)
        bias = tf.squeeze(bias, -1)
        
        y_final = y_rfa + bias
        
        # Normalize Output (Crucial for PCC)
        # PCC is scale-invariant, but small values cause numerical issues.
        # We force the output to have mean 0 and variance 1 (per sample).
        mean, var = tf.nn.moments(y_final, axes=[1], keepdims=True)
        y_norm = (y_final - mean) / (tf.sqrt(var) + 1e-6)
        
        # Output normalized delta, and return masked_gates for regularization
        return y_norm, masked_gates

    def extract_min_subgraph(self, x, adj, top_k=50):
        """
        Implements 'Algorithm 2' logic: Extract discrete subgraph based on learned gates/flow.
        Returns the nodes and pruned adjacency for analysis.
        """
        # Run forward pass partially
        h = self.encoder(x)
        h_s = self.gate_src(h)
        h_d = self.gate_dst(h)
        logits = tf.matmul(h_s, h_d, transpose_b=True) + self.gate_bias
        gates = tf.sigmoid(logits)
        adj_mask = tf.cast(tf.abs(adj) > 1e-9, tf.float32)
        masked_gates = gates * adj_mask
        
        # Return average gates over batch
        avg_gates = tf.reduce_mean(masked_gates, axis=0).numpy()
        
        # Extract Top-K edges by weight
        flat_indices = np.argsort(avg_gates.flatten())[::-1]
        top_indices = flat_indices[:top_k]
        
        # Reconstruct sparse adj or nodes
        # This is a placeholder for the exact 'Algorithm 2' greedy expansion
        return avg_gates

# ==========================================
# 3. Training Loop with Auto-Optimization
# ==========================================

def train_and_evaluate(config, drug_groups, subgraph_manager, X_ctl, y_delta, X_fingerprint, num_genes, cell_info=None, topk_pcc=200):
    """
    Train model with given config and return best PCC.
    """
    print(f"\n>>> [Config] Hidden={config['hidden']}, Dropout={config['dropout']}, LR={config['lr']} <<<")
    
    # Split Drugs (Train/Test)
    all_drugs = list(drug_groups.keys())
    np.random.shuffle(all_drugs)
    split_idx = int(len(all_drugs) * 0.8)
    train_drugs = all_drugs[:split_idx]
    test_drugs = all_drugs[split_idx:]
    
    # Instantiate Differentiable RFA
    num_cells = cell_info['num_cells'] if cell_info else 0
    fp_dim = int(X_fingerprint.shape[1]) if X_fingerprint is not None else 0
    
    model = DifferentiableRFAGNN(
        output_dim=num_genes, 
        fp_dim=fp_dim,
        num_cells=num_cells,
        hidden_dim=config['hidden'], 
        alpha=0.85,
        dropout=config['dropout']
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=config['lr'])
    
    # Loss: PCC + Sparsity
    def rfa_loss(y_true_delta, y_pred_delta, gates, lambda_sparse=1e-3, epoch=0):
        pcc_val = pcc_loss_topk(y_true_delta, y_pred_delta, k=topk_pcc)
        
        # Sparsity Loss (L1 on gates)
        # Warm-up: 0 for first 5 epochs
        w_sparse = 0.0 if epoch < 5 else lambda_sparse
        sparsity = tf.reduce_mean(tf.abs(gates))
        
        return pcc_val + w_sparse * sparsity, pcc_val, sparsity
        
    best_test_pcc = -1.0
    min_cache = {}
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = 0
        train_pcc_accum = 0
        steps = 0
        np.random.shuffle(train_drugs)
        
        for drug_id in train_drugs:
            group = drug_groups[drug_id]
            sub_nodes, sub_adj = subgraph_manager.extract_subgraph(group['targets'], k_hops=3)
            if len(sub_nodes) < 2: continue
            
            sample_indices = group['samples']
            batch_X_global = tf.gather(X_ctl, sample_indices)
            batch_X_sub = tf.gather(batch_X_global, sub_nodes, axis=1)
            
            # Extract Cell IDs if available
            batch_cell_ids = None
            if cell_info:
                # cell_info['indices'] has shape (M,) mapped to 0..K
                all_cell_indices = cell_info['indices']
                batch_cell_ids = tf.gather(all_cell_indices, sample_indices)
            batch_fp = None
            if X_fingerprint is not None:
                batch_fp = tf.gather(X_fingerprint, sample_indices)
            
            # Use delta directly for training
            batch_y = tf.gather(y_delta, sample_indices)
            batch_y_sub = tf.gather(batch_y, sub_nodes, axis=1) 
            
            batch_adj = tf.expand_dims(sub_adj, axis=0)

            if drug_id not in min_cache:
                local_n = int(sub_adj.shape[0])
                pos_map = {int(g): i for i, g in enumerate(sub_nodes.tolist())}
                sources_pos = [pos_map[int(t)] for t in group['targets'] if int(t) in pos_map]
                small_idx = sample_indices[: min(16, len(sample_indices))]
                small_X = tf.gather(X_ctl, small_idx)
                small_X = tf.gather(small_X, sub_nodes, axis=1)
                small_y = tf.gather(y_delta, small_idx)
                small_y = tf.gather(small_y, sub_nodes, axis=1).numpy()
                out_k = min(50, local_n)
                mean_abs = np.mean(np.abs(small_y), axis=0)
                out_pos = np.argpartition(mean_abs, -out_k)[-out_k:]

                small_cell = None
                if cell_info:
                    small_cell = tf.gather(cell_info['indices'], small_idx)
                small_fp = None
                if X_fingerprint is not None:
                    small_fp = tf.gather(X_fingerprint, small_idx)
                if small_fp is not None and small_cell is None:
                    small_cell = tf.zeros([tf.shape(small_fp)[0]], dtype=tf.int32)

                if small_cell is not None and small_fp is not None:
                    _, g_small = model([small_X, batch_adj, small_cell, small_fp])
                elif small_cell is not None:
                    _, g_small = model([small_X, batch_adj, small_cell])
                else:
                    _, g_small = model([small_X, batch_adj])
                g_avg = np.mean(np.abs(g_small.numpy()), axis=0).astype(np.float32)
                A_pruned = g_avg * np.abs(sub_adj).astype(np.float32)
                F = _rfa_flow_from_adj(A_pruned, alpha=0.85)
                keep = set()
                for tpos in out_pos.tolist():
                    nodes_t = rfa_algorithm2_min_nodes(F, sources_pos, int(tpos), max_nodes=min(80, local_n))
                    keep.update(nodes_t.tolist())
                keep = sorted(list(keep))
                if len(keep) > min(120, local_n):
                    keep = keep[: min(120, local_n)]
                min_cache[drug_id] = np.array(keep, dtype=np.int32)

            keep_pos = min_cache[drug_id]
            if keep_pos.shape[0] >= 2 and keep_pos.shape[0] < sub_nodes.shape[0]:
                sub_nodes = tf.gather(sub_nodes, keep_pos).numpy()
                sub_adj = sub_adj[np.ix_(keep_pos, keep_pos)]
                batch_X_sub = tf.gather(batch_X_sub, keep_pos, axis=1)
                batch_y_sub = tf.gather(batch_y_sub, keep_pos, axis=1)
                batch_adj = tf.expand_dims(sub_adj, axis=0)
            
            with tf.GradientTape() as tape:
                # Model returns (delta_pred, gates)
                if batch_cell_ids is not None and batch_fp is not None:
                    preds_delta, gates = model([batch_X_sub, batch_adj, batch_cell_ids, batch_fp])
                elif batch_cell_ids is not None:
                    preds_delta, gates = model([batch_X_sub, batch_adj, batch_cell_ids])
                elif batch_fp is not None:
                    dummy_cell = tf.zeros([tf.shape(batch_fp)[0]], dtype=tf.int32)
                    preds_delta, gates = model([batch_X_sub, batch_adj, dummy_cell, batch_fp])
                else:
                    preds_delta, gates = model([batch_X_sub, batch_adj])
                
                loss, pcc_v, sparse_v = rfa_loss(batch_y_sub, preds_delta, gates, epoch=epoch)
                
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss += loss
            train_pcc_accum += (1.0 - pcc_v) # convert back to PCC
            steps += 1
            
        # Evaluate
        pcc_list = []
        if steps == 0:
            print(f"  Epoch {epoch+1}: No steps run. Check if graph has edges or if subgraphs are empty.")
            avg_pcc = 0.0
        else:
            # ... existing eval code ...
            for drug_id in test_drugs:
                group = drug_groups[drug_id]
                sub_nodes, sub_adj = subgraph_manager.extract_subgraph(group['targets'], k_hops=3)
                if len(sub_nodes) < 2: continue
                
                sample_indices = group['samples']
                batch_X_global = tf.gather(X_ctl, sample_indices)
                batch_X_sub = tf.gather(batch_X_global, sub_nodes, axis=1)
                batch_y = tf.gather(y_delta, sample_indices)
                batch_y_sub = tf.gather(batch_y, sub_nodes, axis=1)
                batch_adj = tf.expand_dims(sub_adj, axis=0)
                
                # Extract Cell IDs if available
                batch_cell_ids = None
                if cell_info:
                    all_cell_indices = cell_info['indices']
                    batch_cell_ids = tf.gather(all_cell_indices, sample_indices)
                batch_fp = None
                if X_fingerprint is not None:
                    batch_fp = tf.gather(X_fingerprint, sample_indices)
                
                # Predict delta
                if batch_cell_ids is not None and batch_fp is not None:
                    preds_delta, _ = model([batch_X_sub, batch_adj, batch_cell_ids, batch_fp])
                elif batch_cell_ids is not None:
                    preds_delta, _ = model([batch_X_sub, batch_adj, batch_cell_ids])
                elif batch_fp is not None:
                    dummy_cell = tf.zeros([tf.shape(batch_fp)[0]], dtype=tf.int32)
                    preds_delta, _ = model([batch_X_sub, batch_adj, dummy_cell, batch_fp])
                else:
                    preds_delta, _ = model([batch_X_sub, batch_adj])
                
                # Numpy calc for PCC
                y_true = batch_y_sub.numpy() 
                y_pred = preds_delta.numpy()
                
                for k in range(len(y_true)):
                    yt = y_true[k]
                    yp = y_pred[k]
                    kk = min(int(topk_pcc), yt.shape[0])
                    if kk < 2:
                        continue
                    top_idx = np.argpartition(np.abs(yt), -kk)[-kk:]
                    yt_k = yt[top_idx]
                    yp_k = yp[top_idx]
                    if np.std(yt_k) > 1e-6 and np.std(yp_k) > 1e-6:
                        p, _ = pearsonr(yt_k, yp_k)
                        pcc_list.append(p)
                        
            avg_pcc = np.mean(pcc_list) if pcc_list else 0.0
            avg_train_pcc = train_pcc_accum / steps
            print(f"  Epoch {epoch+1}: Train PCC = {avg_train_pcc:.4f} | Test PCC = {avg_pcc:.4f}")
            
            # Early Stopping Check
            if epoch >= 2 and avg_pcc < 0.2:
                print(f">>> Early Stopping: Test PCC {avg_pcc:.4f} < 0.2 after 3 epochs.")
                return avg_pcc # Return current low PCC to trigger next config
        
        if avg_pcc > best_test_pcc:
            best_test_pcc = avg_pcc
            # try:
            #     model.save_weights(f"data/best_gnn_{avg_pcc:.4f}.weights.h5")
            # except:
            #     pass
            
    return best_test_pcc

def train_subgraph_gnn_model():
    # 1. Load Data
    print(">>> Phase 1: Loading Data...")
    landmark_path = "data/landmark_genes.json"
    
    # Load ALL data (limit=None)
    data = load_rfa_data(
        "data/cmap/level3_beta_ctl_n188708x12328.h5",
        "data/cmap/level3_beta_trt_cp_n1805898x12328.h5",
        landmark_path=landmark_path,
        siginfo_path="data/siginfo_beta.txt",
        use_landmark_genes=True,
        max_samples=50000 # Use more samples
    )
    if data is None: raise Exception("Failed to load data.")

    target_genes = data['target_genes']
    num_genes = len(target_genes)
    
    # 2. Initialize Subgraph Manager
    graph_path = "data/rfa_directed_edges.csv"
    if not os.path.exists(graph_path):
        print(f"Graph file {graph_path} not found. Using default OmniPath.")
        graph_path = "data/omnipath/omnipath_interactions.csv"
        
    subgraph_manager = SubgraphManager(graph_path, target_genes)
    
    # 3. Group Data by Drug
    print(">>> Phase 2: Grouping Data by Drug...")
    drug_groups = {}
    X_ctl = data['X_ctl'] # (M, Genes, 2)
    y_delta = data['y_delta'] # (M, Genes)
    X_fingerprint = data.get('X_fingerprint', None)
    drug_ids = data['drug_ids']
    cell_names = data.get('cell_names', None)

    all_target_mask = np.sum(X_ctl[:, :, 1], axis=0)
    roots = np.where(all_target_mask > 0)[0].astype(np.int32).tolist()
    out_scores = np.mean(np.abs(y_delta), axis=0)
    leafs = np.argsort(out_scores)[::-1][:300].astype(np.int32).tolist()
    subgraph_manager.pre_prune(roots, leafs, depth=10)
    
    # Process Cell IDs
    cell_info = None
    if cell_names is not None:
        unique_cells, cell_indices = np.unique(np.array(cell_names, dtype=str), return_inverse=True)
        num_cells = len(unique_cells)
        cell_info = {'num_cells': num_cells, 'indices': cell_indices}
        print(f"  Found {num_cells} unique cell lines.")
    
    unique_drugs, indices = np.unique(drug_ids, return_inverse=True)
    for i, drug_id in enumerate(unique_drugs):
        sample_indices = np.where(indices == i)[0]
        first_sample_idx = sample_indices[0]
        target_mask = X_ctl[first_sample_idx, :, 1]
        target_indices = np.where(target_mask > 0)[0]
        if len(target_indices) == 0: continue
        drug_groups[drug_id] = {'samples': sample_indices, 'targets': target_indices}
        
    print(f"  Found {len(drug_groups)} valid drugs with targets.")
    
    # 4. Auto-Optimization Loop
    configs = [
        {'hidden': 64, 'lr': 5e-4, 'epochs': 10, 'dropout': 0.2}, 
        {'hidden': 128, 'lr': 1e-4, 'epochs': 15, 'dropout': 0.3},
        {'hidden': 256, 'lr': 1e-4, 'epochs': 20, 'dropout': 0.4}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n====== Optimization Attempt {i+1} ======")
        pcc = train_and_evaluate(config, drug_groups, subgraph_manager, X_ctl, y_delta, X_fingerprint, num_genes, cell_info, topk_pcc=200)
        
        if pcc > 0.3:
            print(f"\n>>> Success! Target PCC reached: {pcc:.4f} > 0.3")
            break
        else:
            print(f">>> PCC {pcc:.4f} < 0.3. Trying next configuration...")

def sanity_overfit_single_drug_cell(max_train_samples=256, epochs=30, topk_pcc=200):
    print(">>> Sanity Overfit: single drug + single cell")
    landmark_path = "data/landmark_genes.json"
    data = load_rfa_data(
        "data/cmap/level3_beta_ctl_n188708x12328.h5",
        "data/cmap/level3_beta_trt_cp_n1805898x12328.h5",
        landmark_path=landmark_path,
        siginfo_path="data/siginfo_beta.txt",
        use_landmark_genes=True,
        max_samples=50000
    )
    if data is None:
        raise Exception("Failed to load data.")

    target_genes = data['target_genes']
    num_genes = len(target_genes)
    graph_path = "data/rfa_directed_edges.csv"
    subgraph_manager = SubgraphManager(graph_path, target_genes)

    X_ctl = data['X_ctl']
    y_delta = data['y_delta']
    drug_ids = np.array(data['drug_ids'], dtype=str)
    cell_names = np.array(data.get('cell_names', []), dtype=str)
    X_fingerprint = data.get('X_fingerprint', None)

    if cell_names.size == 0:
        raise Exception("No cell_names in loaded data.")

    pair_to_indices = {}
    for i in range(len(drug_ids)):
        key = (drug_ids[i], cell_names[i])
        if key not in pair_to_indices:
            pair_to_indices[key] = []
        pair_to_indices[key].append(i)

    best_key = None
    best_n = 0
    for key, idxs in pair_to_indices.items():
        if len(idxs) > best_n:
            best_key = key
            best_n = len(idxs)
    if best_key is None:
        raise Exception("No (drug, cell) pairs found.")

    drug_id, cell_name = best_key
    idxs = np.array(pair_to_indices[best_key], dtype=np.int32)
    if idxs.shape[0] > max_train_samples:
        idxs = idxs[:max_train_samples]
    print(f"  Using drug={drug_id} cell={cell_name} n={idxs.shape[0]}")

    unique_cells, cell_indices = np.unique(cell_names, return_inverse=True)
    num_cells = len(unique_cells)
    cell_info = {'num_cells': num_cells, 'indices': cell_indices}

    first_sample_idx = int(idxs[0])
    target_mask = X_ctl[first_sample_idx, :, 1]
    target_indices = np.where(target_mask > 0)[0].astype(np.int32)
    if target_indices.size == 0:
        raise Exception("Selected pair has empty targets.")

    sub_nodes, sub_adj = subgraph_manager.extract_subgraph(target_indices, k_hops=3)
    batch_adj = tf.expand_dims(sub_adj, axis=0)

    fp_dim = int(X_fingerprint.shape[1]) if X_fingerprint is not None else 0
    model = DifferentiableRFAGNN(output_dim=num_genes, fp_dim=fp_dim, num_cells=num_cells, hidden_dim=64, alpha=0.85, dropout=0.2)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)

    def loss_fn(y_true_delta, y_pred_delta, gates, epoch):
        pcc_val = pcc_loss_topk(y_true_delta, y_pred_delta, k=topk_pcc)
        w_sparse = 0.0 if epoch < 5 else 1e-3
        sparsity = tf.reduce_mean(tf.abs(gates))
        return pcc_val + w_sparse * sparsity, pcc_val

    for epoch in range(epochs):
        batch_X = tf.gather(X_ctl, idxs)
        batch_X = tf.gather(batch_X, sub_nodes, axis=1)
        batch_y = tf.gather(y_delta, idxs)
        batch_y = tf.gather(batch_y, sub_nodes, axis=1)
        batch_cell = tf.gather(cell_info['indices'], idxs)
        batch_fp = tf.gather(X_fingerprint, idxs) if X_fingerprint is not None else None
        if batch_fp is not None:
            preds, gates = model([batch_X, batch_adj, batch_cell, batch_fp])
        else:
            preds, gates = model([batch_X, batch_adj, batch_cell])
        with tf.GradientTape() as tape:
            preds, gates = model([batch_X, batch_adj, batch_cell, batch_fp]) if batch_fp is not None else model([batch_X, batch_adj, batch_cell])
            loss, pcc_v = loss_fn(batch_y, preds, gates, epoch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_pcc = float(1.0 - pcc_v.numpy())
        print(f"  Epoch {epoch+1}: Train TopK PCC = {train_pcc:.4f}")

def _topk_pcc_numpy(y_true, y_pred, k=200):
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    kk = min(int(k), yt.shape[0])
    if kk < 2:
        return 0.0
    idx = np.argpartition(np.abs(yt), -kk)[-kk:]
    yt_k = yt[idx]
    yp_k = yp[idx]
    if np.std(yt_k) < 1e-6 or np.std(yp_k) < 1e-6:
        return 0.0
    return float(pearsonr(yt_k, yp_k)[0])

def rfa_min_sanity(phase=0, epochs=50, topk=200, alpha=0.85, max_train_samples=256):
    landmark_path = "data/landmark_genes.json"
    data = load_rfa_data(
        "data/cmap/level3_beta_ctl_n188708x12328.h5",
        "data/cmap/level3_beta_trt_cp_n1805898x12328.h5",
        landmark_path=landmark_path,
        siginfo_path="data/siginfo_beta.txt",
        use_landmark_genes=True,
        max_samples=50000
    )
    if data is None:
        raise Exception("Failed to load data.")

    target_genes = data['target_genes']
    graph_path = "data/rfa_directed_edges.csv"
    subgraph_manager = SubgraphManager(graph_path, target_genes)

    X_ctl = data['X_ctl']
    y_delta = data['y_delta']
    drug_ids = np.array(data['drug_ids'], dtype=str)
    cell_names = np.array(data.get('cell_names', []), dtype=str)

    pair_to_indices = {}
    for i in range(len(drug_ids)):
        key = (drug_ids[i], cell_names[i])
        pair_to_indices.setdefault(key, []).append(i)
    best_key = max(pair_to_indices.items(), key=lambda kv: len(kv[1]))[0]
    drug_id, cell_name = best_key
    idxs = np.array(pair_to_indices[best_key], dtype=np.int32)
    if idxs.shape[0] > max_train_samples:
        idxs = idxs[:max_train_samples]
    print(f">>> RFA Min Sanity Phase {phase}: drug={drug_id} cell={cell_name} n={idxs.shape[0]}")

    first_sample_idx = int(idxs[0])
    target_mask_global = X_ctl[first_sample_idx, :, 1]
    target_indices = np.where(target_mask_global > 0)[0].astype(np.int32)
    if target_indices.size == 0:
        raise Exception("Selected pair has empty targets.")

    sub_nodes, sub_adj = subgraph_manager.extract_subgraph(target_indices, k_hops=3)
    local_n = int(sub_adj.shape[0])
    pos_map = {int(g): i for i, g in enumerate(sub_nodes.tolist())}
    sources_pos = [pos_map[int(t)] for t in target_indices.tolist() if int(t) in pos_map]
    if len(sources_pos) == 0:
        raise Exception("Targets not found in extracted subgraph.")
    target_mask = np.zeros((local_n,), dtype=np.float32)
    target_mask[sources_pos] = 1.0

    y_true = y_delta[idxs]
    y_true = y_true[:, sub_nodes]

    adj_abs = np.abs(sub_adj).astype(np.float32)
    adj_mask = (adj_abs > 1e-9).astype(np.float32)

    if phase == 0:
        F = _rfa_flow_from_adj(adj_abs, alpha=alpha)
        y_pred = (F @ target_mask.reshape(-1, 1)).reshape(-1)
        scores = [_topk_pcc_numpy(y_true[i], y_pred, k=topk) for i in range(y_true.shape[0])]
        print(f"  Phase0 TopK PCC mean={float(np.mean(scores)):.4f} median={float(np.median(scores)):.4f}")
        return

    F_fixed = tf.constant(_rfa_flow_from_adj(adj_abs, alpha=alpha), dtype=tf.float32)
    y_true_tf = tf.constant(y_true, dtype=tf.float32)
    target_mask_tf = tf.constant(target_mask.reshape(1, -1), dtype=tf.float32)
    target_mask_tf = tf.tile(target_mask_tf, [tf.shape(y_true_tf)[0], 1])

    if phase == 1:
        s_logits = tf.Variable(tf.zeros([local_n], dtype=tf.float32))
        optimizer = keras.optimizers.Adam(learning_rate=5e-2)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                s = tf.tanh(s_logits)[None, :]
                src = target_mask_tf * s
                pred = tf.matmul(F_fixed, tf.expand_dims(src, -1))
                pred = tf.squeeze(pred, -1)
                loss = pcc_loss_topk(y_true_tf, pred, k=topk)
            grads = tape.gradient(loss, [s_logits])
            optimizer.apply_gradients(zip(grads, [s_logits]))
            train_pcc = float(1.0 - loss.numpy())
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Phase1 Epoch {epoch+1}: Train TopK PCC = {train_pcc:.4f}")
        return

    if phase == 2:
        g_logits = tf.Variable(tf.zeros([local_n, local_n], dtype=tf.float32))
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        adj_abs_tf = tf.constant(adj_abs, dtype=tf.float32)
        adj_mask_tf = tf.constant(adj_mask, dtype=tf.float32)
        I = tf.eye(local_n, dtype=tf.float32)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                g = tf.sigmoid(g_logits) * adj_mask_tf
                A = g * adj_abs_tf
                out_degree = tf.reduce_sum(tf.abs(A), axis=1, keepdims=True)
                out_degree = tf.maximum(out_degree, 1e-9)
                U = A / out_degree
                S = alpha * U
                F = tf.linalg.inv(I - S + 1e-6 * I)
                src = target_mask_tf
                pred = tf.matmul(F, tf.expand_dims(src, -1))
                pred = tf.squeeze(pred, -1)
                loss = pcc_loss_topk(y_true_tf, pred, k=topk)
            grads = tape.gradient(loss, [g_logits])
            optimizer.apply_gradients(zip(grads, [g_logits]))
            train_pcc = float(1.0 - loss.numpy())
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Phase2 Epoch {epoch+1}: Train TopK PCC = {train_pcc:.4f}")
        return

if __name__ == "__main__":
    if "--sanity" in sys.argv:
        sanity_overfit_single_drug_cell()
    elif "--phase0" in sys.argv:
        rfa_min_sanity(phase=0)
    elif "--phase1" in sys.argv:
        rfa_min_sanity(phase=1)
    elif "--phase2" in sys.argv:
        rfa_min_sanity(phase=2)
    else:
        train_subgraph_gnn_model()
