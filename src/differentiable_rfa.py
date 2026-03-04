import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class DifferentiableRFAGNN(keras.Model):
    def __init__(self, adj_matrix, hidden_dim=64, alpha=0.85, num_gnn_layers=2):
        """
        Differentiable RFA-GNN Model with Edge Pruning.
        
        Args:
            adj_matrix: (N, N) numpy array or tensor, the static Signed Adjacency Matrix of the original graph.
                        Should contain values {-1, 0, 1} or float weights.
            hidden_dim: Hidden dimension for GNN node embeddings.
            alpha: RFA restart probability / scaling factor (default 0.85).
            num_gnn_layers: Number of GNN message passing layers.
        """
        super(DifferentiableRFAGNN, self).__init__()
        
        # Static Graph Structure
        # Shape: (N, N)
        self.adj_static = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        self.num_nodes = adj_matrix.shape[0]
        self.alpha = alpha
        
        # Masks for existing edges (where adj != 0) to enforce structure
        self.adj_mask = tf.cast(tf.abs(self.adj_static) > 1e-6, tf.float32)
        
        # 1. GNN Encoder Layers
        # We use a simple GCN-like update or just MLP per node mixed with adjacency
        # Here we implement a basic Graph Convolution logic for the "Pruning GNN"
        self.gnn_layers = []
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(layers.Dense(hidden_dim, activation='relu'))
            
        self.node_encoder = layers.Dense(hidden_dim, activation='relu')
        
        # 2. Edge Gating Mechanism
        # Predicts a scalar m_ij for every pair (but we will mask non-edges)
        # We use a bilinear layer or concatenation: m_ij = sigmoid(W * [h_i || h_j])
        # To save memory, we can compute h_src and h_dst projections.
        self.gate_src_proj = layers.Dense(hidden_dim, use_bias=False)
        self.gate_dst_proj = layers.Dense(hidden_dim, use_bias=False)
        self.gate_bias = self.add_weight(name="gate_bias", shape=(1,), initializer="zeros")
        
    def call(self, inputs):
        """
        Args:
            inputs: tuple (x_ctl, targets)
                x_ctl: (Batch, N) or (Batch, N, 1) - Baseline expression
                targets: (Batch, N) or (Batch, N, 1) - Binary target vector (1 for drug targets)
        
        Returns:
            final_pred: (Batch, N) - Predicted post-treatment expression
        """
        x_ctl, targets = inputs
        
        # Ensure inputs are (Batch, N, 1)
        if len(x_ctl.shape) == 2: x_ctl = tf.expand_dims(x_ctl, -1)
        if len(targets.shape) == 2: targets = tf.expand_dims(targets, -1)
        
        # -----------------------------------------------------------
        # Step 1: Feature Construction & GNN Pruning
        # -----------------------------------------------------------
        # Node Features: [x_ctl, targets] -> (Batch, N, 2)
        node_features = tf.concat([x_ctl, targets], axis=-1)
        
        # Initial Embedding -> (Batch, N, H)
        h = self.node_encoder(node_features)
        
        # GNN Propagation (Simple GCN: A * H * W)
        # Using the absolute adjacency to allow flow in both directions for feature gathering if needed,
        # or just the directed graph. User said "Propagate on original graph G".
        # We'll use the static adjacency for propagation.
        adj_norm = self.adj_mask  # Use binary structure for GNN support
        
        for layer in self.gnn_layers:
            # Aggregate neighbors: (N, N) @ (B, N, H) -> (B, N, H)
            # tf.matmul works for (N, N) x (B, N, H) broadcasting the first dim? 
            # No, tf.matmul(a, b) requires rank matching or standard broadcasting.
            # We treat adj_norm as (1, N, N) for broadcasting.
            agg = tf.matmul(tf.expand_dims(adj_norm, 0), h) 
            h = layer(agg) # Update
            
        # -----------------------------------------------------------
        # Step 2: Edge Gating (m_ij)
        # -----------------------------------------------------------
        # Compute edge gates based on node embeddings
        # m_ij = sigmoid((h_i W_s) @ (h_j W_d)^T)
        
        # Projections: (Batch, N, H)
        h_src = self.gate_src_proj(h)
        h_dst = self.gate_dst_proj(h)
        
        # Similarity / Interaction map: (Batch, N, N)
        # (B, N, H) @ (B, H, N) -> (B, N, N)
        logits = tf.matmul(h_src, h_dst, transpose_b=True) + self.gate_bias
        
        # Apply Sigmoid to get gates in [0, 1]
        gates = tf.sigmoid(logits)
        
        # Mask out non-existing edges from the original graph
        # Expand adj_mask to (1, N, N)
        masked_gates = gates * tf.expand_dims(self.adj_mask, 0)
        
        # -----------------------------------------------------------
        # Step 3: Construct Pruned Graph & RFA Matrix
        # -----------------------------------------------------------
        # A_tilde_ij = m_ij * A_ij (Element-wise)
        # self.adj_static contains signs/weights. 
        # (1, N, N) * (B, N, N) -> (B, N, N)
        A_tilde = tf.expand_dims(self.adj_static, 0) * masked_gates
        
        # Normalize to get U_tilde
        # Row sum of absolute values: sum_j |A_tilde_ij|
        # Avoid division by zero with epsilon
        abs_A = tf.abs(A_tilde)
        out_degree = tf.reduce_sum(abs_A, axis=2, keepdims=True) # (B, N, 1)
        out_degree = tf.maximum(out_degree, 1e-9) # Avoid 0 division
        
        # U_tilde = D^-1 * A_tilde
        # Broadcasting: (B, N, N) / (B, N, 1)
        U_tilde = A_tilde / out_degree
        
        # S_tilde = alpha * U_tilde
        S_tilde = self.alpha * U_tilde
        
        # -----------------------------------------------------------
        # Step 4: Inversion & Prediction
        # -----------------------------------------------------------
        # F_tilde = (I - S_tilde)^-1
        # Identity matrix (Batch, N, N)
        I = tf.eye(self.num_nodes, batch_shape=[tf.shape(x_ctl)[0]])
        
        # Matrix Inversion (Differentiable)
        # Note: For very large graphs, Neumann series approximation is better.
        # But for N ~ 1000, direct inv is okay on GPU.
        # (I - S)
        M_to_inv = I - S_tilde
        
        # F_tilde: (Batch, N, N)
        F_tilde = tf.linalg.inv(M_to_inv)
        
        # Predict Increment: y_hat = Sum_{t in Td} F[:, t]
        # This is equivalent to F @ targets
        # F: (B, N, N), targets: (B, N, 1) -> (B, N, 1)
        y_hat = tf.matmul(F_tilde, targets)
        
        # Remove last dim -> (Batch, N)
        y_hat = tf.squeeze(y_hat, -1)
        x_ctl_squeeze = tf.squeeze(x_ctl, -1)
        
        # Output: x_ctl + y_hat
        final_output = x_ctl_squeeze + y_hat
        
        return final_output

# Helper to instantiate and test
if __name__ == "__main__":
    # Dummy Test
    N = 10
    adj = np.random.choice([-1, 0, 1], size=(N, N)).astype(np.float32)
    # Remove self loops and ensure sparsity for realism
    adj = adj * (np.random.rand(N, N) > 0.7)
    np.fill_diagonal(adj, 0)
    
    model = DifferentiableRFAGNN(adj, hidden_dim=16)
    
    x_c = tf.random.normal((5, N))
    t_d = tf.cast(tf.random.uniform((5, N)) > 0.8, tf.float32)
    
    out = model([x_c, t_d])
    print("Output shape:", out.shape)
    print("Sample output:", out[0].numpy())
