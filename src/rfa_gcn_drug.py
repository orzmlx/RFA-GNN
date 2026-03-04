import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class GreedyStructureLearner(layers.Layer):
    """
    贪心结构学习层 (Greedy Structure Learner)
    对应论文中的 Algorithm 2:
    根据药物特征和基因状态，贪心地选择最重要的 K 个邻居进行聚合。
    """
    def __init__(self, k_neighbors=10, **kwargs):
        super(GreedyStructureLearner, self).__init__(**kwargs)
        self.k_neighbors = k_neighbors
        
    def build(self, input_shape):
        # input_shape: [adj, features]
        feature_dim = input_shape[1][-1]
        
        self.attn_kernel = self.add_weight(
            shape=(feature_dim, 1),
            initializer='glorot_uniform',
            name='attn_kernel'
        )
        super(GreedyStructureLearner, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: [adj, features]
        adj, features = inputs
        
        # Compute "Importance"
        importance = tf.matmul(features, self.attn_kernel)
        scores = importance + tf.transpose(importance, perm=[0, 2, 1])
        
        # Masking (OmniPath Constraint)
        adj_mask = tf.expand_dims(adj, axis=0)
        scores = tf.where(adj_mask > 0, scores, -1e9)
        
        # Greedy Top-K
        top_k_values, top_k_indices = tf.nn.top_k(scores, k=self.k_neighbors)
        top_k_weights = tf.nn.softmax(top_k_values, axis=-1)
        
        return top_k_weights, top_k_indices

class DrugModulatedRFALayer(layers.Layer):
    """
    药物调节的 RFA 层
    """
    def __init__(self, output_dim, alpha=0.5, activation='relu', k_neighbors=15, **kwargs):
        super(DrugModulatedRFALayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.alpha = alpha
        self.activation = keras.activations.get(activation)
        self.greedy_learner = GreedyStructureLearner(k_neighbors=k_neighbors)

    def build(self, input_shape):
        feature_dim = input_shape[1][-1]
        self.kernel = self.add_weight(shape=(feature_dim, self.output_dim), initializer='glorot_uniform')
        self.bias = self.add_weight(shape=(self.output_dim,), initializer='zeros')
        super(DrugModulatedRFALayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: [adj, features]
        adj, features = inputs
        top_k_weights, top_k_indices = self.greedy_learner([adj, features])
        
        support = tf.matmul(features, self.kernel)
        
        # Gather neighbors
        neighbor_features = tf.gather(support, top_k_indices, batch_dims=1)
        
        weights_expanded = tf.expand_dims(top_k_weights, axis=-1)
        aggregated = tf.reduce_sum(neighbor_features * weights_expanded, axis=2)
        
        if features.shape[-1] == self.output_dim:
            output = self.alpha * features + (1 - self.alpha) * aggregated
        else:
            output = self.alpha * support + (1 - self.alpha) * aggregated
            
        output += self.bias
        return self.activation(output)

class RFAGCN_Drug(keras.Model):
    def __init__(self, num_genes, num_cells=10, drug_dim=None, hidden_dim=64, alpha=0.3, dropout=0.2, use_residual=True):
        super(RFAGCN_Drug, self).__init__()
        self.use_residual = use_residual
        self.gene_embedding = layers.Dense(hidden_dim, activation='relu')
        # 移除 drug_fc
        
        self.cell_embedding = layers.Embedding(num_cells, hidden_dim)
        
        self.rfa1 = DrugModulatedRFALayer(hidden_dim, alpha=alpha)
        self.rfa2 = DrugModulatedRFALayer(hidden_dim, alpha=alpha)
        
        self.concat = layers.Concatenate(axis=-1)
        self.dropout = layers.Dropout(dropout)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(1)

    def call(self, inputs):
        # adj: (N, N)
        # ctl_expr: (Batch, N)
        # drug_targets: (Batch, N) - Binary
        # cell_idx: (Batch,)
        adj, ctl_expr, drug_targets, cell_idx = inputs
        
        # 1. 构建节点特征: [Ctl, Target]
        if len(ctl_expr.shape) == 2:
            x_expr = tf.expand_dims(ctl_expr, axis=-1)
            x_target = tf.expand_dims(drug_targets, axis=-1)
            x_input = tf.concat([x_expr, x_target], axis=-1) # (Batch, N, 2)
            base_expr = ctl_expr
        else:
            x_input = ctl_expr
            base_expr = ctl_expr[..., 0]
        
        x = self.gene_embedding(x_input) # (Batch, N, H)
        
        # 2. RFA 传播
        x = self.rfa1([adj, x])
        x = self.dropout(x)
        x = self.rfa2([adj, x])
        
        # 3. 融合 Cell 信息
        cell_emb = self.cell_embedding(cell_idx) # (Batch, H)
        cell_emb = tf.expand_dims(cell_emb, axis=1) # (Batch, 1, H)
        cell_emb = tf.tile(cell_emb, [1, x.shape[1], 1]) # (Batch, N, H)
        
        # Drug 信息已经融合在 x 里了，这里只需要 concat x 和 cell
        combined = self.concat([x, cell_emb])
        
        out = self.dense1(combined)
        out = self.dense2(out)
        predicted = tf.squeeze(out, axis=-1)
        
        if self.use_residual:
            return base_expr + predicted
        return predicted
