import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --------------------------
# 1. 数据准备（模拟生物调控网络）
# --------------------------
def generate_simulated_data(num_nodes=50, num_edges=150):
    """生成模拟调控网络数据：邻接矩阵+节点特征+源信号+真实流标签"""
    # 1.1 邻接矩阵 W (num_nodes × num_nodes)：有向带符号（±1表示激活/抑制）
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edge_indices = np.random.choice(num_nodes, size=(2, num_edges), replace=False)
    adj[edge_indices[0], edge_indices[1]] = np.random.choice([1.0, -1.0], size=num_edges)
    
    # 1.2 节点特征：模拟基因本体(GO)特征（100维one-hot）
    node_feat = tf.one_hot(np.random.choice(100, size=num_nodes), depth=100)
    
    # 1.3 源信号 s：模拟药物扰动（仅1个源节点激活）
    source_signal = np.zeros((num_nodes, 1), dtype=np.float32)
    source_node = np.random.choice(num_nodes)
    source_signal[source_node] = 1.0
    
    # 1.4 真实流标签：用原始RFA公式生成（模拟真实调控流）
    alpha = 0.99
    I = np.eye(num_nodes)
    # 对adj进行行归一化，保证收敛
    W_norm = adj / (np.sum(np.abs(adj), axis=1, keepdims=True) + 1e-8)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W_norm)))
    if spectral_radius > 0:
        S = (alpha / spectral_radius) * W_norm
    else:
        S = W_norm
        
    true_flow = np.linalg.inv(I - S) @ source_signal  # 原始RFA流计算
    
    return adj, node_feat, source_signal, true_flow

# 生成数据
adj, node_feat, source_signal, true_flow = generate_simulated_data()
num_nodes = node_feat.shape[0]
node_feat_dim = node_feat.shape[1]

# --------------------------
# 2. 核心模块
# --------------------------

class FeatureTransformation(layers.Layer):
    """
    第一阶段：学习算子 (Operator Learning)
    使用 MLP 将原始基因特征变换为隐空间特征
    """
    def __init__(self, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_dim, activation="relu")
        self.dense2 = layers.Dense(self.hidden_dim) # 线性输出，作为 H_0
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

class ExplicitRFAPropagation(layers.Layer):
    """
    第二阶段：显式 RFA 传播 (Explicit RFA Algorithm)
    严格执行公式：H^{k+1} = (1-alpha) * H_0 + alpha * A * H^k
    """
    def __init__(self, k_hops=10, alpha=0.9, **kwargs):
        super().__init__(**kwargs)
        self.k_hops = k_hops
        self.alpha = alpha
        
    def call(self, inputs):
        # H_0: 初始变换后的特征 (Z)
        # adj: 邻接矩阵 (W)
        H_0, adj = inputs
        
        # 初始化 H 为 H_0
        H = H_0
        
        # 显式迭代 RFA 算法
        for _ in range(self.k_hops):
            # 1. 聚合邻居信息 (A * H)
            # 这一步体现了利用 OmniPath 的物理结构
            propagated = tf.matmul(adj, H)
            
            # 2. RFA 更新公式 (带残差连接)
            # (1 - alpha) * H_init + alpha * A * H_curr
            H = (1 - self.alpha) * H_0 + self.alpha * propagated
            
        return H

# --------------------------
# 3. 完整模型构建 (APPNP 风格)
# --------------------------
def build_explicit_rfa_gnn(num_nodes, node_feat_dim):
    # 输入层
    node_feat_input = layers.Input(shape=(node_feat_dim,), name="node_feat")
    adj_input = layers.Input(shape=(num_nodes,), name="adj")
    source_input = layers.Input(shape=(1,), name="source_signal")
    
    # 1. 重塑邻接矩阵
    adj_reshaped = layers.Reshape((num_nodes, num_nodes))(adj_input)
    
    # 2. 学习算子 (变换特征)
    # 这一步让模型适应 CMAP 数据，学习特征的非线性关系
    feature_transform = FeatureTransformation(hidden_dim=64)
    H_0 = feature_transform(node_feat_input)
    
    # 3. 融合源信号 (模拟药物扰动)
    # 将源信号作为一种特殊的特征叠加进去
    source_feat = layers.Dense(64)(source_input) # 将标量源信号映射到 64 维
    H_0_combined = H_0 + tf.tile(source_feat, [num_nodes, 1])
    
    # 4. 显式 RFA 传播
    # 这一步严格遵循 RFA 算法，利用 OmniPath 结构进行递归聚合
    rfa_prop = ExplicitRFAPropagation(k_hops=10, alpha=0.9)
    H_final = rfa_prop([H_0_combined, adj_reshaped])
    
    # 5. 输出层 (映射回 1 维调控流)
    predicted_flow = layers.Dense(1)(H_final)
    
    # 构建模型
    model = models.Model(
        inputs=[node_feat_input, adj_input, source_input],
        outputs=predicted_flow,
        name="Explicit_RFA_GNN"
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

# 初始化模型
model = build_explicit_rfa_gnn(num_nodes, node_feat_dim)
model.summary()

# --------------------------
# 4. 训练与验证
# --------------------------
# 适配输入格式
adj_input = adj.reshape(1, num_nodes * num_nodes)
node_feat_input = node_feat[np.newaxis, ...]
source_input = source_signal[np.newaxis, ...]
true_flow_input = true_flow[np.newaxis, ...]

print("\n开始训练显式 RFA 模型...")
history = model.fit(
    x=[node_feat_input, adj_input, source_input],
    y=true_flow_input,
    epochs=50,
    batch_size=1,
    verbose=1
)

# 预测验证
predicted_flow = model.predict([node_feat_input, adj_input, source_input], verbose=0)
print("\n--- 验证结果 ---")
print("原始RFA流（前5个节点）:", true_flow[:5].flatten())
print("模型预测流（前5个节点）:", predicted_flow[0, :5].flatten())
print("MSE损失:", model.evaluate([node_feat_input, adj_input, source_input], true_flow_input, verbose=0))
