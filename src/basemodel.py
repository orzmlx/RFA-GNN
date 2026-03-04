import tensorflow as tf
import numpy as np

class RegulatoryFlowBaseModel(tf.keras.Model):
    def __init__(self, num_genes, alpha=0.9):
        super().__init__()
        self.num_genes = num_genes
        self.alpha = alpha
        # 可学习的有符号边权重（对称）
        # 初始化为小随机数
        w_init = tf.random_normal_initializer(stddev=0.05)
        self.W = tf.Variable(
            initial_value=w_init(shape=(num_genes, num_genes), dtype='float32'),
            trainable=True, name='W')

    def call(self, p, training=False):
        # p: (batch, num_genes) 输入扰动（如药物作用）
        # 1. 构建对称邻接矩阵
        A = (self.W + tf.transpose(self.W)) / 2.0  # 保证无向图
        # 2. 计算未归一化 Step 矩阵 U
        abs_A = tf.abs(A)
        U = tf.where(abs_A > 0, A / (tf.reduce_sum(abs_A, axis=1, keepdims=True) + 1e-8), tf.zeros_like(A))
        # 3. 计算缩放 Step 矩阵 S
        S = self.alpha * U
        # 4. 计算调控流矩阵 F = (I - S)^{-1}
        I = tf.eye(self.num_genes)
        F = tf.linalg.inv(I - S)
        # 5. 预测表达 x^ = F @ p^T
        x_pred = tf.matmul(p, tf.transpose(F))  # (batch, num_genes)
        return x_pred, A, S, F

    def compute_loss(self, x_pred, x_true):
        # MSE loss
        return tf.reduce_mean(tf.square(x_pred - x_true))

    def edge_contribution(self, x_pred, x_true):
        # 计算每条边的贡献度 |dLoss/dW_ij|
        with tf.GradientTape() as tape:
            tape.watch(self.W)
            loss = self.compute_loss(x_pred, x_true)
        grad = tape.gradient(loss, self.W)
        return tf.abs(grad)

    def prune_edges(self, threshold=1e-3):
        # 剪枝：将贡献度低于阈值的边置零
        contrib = self.edge_contribution(self.last_x_pred, self.last_x_true)
        mask = tf.where(contrib > threshold, tf.ones_like(self.W), tf.zeros_like(self.W))
        self.W.assign(self.W * mask)

# 用法示例：
# model = RegulatoryFlowBaseModel(num_genes=978)
# x_pred, A, S, F = model(p)
# loss = model.compute_loss(x_pred, x_true)
# grads = tape.gradient(loss, model.trainable_variables)
# ...
