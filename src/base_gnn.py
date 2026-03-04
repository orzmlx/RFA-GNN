import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GraphAttentionLayer(layers.Layer):
    def __init__(self, output_dim, num_heads=1, activation='relu', **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        
    def build(self, input_shape):
        # input_shape: [adj, features]
        # features: (Batch, N, F)
        feature_dim = input_shape[1][-1]
        
        self.kernels = []
        self.attn_kernels = []
        
        for _ in range(self.num_heads):
            # ## kernel 是什么？（特征变换矩阵 W）
            # - 形状： (F, Out）   
            # - F ：输入特征维度（每个节点输入向量的长度）
            # - Out ：这个 head 输出的特征维度（每个节点输出向量的长度）
            # - 作用：把节点的原始特征做一次线性变换：
            # [h = XW] ,这是内部进行一个变换，用来把节点的原始特征映射到一个新的空间表示。
            # 在代码里就是：
            # - h = tf.matmul(features, self.kernels[i]) （ base_gnn.py:L35-L37 ）
            # 直觉： kernel 决定“节点特征怎么被编码成更适合做注意力/聚合的表示”。
            kernel = self.add_weight(shape=(feature_dim, self.output_dim), initializer='glorot_uniform', name=f'kernel_{_}')
            #    形状： (2*Out, 1)
            #    之所以是 2*Out ，是因为 GAT 的标准注意力打分用的是节点 i 和节点 j 的拼接向量：
            #   [e_{ij} = \text{LeakyReLU}\left( a^T [h_i ,||, h_j] \right)
            #   ] [h_i || h_j] 拼接后长度就是 2*Out ，乘上 a （一个长度 2*Out 的向量）得到标量分数。
            # - 代码里没有显式拼接，而是把 a 拆成两半（“trick”）：
            
            #   - a = [a_left ; a_right]
            #   - 于是：
            #     [a^T [h_i||h_j] = a_{left}^T h_i + a_{right}^T h_j]
            #     对应代码：
            #   - attn_for_self = h @ a_left （ base_gnn.py:L42 ）
            #   - attn_for_neighs = h @ a_right （ base_gnn.py:L43 ）
            #   - 然后广播相加得到 (B,N,N) 的 scores （ base_gnn.py:L45-L47 ）
            # 直觉： attn_kernel 决定“节点 i 该更关注哪些邻居 j”
            attn_kernel = self.add_weight(shape=(2 * self.output_dim, 1), initializer='glorot_uniform', name=f'attn_kernel_{_}')
            self.kernels.append(kernel)
            self.attn_kernels.append(attn_kernel)
            
        super(GraphAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, return_embeddings=False):
        adj, features = inputs # adj: (N, N), features: (B, N, F)
        
        outputs = []
        for i in range(self.num_heads):
            # 1. Linear Transform: H = XW
            h = tf.matmul(features, self.kernels[i]) # (B, N, Out)
            
            # 2. Attention Mechanism
            # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            # Trick: a^T [h_i || h_j] = a_1^T h_i + a_2^T h_j
            
            attn_for_self = tf.matmul(h, self.attn_kernels[i][:self.output_dim]) # (B, N, 1)
            attn_for_neighs = tf.matmul(h, self.attn_kernels[i][self.output_dim:]) # (B, N, 1)
            
            # Broadcasting: (B, N, 1) + (B, 1, N) -> (B, N, N)
            scores = attn_for_self + tf.transpose(attn_for_neighs, perm=[0, 2, 1])
            scores = tf.nn.leaky_relu(scores)
            
            # 3. Masking based on Graph Structure
            # adj (N, N) -> (1, N, N)
            edge_weight = tf.cast(tf.expand_dims(adj, axis=0), tf.float32) # (1, N, N)
            mask = tf.not_equal(edge_weight, 0.0) # 创建一个布尔掩码，用于标记哪些边是真实存在的（即邻接矩阵中对应位置的值不为 0）
            # -1e9 ensures softmax -> 0
            scores = tf.where(mask, scores, -1e9)
            
            # 4. Softmax normalization
            attn_weights = tf.nn.softmax(scores, axis=-1)
            
            # 5. Aggregation
            node_repr = tf.matmul(attn_weights * edge_weight, h)
            outputs.append(node_repr)
            
        # Concat heads
        if self.num_heads > 1:
            output = tf.concat(outputs, axis=-1)
        else:
            output = outputs[0]
            
        return self.activation(output)


class GraphAttentionLayerSparse(layers.Layer):
    def __init__(self, output_dim, num_heads=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = input_shape[2][-1]
        self.kernels = []
        self.attn_kernels = []
        for i in range(self.num_heads):
            kernel = self.add_weight(shape=(feature_dim, self.output_dim), initializer="glorot_uniform", name=f"kernel_{i}")
            attn_kernel = self.add_weight(shape=(2 * self.output_dim, 1), initializer="glorot_uniform", name=f"attn_kernel_{i}")
            self.kernels.append(kernel)
            self.attn_kernels.append(attn_kernel)
        super().build(input_shape)

    def call(self, inputs):
        edge_index, edge_weight, features = inputs
        edge_index = tf.cast(edge_index, tf.int32)
        edge_weight = tf.cast(edge_weight, tf.float32)

        src = edge_index[0]
        dst = edge_index[1]
        n_nodes = tf.shape(features)[1]
        batch_size = tf.shape(features)[0]
        num_edges = tf.shape(src)[0]
        dst_rep = tf.tile(dst[None, :], [batch_size, 1])
        b_rep = tf.repeat(tf.range(batch_size)[:, None], repeats=num_edges, axis=1)
        seg_ids = tf.reshape(b_rep * n_nodes + dst_rep, [-1])
        num_segs = batch_size * n_nodes

        outputs = []
        for i in range(self.num_heads):
            h = tf.matmul(features, self.kernels[i])
            h_src = tf.gather(h, src, axis=1)
            h_dst = tf.gather(h, dst, axis=1)
            a_left = self.attn_kernels[i][: self.output_dim]
            a_right = self.attn_kernels[i][self.output_dim :]
            e_dst = tf.tensordot(h_dst, a_left, axes=[[2], [0]])
            e_src = tf.tensordot(h_src, a_right, axes=[[2], [0]])
            e = tf.nn.leaky_relu(tf.squeeze(e_dst + e_src, axis=-1))

            e_flat = tf.reshape(e, [-1])
            max_per_seg = tf.math.unsorted_segment_max(e_flat, seg_ids, num_segs)
            exp = tf.exp(e_flat - tf.gather(max_per_seg, seg_ids))
            denom = tf.math.unsorted_segment_sum(exp, seg_ids, num_segs)
            alpha_flat = exp / (tf.gather(denom, seg_ids) + 1e-9)
            alpha = tf.reshape(alpha_flat, [batch_size, num_edges])

            msg = alpha[:, :, None] * edge_weight[None, :, None] * h_src
            msg_flat = tf.reshape(msg, [batch_size * num_edges, self.output_dim])
            out_flat = tf.math.unsorted_segment_sum(msg_flat, seg_ids, num_segs)
            node_repr = tf.reshape(out_flat, [batch_size, n_nodes, self.output_dim])
            outputs.append(node_repr)

        if self.num_heads > 1:
            output = tf.concat(outputs, axis=-1)
        else:
            output = outputs[0]
        return self.activation(output)

class BaseLineGAT(keras.Model):
    def __init__(self, num_genes, num_cells=10, num_drugs=None, fingerprint_dim=0, 
                    hidden_dim=64, num_heads=4, 
                    dropout=0.2, use_residual=False, 
                    use_drug_fp_embedding= True, attention_layer_number = 10, output_after_embedding=False, per_node_embedding=False,
                    use_sparse_adj=False, use_cell_embedding=True, **kwargs):
        super(BaseLineGAT, self).__init__(**kwargs)
        self.num_genes = int(num_genes)
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        self.use_drug_fp_embedding = use_drug_fp_embedding # Now refers to Fingerprint Projection
        self.attention_layer_number = int(attention_layer_number)
        self.output_after_embedding = bool(output_after_embedding)
        self.per_node_embedding = bool(per_node_embedding)
        self.use_sparse_adj = bool(use_sparse_adj)
        self.use_cell_embedding = bool(use_cell_embedding)
        # Embedding for [Ctl, Target, Cell]
        self.expr_embedding = layers.Dense(hidden_dim, activation="relu")
        # 为了防止target 信号被淹没，不在和 expression 信号合并，而是单独投影到 hidden_dim 维度,并对其进行缩放
        self.target_embedding = layers.Dense(hidden_dim, activation="relu")
        self.target_scale_logit = self.add_weight(shape=(), initializer="zeros", name="target_scale_logit")

        if self.per_node_embedding:
            self.node_out_kernel = self.add_weight(
                shape=(self.num_genes, hidden_dim),
                initializer="glorot_uniform",
                name="node_out_kernel",
            )
            self.node_out_bias = self.add_weight(
                shape=(self.num_genes,),
                initializer="zeros",
                name="node_out_bias",
            )
        
        if self.use_cell_embedding:
            self.cell_embedding = layers.Embedding(num_cells, hidden_dim)
        
        # Drug Conditioning
        if self.use_drug_fp_embedding:
            if fingerprint_dim <= 0:
                raise Exception("fingerprint_dim must be greater than 0 when use_drug_embedding is True")
            # 同时生成缩放参数和偏移参数，两个参数的维度都与特征维度相同，所以总维度是 hidden_dim * 2
            print(f"启用药物指纹FiLM调制 (Dim: {fingerprint_dim} -> {hidden_dim})")
            self.drug_film = keras.Sequential([
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(
                    hidden_dim * 2,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                ),
            ])
        
        head_dim = hidden_dim // num_heads
        if head_dim * num_heads != hidden_dim:
            print(f"Warning: hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads}).")

        if self.attention_layer_number < 1:
            raise ValueError("attention_layer_number must be >= 1")

        self.gat_layers = []
        self.attn_norms = []
        self.attn_dropouts = []
        self.ffn_layers = []
        self.ffn_norms = []
        self.ffn_dropouts = []

        for _ in range(self.attention_layer_number):
            if self.use_sparse_adj:
                self.gat_layers.append(GraphAttentionLayerSparse(head_dim, num_heads=num_heads, activation="relu"))
            else:
                self.gat_layers.append(GraphAttentionLayer(head_dim, num_heads=num_heads, activation='relu'))
            self.attn_norms.append(layers.LayerNormalization())
            self.attn_dropouts.append(layers.Dropout(dropout))
            # ffn先投射到较高维度，然后降维，即采用“扩展-收缩”的结构
            self.ffn_layers.append(keras.Sequential([
                layers.Dense(hidden_dim * 2, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim),
            ]))
            self.ffn_norms.append(layers.LayerNormalization())
            self.ffn_dropouts.append(layers.Dropout(dropout))

        # Output
        self.dense = layers.Dense(1) # Output delta
        
        # Optional: Fusion layer if we concat
        self.fusion = layers.Concatenate(axis=-1)

        # Input Projection for Residual Connection
        # Since input dim (1 or 2) != hidden_dim, we need a projection to add residual
        self.input_proj = layers.Dense(hidden_dim)
        
    def call(self, inputs, return_embeddings=False):
        # inputs: [adj, ctl_expr, drug_targets, cell_idx, drug_fp]
        if self.use_sparse_adj:
            if self.use_drug_fp_embedding:
                edge_index, edge_weight, ctl_expr, drug_targets, cell_idx, drug_fp = inputs
            else:
                edge_index, edge_weight, ctl_expr, drug_targets, cell_idx = inputs
        else:
            if self.use_drug_fp_embedding:
                adj, ctl_expr, drug_targets, cell_idx, drug_fp = inputs
            else:
                adj, ctl_expr, drug_targets, cell_idx = inputs
        if len(ctl_expr.shape) == 2:
            ctl_expr_base = ctl_expr
        elif len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 1:
            ctl_expr_base = tf.squeeze(ctl_expr, axis=-1)
        elif len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 2:
            ctl_expr_base = ctl_expr[..., 0]
        else:
            raise ValueError("ctl_expr must be (B, N), (B, N, 1) or (B, N, 2)")

        x_expr = tf.expand_dims(ctl_expr_base, axis=-1)
             
        if len(drug_targets.shape) != 2:
            raise ValueError("drug_targets must be (B, N)")
        x_target = tf.expand_dims(drug_targets, axis=-1)
        # 1. Base Features softplus(x)=log(1+e x)
        x_expr_emb = self.expr_embedding(x_expr)
        x_target_emb = self.target_embedding(x_target)
        target_scale = tf.nn.softplus(self.target_scale_logit)
        
        # 2. Add Cell Context
        x = x_expr_emb + target_scale * x_target_emb
        if self.use_cell_embedding:
            cell_emb = self.cell_embedding(cell_idx)
            cell_emb = tf.expand_dims(cell_emb, axis=1)
            cell_emb = tf.tile(cell_emb, [1, tf.shape(x_expr)[1], 1])
            x = x + cell_emb
        
        # 3. 药物调制（只在输入层注入一次），通过 tanh 将 gamma 限制在 (-1, 1) 之间
        if self.use_drug_fp_embedding:
            film = self.drug_film(drug_fp)
            gamma, beta = tf.split(film, num_or_size_splits=2, axis=-1)
            gamma = tf.tanh(gamma)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        x_in = x
        for i in range(self.attention_layer_number):
            res = x_in
            if self.use_sparse_adj:
                x_attn = self.gat_layers[i]([edge_index, edge_weight, x_in])
            else:
                x_attn = self.gat_layers[i]([adj, x_in])
            x_attn = self.attn_dropouts[i](x_attn)
            x_in = self.attn_norms[i](x_attn + res)
            x_ffn = self.ffn_layers[i](x_in)
            x_ffn = self.ffn_dropouts[i](x_ffn)
            x_in = self.ffn_norms[i](x_in + x_ffn)

        if self.per_node_embedding:
            if tf.executing_eagerly() and int(x_in.shape[1]) != self.num_genes:
                raise ValueError(f"num_genes mismatch: model={self.num_genes}, input={int(x_in.shape[1])}")
            predicted = tf.einsum("bnh,nh->bn", x_in, self.node_out_kernel) + self.node_out_bias[None, :]
        else:
            out = self.dense(x_in)
            predicted = tf.squeeze(out, axis=-1)

        if bool(return_embeddings):
            return predicted, x_in
        if self.use_residual:
            return ctl_expr_base + predicted
        return predicted
