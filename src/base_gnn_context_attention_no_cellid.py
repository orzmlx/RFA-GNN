import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GraphAttentionLayerContextDense(layers.Layer):
    def __init__(self, output_dim, num_heads=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = int(input_shape[1][-1])
        context_dim = int(input_shape[2][-1])
        self.kernels = []
        self.attn_kernels = []
        self.context_self_kernels = []
        self.context_neigh_kernels = []

        for i in range(self.num_heads):
            self.kernels.append(
                self.add_weight(
                    shape=(feature_dim, self.output_dim),
                    initializer="glorot_uniform",
                    name=f"kernel_{i}",
                )
            )
            self.attn_kernels.append(
                self.add_weight(
                    shape=(2 * self.output_dim, 1),
                    initializer="glorot_uniform",
                    name=f"attn_kernel_{i}",
                )
            )
            self.context_self_kernels.append(
                self.add_weight(
                    shape=(context_dim, self.output_dim),
                    initializer="glorot_uniform",
                    name=f"context_self_kernel_{i}",
                )
            )
            self.context_neigh_kernels.append(
                self.add_weight(
                    shape=(context_dim, self.output_dim),
                    initializer="glorot_uniform",
                    name=f"context_neigh_kernel_{i}",
                )
            )
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        adj, features, context = inputs
        outputs = []
        attns = []

        for i in range(self.num_heads):
            h = tf.matmul(features, self.kernels[i])

            attn_for_self = tf.matmul(h, self.attn_kernels[i][: self.output_dim])
            attn_for_neighs = tf.matmul(h, self.attn_kernels[i][self.output_dim :])
            scores = attn_for_self + tf.transpose(attn_for_neighs, perm=[0, 2, 1])

            ctx_self = tf.matmul(context, self.context_self_kernels[i])
            ctx_neigh = tf.matmul(context, self.context_neigh_kernels[i])
            ctx_self_scores = tf.reduce_sum(h * ctx_self[:, None, :], axis=-1, keepdims=True)
            ctx_neigh_scores = tf.reduce_sum(h * ctx_neigh[:, None, :], axis=-1, keepdims=True)
            scores = scores + ctx_self_scores + tf.transpose(ctx_neigh_scores, perm=[0, 2, 1])
            scores = tf.nn.leaky_relu(scores)

            edge_weight = tf.cast(tf.expand_dims(adj, axis=0), tf.float32)
            mask = tf.not_equal(edge_weight, 0.0)
            scores = tf.where(mask, scores, -1e9)
            attn_weights = tf.nn.softmax(scores, axis=-1)

            node_repr = tf.matmul(attn_weights * edge_weight, h)
            outputs.append(node_repr)
            if bool(return_attention):
                attns.append(attn_weights)

        if self.num_heads > 1:
            output = tf.concat(outputs, axis=-1)
        else:
            output = outputs[0]

        output = self.activation(output)
        if bool(return_attention):
            return output, tf.stack(attns, axis=1)
        return output


class GraphAttentionLayerContextSparse(layers.Layer):
    def __init__(self, output_dim, num_heads=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = int(input_shape[2][-1])
        context_dim = int(input_shape[3][-1])
        self.kernels = []
        self.attn_kernels = []
        self.context_self_kernels = []
        self.context_neigh_kernels = []

        for i in range(self.num_heads):
            self.kernels.append(
                self.add_weight(
                    shape=(feature_dim, self.output_dim),
                    initializer="glorot_uniform",
                    name=f"kernel_{i}",
                )
            )
            self.attn_kernels.append(
                self.add_weight(
                    shape=(2 * self.output_dim, 1),
                    initializer="glorot_uniform",
                    name=f"attn_kernel_{i}",
                )
            )
            self.context_self_kernels.append(
                self.add_weight(
                    shape=(context_dim, self.output_dim),
                    initializer="glorot_uniform",
                    name=f"context_self_kernel_{i}",
                )
            )
            self.context_neigh_kernels.append(
                self.add_weight(
                    shape=(context_dim, self.output_dim),
                    initializer="glorot_uniform",
                    name=f"context_neigh_kernel_{i}",
                )
            )
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        edge_index, edge_weight, features, context = inputs
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
        attns = []
        for i in range(self.num_heads):
            h = tf.matmul(features, self.kernels[i])
            h_src = tf.gather(h, src, axis=1)
            h_dst = tf.gather(h, dst, axis=1)

            a_left = self.attn_kernels[i][: self.output_dim]
            a_right = self.attn_kernels[i][self.output_dim :]
            e_dst = tf.tensordot(h_dst, a_left, axes=[[2], [0]])
            e_src = tf.tensordot(h_src, a_right, axes=[[2], [0]])
            e = tf.squeeze(e_dst + e_src, axis=-1)

            ctx_self = tf.matmul(context, self.context_self_kernels[i])
            ctx_neigh = tf.matmul(context, self.context_neigh_kernels[i])
            ctx_dst = tf.reduce_sum(h_dst * ctx_self[:, None, :], axis=-1)
            ctx_src = tf.reduce_sum(h_src * ctx_neigh[:, None, :], axis=-1)
            e = tf.nn.leaky_relu(e + ctx_dst + ctx_src)

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
            if bool(return_attention):
                attns.append(alpha)

        if self.num_heads > 1:
            output = tf.concat(outputs, axis=-1)
        else:
            output = outputs[0]
        output = self.activation(output)
        if bool(return_attention):
            return output, tf.stack(attns, axis=1)
        return output


class BaseLineGATContextAttentionNoCellId(keras.Model):
    def __init__(
        self,
        num_genes,
        num_cells=10,
        num_drugs=None,
        fingerprint_dim=0,
        hidden_dim=64,
        num_heads=4,
        dropout=0.2,
        use_residual=False,
        use_drug_fp_embedding=True,
        attention_layer_number=10,
        output_after_embedding=False,
        per_node_embedding=False,
        use_sparse_adj=False,
        use_cell_embedding=False,
        context_hidden_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_genes = int(num_genes)
        self.use_residual = bool(use_residual)
        self.hidden_dim = int(hidden_dim)
        self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
        self.attention_layer_number = int(attention_layer_number)
        self.output_after_embedding = bool(output_after_embedding)
        self.per_node_embedding = bool(per_node_embedding)
        self.use_sparse_adj = bool(use_sparse_adj)
        self.context_hidden_dim = int(context_hidden_dim or hidden_dim * 2)

        self.expr_embedding = layers.Dense(hidden_dim, activation="relu")
        self.target_embedding = layers.Dense(hidden_dim, activation="relu")
        self.target_scale_logit = self.add_weight(shape=(), initializer="zeros", name="target_scale_logit")

        self.context_norm = layers.LayerNormalization(axis=-1)
        self.context_encoder = keras.Sequential(
            [
                layers.Dense(self.context_hidden_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim, activation="relu"),
            ]
        )
        self.context_delta = layers.Dense(hidden_dim)
        self.context_gate = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim, activation="sigmoid"),
            ]
        )
        self.context_scale_logit = self.add_weight(shape=(), initializer="zeros", name="context_scale_logit")

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

        if self.use_drug_fp_embedding:
            if fingerprint_dim <= 0:
                raise ValueError("fingerprint_dim must be greater than 0 when use_drug_fp_embedding is True")
            self.drug_film = keras.Sequential(
                [
                    layers.Dense(hidden_dim, activation="relu"),
                    layers.Dense(
                        hidden_dim * 2,
                        kernel_initializer="zeros",
                        bias_initializer="zeros",
                    ),
                ]
            )

        head_dim = hidden_dim // num_heads
        if head_dim * num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")
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
                self.gat_layers.append(GraphAttentionLayerContextSparse(head_dim, num_heads=num_heads, activation="relu"))
            else:
                self.gat_layers.append(GraphAttentionLayerContextDense(head_dim, num_heads=num_heads, activation="relu"))
            self.attn_norms.append(layers.LayerNormalization())
            self.attn_dropouts.append(layers.Dropout(dropout))
            self.ffn_layers.append(
                keras.Sequential(
                    [
                        layers.Dense(hidden_dim * 2, activation="relu"),
                        layers.Dropout(dropout),
                        layers.Dense(hidden_dim),
                    ]
                )
            )
            self.ffn_norms.append(layers.LayerNormalization())
            self.ffn_dropouts.append(layers.Dropout(dropout))

        self.dense = layers.Dense(1)

    def _parse_inputs(self, inputs):
        if self.use_sparse_adj:
            if self.use_drug_fp_embedding:
                if len(inputs) == 6:
                    edge_index, edge_weight, ctl_expr, drug_targets, _cell_idx, drug_fp = inputs
                else:
                    edge_index, edge_weight, ctl_expr, drug_targets, drug_fp = inputs
                return edge_index, edge_weight, ctl_expr, drug_targets, drug_fp
            if len(inputs) == 5:
                edge_index, edge_weight, ctl_expr, drug_targets, _cell_idx = inputs
            else:
                edge_index, edge_weight, ctl_expr, drug_targets = inputs
            return edge_index, edge_weight, ctl_expr, drug_targets, None
        if self.use_drug_fp_embedding:
            if len(inputs) == 5:
                adj, ctl_expr, drug_targets, _cell_idx, drug_fp = inputs
            else:
                adj, ctl_expr, drug_targets, drug_fp = inputs
            return adj, None, ctl_expr, drug_targets, drug_fp
        if len(inputs) == 4:
            adj, ctl_expr, drug_targets, _cell_idx = inputs
        else:
            adj, ctl_expr, drug_targets = inputs
        return adj, None, ctl_expr, drug_targets, None

    def _get_ctl_base(self, ctl_expr):
        if len(ctl_expr.shape) == 2:
            return ctl_expr
        if len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 1:
            return tf.squeeze(ctl_expr, axis=-1)
        if len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 2:
            return ctl_expr[..., 0]
        raise ValueError("ctl_expr must be (B, N), (B, N, 1) or (B, N, 2)")

    def _build_cell_context(self, ctl_expr_base):
        ctl_context = self.context_norm(ctl_expr_base)
        ctl_context = self.context_encoder(ctl_context)
        context_delta = self.context_delta(ctl_context)
        context_scale = tf.nn.softplus(self.context_scale_logit)

        context_gate = self.context_gate(ctl_context)
        fused_context = context_scale * context_gate * context_delta

        return fused_context

    def call(self, inputs, return_embeddings=False, output_attention=False):
        graph_a, graph_b, ctl_expr, drug_targets, drug_fp = self._parse_inputs(inputs)

        ctl_expr_base = self._get_ctl_base(ctl_expr)
        x_expr = tf.expand_dims(ctl_expr_base, axis=-1)
        if len(drug_targets.shape) != 2:
            raise ValueError("drug_targets must be (B, N)")
        x_target = tf.expand_dims(drug_targets, axis=-1)

        x_expr_emb = self.expr_embedding(x_expr)
        x_target_emb = self.target_embedding(x_target)
        target_scale = tf.nn.softplus(self.target_scale_logit)
        x = x_expr_emb + target_scale * x_target_emb

        fused_context = self._build_cell_context(ctl_expr_base)
        x = x + fused_context[:, None, :]

        if self.use_drug_fp_embedding:
            film = self.drug_film(drug_fp)
            gamma, beta = tf.split(film, num_or_size_splits=2, axis=-1)
            gamma = tf.tanh(gamma)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        x_in = x
        attentions = []
        for i in range(self.attention_layer_number):
            res = x_in
            if self.use_sparse_adj:
                if bool(output_attention):
                    x_attn, attn = self.gat_layers[i]([graph_a, graph_b, x_in, fused_context], return_attention=True)
                    attentions.append(attn)
                else:
                    x_attn = self.gat_layers[i]([graph_a, graph_b, x_in, fused_context])
            else:
                if bool(output_attention):
                    x_attn, attn = self.gat_layers[i]([graph_a, x_in, fused_context], return_attention=True)
                    attentions.append(attn)
                else:
                    x_attn = self.gat_layers[i]([graph_a, x_in, fused_context])
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
            predicted = tf.squeeze(self.dense(x_in), axis=-1)

        if bool(return_embeddings) and bool(output_attention):
            return predicted, x_in, attentions
        if bool(return_embeddings):
            return predicted, x_in
        if bool(output_attention):
            return predicted, attentions
        if self.use_residual:
            return ctl_expr_base + predicted
        return predicted