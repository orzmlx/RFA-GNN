import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from base_gnn import GraphAttentionLayer, GraphAttentionLayerSparse


class BaseLineGATHybridContext(keras.Model):
    """UPert variant with a hybrid context: cell identity + control expression profile.

    A convex combination (learned mixing weight) fuses a cell-ID embedding with a
    control-derived context vector.  The fused context is injected before the GAT
    stack, giving every gene node access to both what cell it is in and what that
    cell's baseline state looks like.  Branch-level dropout on the cell embedding
    encourages the model to fall back on the control-derived signal.
    """
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
        use_cell_embedding=True,
        predict_uncertainty=False,
        cell_dropout_rate=0.2,
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
        self.use_cell_embedding = bool(use_cell_embedding)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.cell_dropout_rate = float(cell_dropout_rate)
        self.context_hidden_dim = int(context_hidden_dim or hidden_dim * 2)

        self.expr_embedding = layers.Dense(hidden_dim, activation="relu")
        self.target_embedding = layers.Dense(hidden_dim, activation="relu")
        self.target_scale_logit = self.add_weight(shape=(), initializer="zeros", name="target_scale_logit")

        if self.use_cell_embedding:
            self.cell_embedding = layers.Embedding(num_cells, hidden_dim)
            # Start biased toward control context and let training decide how much cell id to keep.
            self.cell_mix_logit = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(-3.0),
                name="cell_mix_logit",
            )
            # Dropping the whole branch works better here than dropping random embedding coordinates.
            self.cell_dropout = layers.Dropout(self.cell_dropout_rate, noise_shape=(None, 1))
        else:
            self.cell_embedding = None
            self.cell_mix_logit = None
            self.cell_dropout = None

        self.context_norm = layers.LayerNormalization(axis=-1)
        self.context_encoder = keras.Sequential(
            [
                layers.Dense(self.context_hidden_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim, activation="relu"),
            ]
        )
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
            if self.predict_uncertainty:
                self.node_logvar_kernel = self.add_weight(
                    shape=(self.num_genes, hidden_dim),
                    initializer="glorot_uniform",
                    name="node_logvar_kernel",
                )
                self.node_logvar_bias = self.add_weight(
                    shape=(self.num_genes,),
                    initializer="zeros",
                    name="node_logvar_bias",
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
                self.gat_layers.append(GraphAttentionLayerSparse(head_dim, num_heads=num_heads, activation="relu"))
            else:
                self.gat_layers.append(GraphAttentionLayer(head_dim, num_heads=num_heads, activation="relu"))
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
        if self.predict_uncertainty:
            self.dense_logvar = layers.Dense(1)

    def _parse_inputs(self, inputs):
        if self.use_sparse_adj:
            if self.use_drug_fp_embedding:
                edge_index, edge_weight, ctl_expr, drug_targets, cell_idx, drug_fp = inputs
                return edge_index, edge_weight, ctl_expr, drug_targets, cell_idx, drug_fp
            edge_index, edge_weight, ctl_expr, drug_targets, cell_idx = inputs
            return edge_index, edge_weight, ctl_expr, drug_targets, cell_idx, None
        if self.use_drug_fp_embedding:
            adj, ctl_expr, drug_targets, cell_idx, drug_fp = inputs
            return adj, None, ctl_expr, drug_targets, cell_idx, drug_fp
        adj, ctl_expr, drug_targets, cell_idx = inputs
        return adj, None, ctl_expr, drug_targets, cell_idx, None

    def _get_ctl_base(self, ctl_expr):
        if len(ctl_expr.shape) == 2:
            return ctl_expr
        if len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 1:
            return tf.squeeze(ctl_expr, axis=-1)
        if len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 2:
            return ctl_expr[..., 0]
        raise ValueError("ctl_expr must be (B, N), (B, N, 1) or (B, N, 2)")

    def _build_cell_context(self, ctl_expr_base, cell_idx, training=False):
        # --- control-derived branch ---
        ctl_context = self.context_norm(ctl_expr_base)
        ctl_context = self.context_encoder(ctl_context, training=training)

        if self.use_cell_embedding:
            # --- cell-identity branch with branch-level dropout ---
            cell_base = self.cell_embedding(cell_idx)
            cell_base = self.cell_dropout(cell_base, training=training)
            # Convex combination: the two branches sum to one, keeping scales comparable.
            cell_scale = tf.nn.sigmoid(self.cell_mix_logit)
            context_scale = 1.0 - cell_scale
            fused_context = cell_scale * cell_base + context_scale * ctl_context
        else:
            fused_context = ctl_context

        return fused_context

    def get_logged_scales(self):
        scales = {"target_scale": float(tf.nn.softplus(self.target_scale_logit).numpy())}
        if self.cell_mix_logit is not None:
            cell_scale = float(tf.nn.sigmoid(self.cell_mix_logit).numpy())
            scales["cell_scale"] = cell_scale
            scales["context_scale"] = 1.0 - cell_scale
        else:
            scales["context_scale"] = 1.0
        return scales

    def call(self, inputs, training=False, return_embeddings=False, output_attention=False):
        graph_a, graph_b, ctl_expr, drug_targets, cell_idx, drug_fp = self._parse_inputs(inputs)

        ctl_expr_base = self._get_ctl_base(ctl_expr)
        x_expr = tf.expand_dims(ctl_expr_base, axis=-1)
        if len(drug_targets.shape) != 2:
            raise ValueError("drug_targets must be (B, N)")
        x_target = tf.expand_dims(drug_targets, axis=-1)

        x_expr_emb = self.expr_embedding(x_expr)
        x_target_emb = self.target_embedding(x_target)
        target_scale = tf.nn.softplus(self.target_scale_logit)
        x = x_expr_emb + target_scale * x_target_emb

        # Inject the fused (cell-id + control) context once before the GAT stack.
        fused_context = self._build_cell_context(ctl_expr_base, cell_idx, training=training)
        x = x + fused_context[:, None, :]

        if self.use_drug_fp_embedding:
            film = self.drug_film(drug_fp, training=training)
            gamma, beta = tf.split(film, num_or_size_splits=2, axis=-1)
            gamma = tf.tanh(gamma)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        x_in = x
        attentions = []
        for i in range(self.attention_layer_number):
            res = x_in
            if self.use_sparse_adj:
                if bool(output_attention):
                    x_attn, attn = self.gat_layers[i]([graph_a, graph_b, x_in], return_attention=True)
                    attentions.append(attn)
                else:
                    x_attn = self.gat_layers[i]([graph_a, graph_b, x_in])
            else:
                if bool(output_attention):
                    x_attn, attn = self.gat_layers[i]([graph_a, x_in], return_attention=True)
                    attentions.append(attn)
                else:
                    x_attn = self.gat_layers[i]([graph_a, x_in])
            x_attn = self.attn_dropouts[i](x_attn, training=training)
            x_in = self.attn_norms[i](x_attn + res)
            x_ffn = self.ffn_layers[i](x_in, training=training)
            x_ffn = self.ffn_dropouts[i](x_ffn, training=training)
            x_in = self.ffn_norms[i](x_in + x_ffn)

        if self.per_node_embedding:
            if tf.executing_eagerly() and int(x_in.shape[1]) != self.num_genes:
                raise ValueError(f"num_genes mismatch: model={self.num_genes}, input={int(x_in.shape[1])}")
            predicted = tf.einsum("bnh,nh->bn", x_in, self.node_out_kernel) + self.node_out_bias[None, :]
            if self.predict_uncertainty:
                logvar = tf.einsum("bnh,nh->bn", x_in, self.node_logvar_kernel) + self.node_logvar_bias[None, :]
        else:
            predicted = tf.squeeze(self.dense(x_in), axis=-1)
            if self.predict_uncertainty:
                logvar = tf.squeeze(self.dense_logvar(x_in), axis=-1)

        if self.predict_uncertainty:
            predicted_out = tf.stack([predicted, logvar], axis=-1)
        else:
            predicted_out = predicted

        if bool(return_embeddings) and bool(output_attention):
            return predicted_out, x_in, attentions
        if bool(return_embeddings):
            return predicted_out, x_in
        if bool(output_attention):
            return predicted_out, attentions
        if self.use_residual:
            if self.predict_uncertainty:
                return tf.stack([ctl_expr_base + predicted, logvar], axis=-1)
            return ctl_expr_base + predicted
        return predicted_out
