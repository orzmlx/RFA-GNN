import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RelationalGraphConvSparse(layers.Layer):
    def __init__(self, out_dim, relation_names, activation="relu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = int(out_dim)
        self.relation_names = list(relation_names)
        self.activation = keras.activations.get(activation)
        self.dropout = float(dropout)
        self.kernels = {}
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(self.dropout)
        self.self_proj = None

    def build(self, input_shape):
        feat_dim = int(input_shape[-1])
        self.self_proj = self.add_weight(shape=(feat_dim, self.out_dim), initializer="glorot_uniform", name="self_proj")
        for r in self.relation_names:
            self.kernels[r] = self.add_weight(shape=(feat_dim, self.out_dim), initializer="glorot_uniform", name=f"kernel_{r}")
        super().build(input_shape)

    def call(self, features, rel_edges):
        n_nodes = tf.shape(features)[1]
        batch_size = tf.shape(features)[0]
        h0 = tf.matmul(features, self.self_proj)
        out = h0

        for r in self.relation_names:
            edge_index, edge_weight = rel_edges.get(r, (None, None))
            if edge_index is None:
                continue
            edge_index = tf.cast(edge_index, tf.int32)
            edge_weight = tf.cast(edge_weight, tf.float32)
            src = edge_index[0]
            dst = edge_index[1]
            num_edges = tf.shape(src)[0]

            h = tf.matmul(features, self.kernels[r])
            h_src = tf.gather(h, src, axis=1)
            msg = edge_weight[None, :, None] * h_src

            dst_rep = tf.tile(dst[None, :], [batch_size, 1])
            b_rep = tf.repeat(tf.range(batch_size)[:, None], repeats=num_edges, axis=1)
            seg_ids = tf.reshape(b_rep * n_nodes + dst_rep, [-1])
            msg_flat = tf.reshape(msg, [batch_size * num_edges, self.out_dim])
            out_flat = tf.math.unsorted_segment_sum(msg_flat, seg_ids, batch_size * n_nodes)
            agg = tf.reshape(out_flat, [batch_size, n_nodes, self.out_dim])
            out = out + agg

        out = self.drop(out)
        out = self.norm(out)
        return self.activation(out)


class RelationalRegulatoryGNN(keras.Model):
    def __init__(
        self,
        num_genes,
        num_cells,
        fingerprint_dim,
        hidden_dim=64,
        num_layers=4,
        dropout=0.2,
        use_cell_embedding=True,
        use_drug_embedding=True,
        relation_names=None,
        per_node_embedding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_genes = int(num_genes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.use_cell_embedding = bool(use_cell_embedding)
        self.use_drug_embedding = bool(use_drug_embedding)
        self.per_node_embedding = bool(per_node_embedding)
        self.relation_names = list(relation_names or [])

        self.shared_embedding = layers.Dense(self.hidden_dim, activation="relu")
        if self.use_cell_embedding:
            self.cell_embedding = layers.Embedding(int(num_cells), self.hidden_dim)

        if self.use_drug_embedding:
            if int(fingerprint_dim) <= 0:
                self.use_drug_embedding = False
            else:
                self.drug_film = keras.Sequential(
                    [
                        layers.Dense(self.hidden_dim, activation="relu"),
                        layers.Dense(
                            self.hidden_dim * 2,
                            kernel_initializer="zeros",
                            bias_initializer="zeros",
                        ),
                    ]
                )

        self.layers_mp = []
        for _ in range(self.num_layers):
            self.layers_mp.append(
                RelationalGraphConvSparse(
                    self.hidden_dim,
                    relation_names=self.relation_names,
                    activation="relu",
                    dropout=float(dropout),
                )
            )

        if self.per_node_embedding:
            self.node_out_kernel = self.add_weight(
                shape=(self.num_genes, self.hidden_dim),
                initializer="glorot_uniform",
                name="node_out_kernel",
            )
            self.node_out_bias = self.add_weight(
                shape=(self.num_genes,),
                initializer="zeros",
                name="node_out_bias",
            )
        else:
            self.out_dense = layers.Dense(1)

    def call(self, inputs, rel_edges):
        ctl_expr, drug_targets, cell_idx, drug_fp = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)

        if len(ctl_expr.shape) == 2:
            ctl_base = ctl_expr
        elif len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 1:
            ctl_base = tf.squeeze(ctl_expr, axis=-1)
        elif len(ctl_expr.shape) == 3 and ctl_expr.shape[-1] == 2:
            ctl_base = ctl_expr[..., 0]
        else:
            raise ValueError("ctl_expr must be (B,N), (B,N,1) or (B,N,2)")

        x_expr = tf.expand_dims(ctl_base, axis=-1)
        if len(drug_targets.shape) == 2:
            x_target = tf.expand_dims(drug_targets, axis=-1)
        else:
            x_target = drug_targets

        x = self.shared_embedding(x_expr) + self.shared_embedding(x_target)

        if self.use_cell_embedding:
            c = self.cell_embedding(cell_idx)
            c = tf.expand_dims(c, axis=1)
            c = tf.tile(c, [1, tf.shape(x)[1], 1])
            x = x + c

        if self.use_drug_embedding:
            film = self.drug_film(drug_fp)
            gamma, beta = tf.split(film, num_or_size_splits=2, axis=-1)
            gamma = tf.tanh(gamma)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        for layer in self.layers_mp:
            x = layer(x, rel_edges)

        if self.per_node_embedding:
            if tf.executing_eagerly() and int(x.shape[1]) != self.num_genes:
                raise ValueError("num_genes mismatch")
            pred = tf.einsum("bnh,nh->bn", x, self.node_out_kernel) + self.node_out_bias[None, :]
        else:
            pred = tf.squeeze(self.out_dense(x), axis=-1)
        return pred


class RelationalWrapperSparse(keras.Model):
    def __init__(self, model, rel_index_np):
        super().__init__()
        self.model_core = model
        self.rel_edges = {}
        for r, (ei, ew) in rel_index_np.items():
            self.rel_edges[r] = (tf.constant(ei, dtype=tf.int32), tf.constant(ew, dtype=tf.float32))

    def call(self, inputs):
        ctl, drug_targets, cell_idx, drug_fp = inputs
        return self.model_core((ctl, drug_targets, cell_idx, drug_fp), self.rel_edges)
