import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class XPertBase(keras.Model):
    def __init__(
        self,
        num_genes,
        fp_dim,
        num_cells,
        hidden_dim=256,
        gene_emb_dim=64,
        cond_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        self.fp_dim = int(fp_dim)
        self.num_cells = int(num_cells)
        self.hidden_dim = int(hidden_dim)

        self.gene_emb = layers.Embedding(self.num_genes, gene_emb_dim)
        self.gene_proj = layers.Dense(hidden_dim, activation="relu")

        self.drug_proj = keras.Sequential(
            [
                layers.Dense(cond_dim, activation="relu"),
                layers.Dense(cond_dim, activation="relu"),
            ]
        )
        self.cell_emb = layers.Embedding(self.num_cells, cond_dim)
        self.cond_proj = layers.Dense(hidden_dim, activation="relu")

        self.ctl_proj = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim, activation="relu"),
            ]
        )
        self.target_proj = layers.Dense(hidden_dim, activation="relu")

        self.fuse = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

    def call(self, inputs, training=False):
        x = tf.cast(inputs[0], tf.float32)
        cell_ids = tf.cast(inputs[1], tf.int32)
        drug_fp = tf.cast(inputs[2], tf.float32)
        gene_ids = tf.cast(inputs[3], tf.int32)

        x_ctl = tf.expand_dims(x[:, :, 0], -1)
        target_mask = tf.expand_dims(x[:, :, 1], -1)

        cond = self.drug_proj(drug_fp) + self.cell_emb(cell_ids)
        cond = self.cond_proj(cond)
        cond_g = tf.tile(tf.expand_dims(cond, 1), [1, tf.shape(x)[1], 1])

        g = self.gene_proj(self.gene_emb(gene_ids))
        g = tf.tile(tf.expand_dims(g, 0), [tf.shape(x)[0], 1, 1])

        ctl = self.ctl_proj(x_ctl)
        t = self.target_proj(target_mask)
        z = tf.concat([ctl, t, cond_g, g, g * cond_g], axis=-1)
        y = self.fuse(z, training=training)
        return tf.squeeze(y, -1)


class ResidualXPertLPPN(keras.Model):
    def __init__(self, base_model, gnn_model, alpha_init=0.2):
        super().__init__()
        self.base = base_model
        self.gnn = gnn_model
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.constant_initializer(float(alpha_init)),
            trainable=True,
        )

    def call(self, inputs, training=False):
        x, adj = inputs[0], inputs[1]
        cell_ids = inputs[2]
        drug_fp = inputs[3]
        dist = inputs[4] if len(inputs) > 4 else None
        gene_ids = inputs[5] if len(inputs) > 5 else None

        y_base = self.base([x, cell_ids, drug_fp, gene_ids], training=training)
        y_res, metrics = self.gnn([x, adj, cell_ids, drug_fp, dist, gene_ids], training=training)
        y = y_base + self.alpha[0] * y_res
        metrics = dict(metrics)
        metrics["alpha"] = self.alpha[0]
        return y, metrics

