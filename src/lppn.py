import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def topk_indices_by_abs(y_true, k):
    y_true = tf.convert_to_tensor(y_true)
    n = tf.shape(y_true)[1]
    k = tf.minimum(tf.cast(k, tf.int32), n)
    return tf.math.top_k(tf.abs(y_true), k=k, sorted=False).indices


def pcc_1_minus_mean(y_true, y_pred, eps=1e-8):
    mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
    my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    xm = y_true - mx
    ym = y_pred - my
    r_num = tf.reduce_sum(xm * ym, axis=1)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + eps)
    r = r_num / r_den
    return 1.0 - tf.reduce_mean(r)


def topk_pcc_loss(y_true, y_pred, k=200):
    idx = topk_indices_by_abs(y_true, k=k)
    y_true_k = tf.gather(y_true, idx, batch_dims=1)
    y_pred_k = tf.gather(y_pred, idx, batch_dims=1)
    return pcc_1_minus_mean(y_true_k, y_pred_k)


def sign_mismatch_loss(y_true, y_pred, k=200):
    idx = topk_indices_by_abs(y_true, k=k)
    y_true_k = tf.gather(y_true, idx, batch_dims=1)
    y_pred_k = tf.gather(y_pred, idx, batch_dims=1)
    return tf.reduce_mean(tf.nn.relu(-y_true_k * y_pred_k))


def hard_concrete_sample(log_alpha, temperature=0.67, gamma=-0.1, zeta=1.1, training=False):
    log_alpha = tf.convert_to_tensor(log_alpha)
    if training:
        u = tf.random.uniform(tf.shape(log_alpha), minval=1e-6, maxval=1.0 - 1e-6, dtype=log_alpha.dtype)
        s = tf.sigmoid((log_alpha + tf.math.log(u) - tf.math.log(1.0 - u)) / temperature)
    else:
        s = tf.sigmoid(log_alpha)
    s_bar = s * (zeta - gamma) + gamma
    z = tf.clip_by_value(s_bar, 0.0, 1.0)
    z_hard = tf.cast(z > 0.5, z.dtype)
    return z + tf.stop_gradient(z_hard - z)

def hard_concrete_l0_prob(log_alpha, temperature=0.67, gamma=-0.1, zeta=1.1):
    log_alpha = tf.convert_to_tensor(log_alpha)
    thresh = temperature * tf.math.log(-gamma / zeta)
    return tf.sigmoid(log_alpha - thresh)


class LPPN(keras.Model):
    def __init__(
        self,
        num_genes,
        hidden_dim=64,
        k_steps=3,
        fp_dim=0,
        num_cells=0,
        drug_emb_dim=64,
        cell_emb_dim=32,
        gene_emb_dim=32,
        dropout=0.1,
        learn_gate=True,
        learn_source=True,
        hard_concrete=True,
        gate_temperature=0.67,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        self.hidden_dim = int(hidden_dim)
        self.k_steps = int(k_steps)
        self.fp_dim = int(fp_dim)
        self.num_cells = int(num_cells)
        self.learn_gate = bool(learn_gate)
        self.learn_source = bool(learn_source)
        self.hard_concrete = bool(hard_concrete)
        self.gate_temperature = float(gate_temperature)

        self.dropout = layers.Dropout(dropout)

        if self.fp_dim > 0:
            self.drug_proj = keras.Sequential(
                [
                    layers.Dense(drug_emb_dim, activation="relu"),
                    layers.Dense(drug_emb_dim, activation="relu"),
                ]
            )
        else:
            self.drug_proj = None

        if self.num_cells > 0:
            self.cell_emb = layers.Embedding(self.num_cells, cell_emb_dim)
            self.cell_proj = layers.Dense(drug_emb_dim, activation="relu")
        else:
            self.cell_emb = None
            self.cell_proj = None

        self.gene_emb = layers.Embedding(self.num_genes, gene_emb_dim)
        self.gene_proj = layers.Dense(drug_emb_dim, activation="relu")
        self.gene_token = layers.Embedding(self.num_genes, self.hidden_dim)
        self.ctl_token = keras.Sequential(
            [
                layers.Dense(self.hidden_dim, activation="relu"),
                layers.Dense(self.hidden_dim, activation="tanh"),
            ]
        )
        self.base_norm = layers.LayerNormalization()

        self.init_mlp = keras.Sequential(
            [
                layers.Dense(self.hidden_dim, activation="tanh"),
                layers.Dense(self.hidden_dim, activation="tanh"),
            ]
        )

        self.gru = layers.GRUCell(self.hidden_dim)

        self.alpha_src = layers.Dense(self.hidden_dim, use_bias=False)
        self.alpha_dst = layers.Dense(self.hidden_dim, use_bias=False)
        self.alpha_cond_proj = layers.Dense(self.hidden_dim, use_bias=False)
        self.alpha_bias = self.add_weight(name="alpha_bias", shape=(1,), initializer=tf.constant_initializer(0.0))

        self.sign_src = layers.Dense(self.hidden_dim, use_bias=False)
        self.sign_dst = layers.Dense(self.hidden_dim, use_bias=False)
        self.sign_cond_proj = layers.Dense(self.hidden_dim, use_bias=False)
        self.sign_bias = self.add_weight(name="sign_bias", shape=(1,), initializer=tf.constant_initializer(0.0))

        self.target_importance = keras.Sequential(
            [
                layers.Dense(self.hidden_dim, activation="relu"),
                layers.Dense(1, activation="tanh"),
            ]
        )

        self.out = layers.Dense(1)
        self.gene_scale = layers.Embedding(self.num_genes, 1, embeddings_initializer="ones")

    def _cond_embedding(self, cell_ids=None, drug_fp=None):
        cond = None
        if self.cell_emb is not None and cell_ids is not None:
            c = self.cell_proj(self.cell_emb(cell_ids))
            cond = c if cond is None else cond + c
        if self.drug_proj is not None and drug_fp is not None:
            d = self.drug_proj(drug_fp)
            cond = d if cond is None else cond + d
        if cond is None:
            cond = tf.zeros([tf.shape(cell_ids)[0], self.hidden_dim], dtype=tf.float32) if cell_ids is not None else tf.zeros([tf.shape(drug_fp)[0], self.hidden_dim], dtype=tf.float32)
        return cond

    def call(self, inputs, training=False):
        x, adj = inputs[0], inputs[1]
        cell_ids = inputs[2] if len(inputs) > 2 else None
        drug_fp = inputs[3] if len(inputs) > 3 else None
        dist = inputs[4] if len(inputs) > 4 else None
        gene_ids = inputs[5] if len(inputs) > 5 else None

        x = tf.cast(x, tf.float32)
        adj = tf.cast(adj, tf.float32)
        if len(adj.shape) == 2:
            adj = tf.expand_dims(adj, 0)

        b = tf.shape(x)[0]
        n = tf.shape(x)[1]

        target_mask = tf.expand_dims(x[:, :, 1], -1)
        x_ctl = tf.expand_dims(x[:, :, 0], -1)

        cond = self._cond_embedding(cell_ids=cell_ids, drug_fp=drug_fp)
        if gene_ids is None:
            gene_ids = tf.range(n, dtype=tf.int32)
        gene_ids = tf.cast(gene_ids, tf.int32)
        g = self.gene_proj(self.gene_emb(gene_ids))
        g = tf.expand_dims(g, 0)
        g = tf.tile(g, [b, 1, 1])
        cond_g = tf.expand_dims(cond, 1)
        cond_g = tf.tile(cond_g, [1, n, 1])

        base = self.base_norm(self.ctl_token(x_ctl) + self.gene_token(gene_ids)[None, :, :])
        s_v = self.target_importance(tf.concat([cond_g, g], axis=-1))
        if not self.learn_source:
            s_v = tf.ones_like(s_v)
        h0_vec = tf.expand_dims(self.init_mlp(cond), 1)
        h = target_mask * s_v * (h0_vec + tf.zeros([b, n, self.hidden_dim], dtype=tf.float32))
        pert = target_mask * s_v * (h0_vec + tf.zeros([b, n, self.hidden_dim], dtype=tf.float32))
        h = pert
        h = self.dropout(h, training=training)

        source = pert

        adj_mask = tf.cast(tf.abs(adj) > 1e-9, tf.float32)
        a = adj * adj_mask
        if adj_mask.shape[0] == 1:
            adj_mask = tf.tile(adj_mask, [b, 1, 1])
            a = tf.tile(a, [b, 1, 1])

        node_mask = None
        if dist is not None:
            dist = tf.cast(dist, tf.float32)
            node_mask = tf.cast(dist <= 3.0, tf.float32)
            node_mask = tf.reshape(node_mask, [1, -1, 1])
            node_mask = tf.tile(node_mask, [b, 1, 1])

        z_mean = tf.constant(0.0, dtype=tf.float32)
        z_gt = tf.constant(0.0, dtype=tf.float32)
        last_h = h
        for _ in range(self.k_steps):
            if node_mask is not None:
                last_h = last_h * node_mask
            h_aug = last_h + base
            if self.learn_gate:
                hs_a = self.alpha_src(h_aug)
                hd_a = self.alpha_dst(h_aug)
                cond_a = self.alpha_cond_proj(cond)
                cond_a = tf.expand_dims(cond_a, 1)
                hs_a = hs_a + cond_a
                hd_a = hd_a + cond_a
                log_alpha = tf.matmul(hs_a, hd_a, transpose_b=True) + self.alpha_bias

                hs_s = self.sign_src(h_aug)
                hd_s = self.sign_dst(h_aug)
                cond_s = self.sign_cond_proj(cond)
                cond_s = tf.expand_dims(cond_s, 1)
                hs_s = hs_s + cond_s
                hd_s = hd_s + cond_s
                logit_sign = tf.matmul(hs_s, hd_s, transpose_b=True) + self.sign_bias

                if self.hard_concrete:
                    z = hard_concrete_sample(log_alpha, temperature=self.gate_temperature, training=training) * adj_mask
                else:
                    z = tf.sigmoid(log_alpha) * adj_mask

                sign = tf.tanh(logit_sign) * adj_mask
                gate = z * sign
            else:
                z = adj_mask
                gate = adj_mask

            z_mean = z_mean + tf.reduce_mean(z)
            z_gt = z_gt + tf.reduce_mean(tf.cast(z > 0.5, tf.float32))

            w = a * gate
            m = tf.matmul(tf.transpose(w, [0, 2, 1]), last_h)
            m = m + source + base
            m = tf.reshape(m, [b * n, self.hidden_dim])
            h_flat = tf.reshape(last_h, [b * n, self.hidden_dim])
            h_new, _ = self.gru(m, [h_flat], training=training)
            last_h = tf.reshape(h_new, [b, n, self.hidden_dim])
            last_h = self.dropout(last_h, training=training)
            if node_mask is not None:
                last_h = last_h * node_mask

        y = self.out(tf.concat([last_h, base], axis=-1))
        y = tf.squeeze(y, -1)
        w_gene = tf.squeeze(self.gene_scale(gene_ids), -1)
        y = y * tf.expand_dims(w_gene, 0)

        depth_pen = tf.constant(0.0, dtype=tf.float32)
        h_abs_mean = tf.reduce_mean(tf.abs(last_h))

        metrics = {
            "z_mean": z_mean / tf.cast(self.k_steps, tf.float32),
            "z_gt_half": z_gt / tf.cast(self.k_steps, tf.float32),
            "depth_pen": depth_pen,
            "h_abs_mean": h_abs_mean,
        }
        return y, metrics
