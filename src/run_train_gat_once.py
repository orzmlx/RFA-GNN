import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def eval_pcc_mse(model, ctl, drug, cells, y_true, loss_mask, batch_size=32, max_eval=None):
    if max_eval is not None and len(ctl) > int(max_eval):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ctl), size=int(max_eval), replace=False)
        ctl = ctl[idx]
        drug = drug[idx]
        cells = cells[idx]
        y_true = y_true[idx]

    pred = model.predict([ctl, drug, cells], batch_size=batch_size, verbose=0)
    valid_indices = np.where(np.asarray(loss_mask)[0] > 0)[0]
    y_true_valid = y_true[:, valid_indices]
    pred_valid = pred[:, valid_indices]

    pcc_list = []
    for i in range(len(y_true_valid)):
        yt = y_true_valid[i]
        yp = pred_valid[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p, _ = pearsonr(yt, yp)
            pcc_list.append(p)
    avg_pcc = float(np.mean(pcc_list)) if pcc_list else 0.0
    mse = float(mean_squared_error(y_true_valid, pred_valid))
    return {"mse": mse, "pcc": avg_pcc}


class PCCCallback(keras.callbacks.Callback):
    def __init__(self, loss_mask, train_data, val_data, batch_size=32, max_eval=2048):
        super().__init__()
        self.loss_mask = loss_mask
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.max_eval = int(max_eval) if max_eval is not None else None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ctl_tr, drug_tr, cell_tr, y_tr = self.train_data
        ctl_va, drug_va, cell_va, y_va = self.val_data
        tr = eval_pcc_mse(self.model, ctl_tr, drug_tr, cell_tr, y_tr, self.loss_mask, batch_size=self.batch_size, max_eval=self.max_eval)
        va = eval_pcc_mse(self.model, ctl_va, drug_va, cell_va, y_va, self.loss_mask, batch_size=self.batch_size, max_eval=self.max_eval)
        logs["pcc"] = tr["pcc"]
        logs["val_pcc"] = va["pcc"]
        print(f"Epoch {epoch+1}: pcc={tr['pcc']:.4f} val_pcc={va['pcc']:.4f}")


class GATWrapper(keras.Model):
    def __init__(self, gat_model, adj_matrix):
        super().__init__()
        self.gat = gat_model
        self.adj = tf.constant(adj_matrix, dtype=tf.float32)

    def call(self, inputs):
        ctl, drug_targets, cell_idx = inputs
        cell_idx = tf.cast(cell_idx, tf.int32)
        return self.gat([self.adj, ctl, drug_targets, cell_idx])


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    root = "/Users/liuxi/Desktop/RFA_GNN"
    if not os.path.exists(root):
        root = os.getcwd()
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from data_loader import load_rfa_data, build_combined_gnn
    from base_gnn import BaseLineGAT

    use_landmark_genes = True
    cell_line = None
    max_samples = 2000
    epochs = 50
    batch_size = 32

    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    string_path = os.path.join(root, "data/string_interactions_mapped.csv")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")

    data = load_rfa_data(
        ctl_path,
        trt_path,
        landmark_path=landmark_path,
        drug_target_path=drug_target_path,
        siginfo_path=siginfo_path,
        use_landmark_genes=use_landmark_genes,
        full_gene_path=full_gene_path,
        max_samples=max_samples,
        cell_lines=cell_line,
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        string_path=string_path,
        target_genes=data["target_genes"],
        confid_threshold=0.9,
        directed=False,
        symbol_to_entrez=data.get("symbol_to_entrez"),
    )
    if len(node_list) != len(data["target_genes"]) or node_list[:50] != data["target_genes"][:50]:
        raise ValueError("Graph node_list 与表达 target_genes 顺序/长度不一致")

    le = LabelEncoder()
    cell_idx = le.fit_transform(data["cell_names"])
    num_cells = int(len(le.classes_))

    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    unique_drugs = np.unique(drug_ids)
    np.random.seed(42)
    test_drugs = np.random.choice(unique_drugs, int(len(unique_drugs) * 0.2), replace=False)
    test_mask = np.isin(drug_ids, test_drugs)
    train_mask = ~test_mask

    X_ctl = np.asarray(data["X_ctl"])
    y_delta = np.asarray(data["y_delta"])
    X_drug = np.asarray(data["X_drug"])

    train_ctl = X_ctl[train_mask]
    train_trt = y_delta[train_mask]
    train_drug = X_drug[train_mask]
    train_cells = cell_idx[train_mask]

    test_ctl = X_ctl[test_mask]
    test_trt = y_delta[test_mask]
    test_drug = X_drug[test_mask]
    test_cells = cell_idx[test_mask]

    model = BaseLineGAT(
        num_genes=int(adj_matrix.shape[0]),
        num_cells=num_cells,
        hidden_dim=64,
        num_heads=4,
        dropout=0.2,
        use_residual=False,
        use_drug_embedding=False,
        attention_layer_number=4,
        output_after_embedding=False,
        per_node_embedding=True,
    )

    loss_mask = tf.constant(data["loss_mask"], dtype=tf.float32)

    def pcc_loss(y_true, y_pred):
        mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
        my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        xm = y_true - mx
        ym = y_pred - my
        r_num = tf.reduce_sum(xm * ym, axis=1)
        r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8)
        r = r_num / r_den
        return 1.0 - tf.reduce_mean(r)

    def masked_combined_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(loss_mask, tf.float32)
        mse = tf.reduce_sum(tf.square(y_true - y_pred) * mask)
        valid_count = tf.reduce_sum(mask)
        batch_n = tf.cast(tf.shape(y_true)[0], tf.float32)
        mse = mse / tf.maximum(valid_count * batch_n, 1.0)
        valid_indices = tf.where(loss_mask[0] > 0)[:, 0]
        yt = tf.gather(y_true, valid_indices, axis=1)
        yp = tf.gather(y_pred, valid_indices, axis=1)
        pcc = pcc_loss(yt, yp)
        return mse + 5.0 * pcc

    wrapped_model = GATWrapper(model, adj_matrix)
    wrapped_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=masked_combined_loss,
        metrics=[keras.metrics.MeanSquaredError()],
    )

    pcc_cb = PCCCallback(
        loss_mask=data["loss_mask"],
        train_data=(train_ctl, train_drug, train_cells, train_trt),
        val_data=(test_ctl, test_drug, test_cells, test_trt),
        batch_size=batch_size,
        max_eval=2048,
    )

    wrapped_model.fit(
        [train_ctl, train_drug, train_cells],
        train_trt,
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=[pcc_cb],
        validation_data=([test_ctl, test_drug, test_cells], test_trt),
        verbose=0,
    )

    train_metrics = eval_pcc_mse(wrapped_model, train_ctl, train_drug, train_cells, train_trt, data["loss_mask"], batch_size=batch_size, max_eval=20000)
    test_metrics = eval_pcc_mse(wrapped_model, test_ctl, test_drug, test_cells, test_trt, data["loss_mask"], batch_size=batch_size, max_eval=None)
    print(f"Train | MSE: {train_metrics['mse']:.4f} | Sample-wise PCC: {train_metrics['pcc']:.4f}")
    print(f"Test  | MSE: {test_metrics['mse']:.4f} | Sample-wise PCC: {test_metrics['pcc']:.4f}")


if __name__ == "__main__":
    main()
