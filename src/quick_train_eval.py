import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

ROOT = "/Users/liuxi/Desktop/RFA_GNN"
if not os.path.exists(ROOT):
    ROOT = os.getcwd()
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from base_gnn import BaseLineGAT
from data_loader import build_combined_gnn, load_rfa_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--confid_threshold", type=float, default=0.9)
    parser.add_argument("--directed", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    max_samples = int(args.max_samples)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    tf_path = os.path.join(ROOT, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(ROOT, "data/omnipath/omnipath_interactions.csv")
    string_path = os.path.join(ROOT, "data/string_interactions_mapped.csv")
    full_gene_path = os.path.join(ROOT, "data/GSE92742_Broad_LINCS_gene_info.txt")
    siginfo_path = os.path.join(ROOT, "data/siginfo_beta.txt")
    landmark_path = os.path.join(ROOT, "data/landmark_genes.json")
    ctl_path = os.path.join(ROOT, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(ROOT, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")

    with open(landmark_path, "r") as f:
        lm = json.load(f)
    if isinstance(lm, dict) and "landmark_genes" in lm:
        lm_list = lm["landmark_genes"]
    elif isinstance(lm, list):
        lm_list = lm
    else:
        lm_list = []
    landmark_genes = [
        str(g.get("entrez_id") or g.get("pr_gene_id"))
        for g in lm_list
        if isinstance(g, dict) and (g.get("entrez_id") or g.get("pr_gene_id"))
    ]
    landmark_genes = [g for g in landmark_genes if g and g != "None"]

    data = load_rfa_data(
        ctl_path,
        trt_path,
        landmark_path=landmark_path,
        siginfo_path=siginfo_path,
        use_landmark_genes=True,
        full_gene_path=full_gene_path,
        max_samples=max_samples,
    )
    if data is None:
        raise RuntimeError("load_rfa_data returned None")

    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        string_path=string_path,
        full_gene_path=full_gene_path,
        landmark_path=landmark_path,
        landmark_genes=data["target_genes"],
        confid_threshold=float(args.confid_threshold),
        directed=bool(args.directed),
    )

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
    if X_ctl.ndim == 3:
        # BaseLineGAT expects (B, N) or (B, N, 1); drop extra channels if present.
        if X_ctl.shape[-1] > 1:
            X_ctl = X_ctl[..., 0]
        elif X_ctl.shape[-1] == 1:
            X_ctl = X_ctl[..., 0]
    y = np.asarray(data["y_delta"])
    X_drug = np.asarray(data["X_drug"])

    train_ctl, train_drug, train_cells, train_y = (
        X_ctl[train_mask],
        X_drug[train_mask],
        cell_idx[train_mask],
        y[train_mask],
    )
    test_ctl, test_drug, test_cells, test_y = (
        X_ctl[test_mask],
        X_drug[test_mask],
        cell_idx[test_mask],
        y[test_mask],
    )

    num_genes = int(adj_matrix.shape[0])
    base = BaseLineGAT(
        num_genes=num_genes,
        num_cells=num_cells,
        hidden_dim=64,
        num_heads=4,
        dropout=0.2,
        use_residual=False,
        use_drug_embedding=False,
    )
    loss_mask = tf.constant(data["loss_mask"], dtype=tf.float32)

    def pcc_loss(y_true, y_pred):
        mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
        my = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        xm = y_true - mx
        ym = y_pred - my
        r_num = tf.reduce_sum(xm * ym, axis=1)
        r_den = tf.sqrt(
            tf.reduce_sum(tf.square(xm), axis=1) * tf.reduce_sum(tf.square(ym), axis=1) + 1e-8
        )
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
        valid_idx = tf.where(loss_mask[0] > 0)[:, 0]
        yt = tf.gather(y_true, valid_idx, axis=1)
        yp = tf.gather(y_pred, valid_idx, axis=1)
        return mse + 5.0 * pcc_loss(yt, yp)

    class Wrap(keras.Model):
        def __init__(self, m, adj):
            super().__init__()
            self.m = m
            self.adj = tf.constant(adj, dtype=tf.float32)

        def call(self, inputs):
            ctl, drug, cell = inputs
            return self.m([self.adj, ctl, drug, tf.cast(cell, tf.int32)])

    model = Wrap(base, adj_matrix)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss=masked_combined_loss)

    model.fit(
        [train_ctl, train_drug, train_cells],
        train_y,
        validation_data=([test_ctl, test_drug, test_cells], test_y),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred = model.predict([test_ctl, test_drug, test_cells], batch_size=batch_size, verbose=0)
    valid_indices = np.where(np.asarray(data["loss_mask"])[0] > 0)[0]
    true_v = test_y[:, valid_indices]
    pred_v = pred[:, valid_indices]

    pcc = []
    for i in range(len(true_v)):
        yt = true_v[i]
        yp = pred_v[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            pcc.append(pearsonr(yt, yp)[0])
    avg_pcc = float(np.mean(pcc)) if pcc else 0.0
    global_pcc = float(pearsonr(true_v.flatten(), pred_v.flatten())[0])
    print("test_n", int(len(true_v)))
    print("avg_sample_pcc", avg_pcc)
    print("global_pcc", global_pcc)


if __name__ == "__main__":
    main()
