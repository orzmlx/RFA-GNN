import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from scipy.stats import pearsonr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_rfa_data
from lppn import LPPN, topk_pcc_loss, sign_mismatch_loss


class SubgraphManager:
    def __init__(self, edges_csv_path, gene_list):
        self.gene_list = gene_list
        self.gene2idx = {g: i for i, g in enumerate(gene_list)}
        self.G = self._build_graph(edges_csv_path)

    def _build_graph(self, csv_path):
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.gene_list)))
        df = pd.read_csv(csv_path)
        edge_map = {}
        for _, row in df.iterrows():
            u = str(int(float(row["source"]))) if str(row["source"]).replace(".", "", 1).isdigit() else str(row["source"])
            v = str(int(float(row["target"]))) if str(row["target"]).replace(".", "", 1).isdigit() else str(row["target"])
            if u in self.gene2idx and v in self.gene2idx:
                uid, vid = self.gene2idx[u], self.gene2idx[v]
                w = float(row.get("weight", 1.0))
                key = (uid, vid)
                if key not in edge_map or abs(w) > abs(edge_map[key]):
                    edge_map[key] = w
        for (uid, vid), w in edge_map.items():
            G.add_edge(uid, vid, weight=w)
        return G

    def _multi_source_dist(self, successors_fn, sources, cutoff):
        dist = {}
        q = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= cutoff:
                continue
            for v in successors_fn(u):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def pre_prune(self, roots, leafs, depth=10):
        import networkx as nx

        roots = [int(x) for x in roots if int(x) in self.G]
        leafs = [int(x) for x in leafs if int(x) in self.G]
        if not roots or not leafs:
            return

        dist_r = self._multi_source_dist(self.G.successors, roots, depth)
        rG = self.G.reverse(copy=False)
        dist_l = self._multi_source_dist(rG.successors, leafs, depth)

        keep = set(roots) | set(leafs)
        for n, dr in dist_r.items():
            dl = dist_l.get(n)
            if dl is None:
                continue
            if dr + dl <= depth:
                keep.add(n)
        before_nodes = self.G.number_of_nodes()
        before_edges = self.G.number_of_edges()
        self.G = self.G.subgraph(sorted(list(keep))).copy()
        print(f"Pre-pruned graph: {before_nodes}->{self.G.number_of_nodes()} nodes, {before_edges}->{self.G.number_of_edges()} edges.")

    def extract_subgraph(self, target_indices, k_hops=3, max_nodes=200):
        import networkx as nx

        current_nodes = set(int(x) for x in target_indices)
        for _ in range(k_hops):
            neighbors = set()
            for node in current_nodes:
                if node in self.G:
                    neighbors.update(self.G.successors(node))
                    neighbors.update(self.G.predecessors(node))
            if len(current_nodes) + len(neighbors) > max_nodes:
                needed = max_nodes - len(current_nodes)
                if needed > 0:
                    current_nodes.update(list(neighbors)[:needed])
                break
            current_nodes.update(neighbors)

        sub_nodes = np.array(sorted(list(current_nodes)), dtype=np.int32)
        sub_G = self.G.subgraph(sub_nodes)
        sub_adj = nx.to_numpy_array(sub_G, nodelist=sub_nodes, weight="weight", dtype=np.float32)
        np.fill_diagonal(sub_adj, 0.0)

        und = sub_G.to_undirected()
        pos_map = {int(g): i for i, g in enumerate(sub_nodes.tolist())}
        sources = [int(x) for x in target_indices if int(x) in pos_map]
        sources_pos = [pos_map[s] for s in sources]
        dist = np.full((len(sub_nodes),), fill_value=float(k_hops + 1), dtype=np.float32)
        if sources_pos:
            q = deque(sources_pos)
            for s in sources_pos:
                dist[s] = 0.0
            while q:
                u = q.popleft()
                u_global = int(sub_nodes[u])
                for v_global in und.neighbors(u_global):
                    vpos = pos_map.get(int(v_global))
                    if vpos is None:
                        continue
                    if dist[vpos] > dist[u] + 1:
                        dist[vpos] = dist[u] + 1
                        q.append(vpos)

        return sub_nodes, sub_adj, dist


def mean_topk_pcc(y_true, y_pred, k=200):
    vals = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        kk = min(int(k), yt.shape[0])
        if kk < 2:
            continue
        idx = np.argpartition(np.abs(yt), -kk)[-kk:]
        yt_k = yt[idx]
        yp_k = yp[idx]
        if np.std(yt_k) < 1e-6 or np.std(yp_k) < 1e-6:
            continue
        vals.append(float(pearsonr(yt_k, yp_k)[0]))
    return float(np.mean(vals)) if vals else 0.0


def train_lppn(
    overfit=False,
    k_steps=3,
    learn_gate=True,
    learn_source=True,
    epochs=None,
):
    data = load_rfa_data(
        "../data/cmap/level3_beta_ctl_n188708x12328.h5",
        "../data/cmap/level3_beta_trt_cp_n1805898x12328.h5",
        landmark_path="../data/landmark_genes.json",
        siginfo_path="../data/siginfo_beta.txt",
        use_landmark_genes=True,
        max_samples=50000,
    )
    if data is None:
        raise RuntimeError("load_rfa_data failed")

    target_genes = data["target_genes"]
    X_ctl = data["X_ctl"]
    y_delta = data["y_delta"]
    drug_ids = np.array(data["drug_ids"], dtype=str)
    cell_names = np.array(data.get("cell_names", []), dtype=str)
    X_fp = data.get("X_fingerprint", None)

    if X_fp is None:
        raise RuntimeError("X_fingerprint missing; check data_loader load_rfa_data")
    if cell_names.size == 0:
        raise RuntimeError("cell_names missing; check data_loader load_rfa_data")

    unique_cells, cell_idx = np.unique(cell_names, return_inverse=True)
    num_cells = len(unique_cells)
    fp_dim = int(X_fp.shape[1])

    graph_path = "data/rfa_directed_edges.csv"
    manager = SubgraphManager(graph_path, target_genes)

    all_target_mask = np.sum(X_ctl[:, :, 1], axis=0)
    roots = np.where(all_target_mask > 0)[0].astype(np.int32).tolist()
    out_scores = np.mean(np.abs(y_delta), axis=0)
    leafs = np.argsort(out_scores)[::-1][:300].astype(np.int32).tolist()
    manager.pre_prune(roots, leafs, depth=10)

    if overfit:
        pair_to_indices = {}
        for i in range(len(drug_ids)):
            pair_to_indices.setdefault((drug_ids[i], cell_idx[i]), []).append(i)
        (drug_id, cidx), idxs = max(pair_to_indices.items(), key=lambda kv: len(kv[1]))
        idxs = np.array(idxs, dtype=np.int32)[:256]
        print(f"Overfit pair: drug={drug_id} cell={unique_cells[int(cidx)]} n={len(idxs)}")
        train_drugs = [drug_id]
        test_drugs = [drug_id]
    else:
        unique_drugs = np.unique(drug_ids).tolist()
        rng = np.random.default_rng(0)
        rng.shuffle(unique_drugs)
        split = int(0.8 * len(unique_drugs))
        train_drugs = unique_drugs[:split]
        test_drugs = unique_drugs[split:]

    drug_to_fp = {}
    for i, did in enumerate(drug_ids.tolist()):
        if did not in drug_to_fp:
            drug_to_fp[did] = X_fp[i]

    drug_groups = {}
    for did in np.unique(drug_ids):
        inds = np.where(drug_ids == did)[0]
        if inds.size == 0:
            continue
        target_mask = X_ctl[inds[0], :, 1]
        targets = np.where(target_mask > 0)[0].astype(np.int32)
        if targets.size == 0:
            continue
        drug_groups[did] = {"samples": inds.astype(np.int32), "targets": targets}

    if overfit and train_drugs:
        did = train_drugs[0]
        if did in drug_groups:
            group = drug_groups[did]
            t = group["targets"]
            y_grp = y_delta[idxs]
            mean_abs = np.mean(np.abs(y_grp[:, t]), axis=0)
            keep_n = min(2, t.shape[0])
            keep_idx = np.argsort(mean_abs)[::-1][:keep_n]
            group["targets"] = t[keep_idx]
            drug_groups[did] = group

    model = LPPN(
        num_genes=len(target_genes),
        hidden_dim=64,
        k_steps=k_steps,
        fp_dim=fp_dim,
        num_cells=num_cells,
        learn_gate=learn_gate,
        learn_source=learn_source,
        dropout=0.1,
    )
    opt = keras.optimizers.Adam(learning_rate=5e-4)

    topk = 200
    lam_sign = 0.1
    lam_gate = 1e-3
    lam_depth = 0.0
    warmup_epochs = 5

    if epochs is None:
        epochs = 50 if overfit else 10
    epochs = int(epochs)

    for epoch in range(epochs):
        train_pcc_vals = []
        diag_h = []
        diag_gate = []
        diag_edge_on = []
        for did in train_drugs:
            if did not in drug_groups:
                continue
            group = drug_groups[did]
            sub_nodes, sub_adj, dist = manager.extract_subgraph(group["targets"], k_hops=3)
            if sub_nodes.size < 2:
                continue
            sample_idx = group["samples"]
            if overfit:
                sample_idx = sample_idx[:256]
            batch_X = tf.gather(X_ctl, sample_idx)
            batch_X = tf.gather(batch_X, sub_nodes, axis=1)
            batch_y = tf.gather(y_delta, sample_idx)
            batch_y = tf.gather(batch_y, sub_nodes, axis=1)
            batch_adj = tf.expand_dims(sub_adj, 0)
            batch_cell = tf.gather(cell_idx, sample_idx)
            batch_fp = tf.convert_to_tensor(np.repeat(drug_to_fp[did][None, :], repeats=len(sample_idx), axis=0), dtype=tf.float32)
            with tf.GradientTape() as tape:
                pred, metrics = model([batch_X, batch_adj, batch_cell, batch_fp, dist, sub_nodes], training=True)
                if epoch < warmup_epochs:
                    l_topk = tf.reduce_mean(tf.square(batch_y - pred))
                else:
                    l_topk = topk_pcc_loss(batch_y, pred, k=topk)
                l_sign = sign_mismatch_loss(batch_y, pred, k=topk)
                l_gate = metrics["z_mean"]
                l_depth = metrics["depth_pen"]
                loss = l_topk + lam_sign * l_sign + lam_gate * l_gate + lam_depth * l_depth
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            if epoch >= warmup_epochs:
                train_pcc_vals.append(float(1.0 - l_topk.numpy()))
            diag_h.append(float(metrics["h_abs_mean"].numpy()))
            diag_gate.append(float(metrics["z_mean"].numpy()))
            diag_edge_on.append(float(metrics["z_gt_half"].numpy()))

        train_pcc = float(np.mean(train_pcc_vals)) if train_pcc_vals else 0.0
        h_abs = float(np.mean(diag_h)) if diag_h else 0.0
        gate_abs = float(np.mean(diag_gate)) if diag_gate else 0.0
        edge_on = float(np.mean(diag_edge_on)) if diag_edge_on else 0.0

        test_scores = []
        for did in test_drugs:
            if did not in drug_groups:
                continue
            group = drug_groups[did]
            sub_nodes, sub_adj, dist = manager.extract_subgraph(group["targets"], k_hops=3)
            if sub_nodes.size < 2:
                continue
            sample_idx = group["samples"]
            batch_X = tf.gather(X_ctl, sample_idx)
            batch_X = tf.gather(batch_X, sub_nodes, axis=1)
            batch_y = tf.gather(y_delta, sample_idx)
            batch_y = tf.gather(batch_y, sub_nodes, axis=1)
            batch_adj = tf.expand_dims(sub_adj, 0)
            batch_cell = tf.gather(cell_idx, sample_idx)
            batch_fp = tf.convert_to_tensor(np.repeat(drug_to_fp[did][None, :], repeats=len(sample_idx), axis=0), dtype=tf.float32)
            pred, _ = model([batch_X, batch_adj, batch_cell, batch_fp, dist, sub_nodes], training=False)
            test_scores.append(mean_topk_pcc(batch_y.numpy(), pred.numpy(), k=topk))
        test_pcc = float(np.mean(test_scores)) if test_scores else 0.0
        if test_scores:
            q25, q50, q75 = np.quantile(np.array(test_scores, dtype=np.float32), [0.25, 0.5, 0.75]).tolist()
            print(f"Epoch {epoch+1}: Train TopK PCC = {train_pcc:.4f} | Test mean={test_pcc:.4f} p25={q25:.4f} p50={q50:.4f} p75={q75:.4f} | mean|h|={h_abs:.4f} mean|gate|={gate_abs:.4f} edge>0.5={edge_on:.3f}")
        else:
            print(f"Epoch {epoch+1}: Train TopK PCC = {train_pcc:.4f} | Test TopK PCC = {test_pcc:.4f} | mean|h|={h_abs:.4f} mean|gate|={gate_abs:.4f} edge>0.5={edge_on:.3f}")


if __name__ == "__main__":
    overfit = "--overfit" in sys.argv
    learn_gate = "--no-gate" not in sys.argv
    learn_source = "--no-source" not in sys.argv
    k_steps = 3
    epochs = None
    for i, a in enumerate(sys.argv):
        if a == "--k" and i + 1 < len(sys.argv):
            k_steps = int(sys.argv[i + 1])
        if a == "--epochs" and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
    train_lppn(overfit=overfit, k_steps=k_steps, learn_gate=learn_gate, learn_source=learn_source, epochs=epochs)
