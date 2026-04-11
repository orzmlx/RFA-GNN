import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GSNN"))

from data_loader import load_rfa_data, build_combined_gnn, subset_anchor_data, build_scheme_a_split_data
from gsnn.models.GSNN import GSNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_split_masks(split_mode, drug_ids, cell_idx, test_frac, seed=42):
    split_mode = str(split_mode).strip()
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")
    rng = np.random.default_rng(int(seed))
    n = len(drug_ids)
    if split_mode == "warm":
        if n < 2:
            raise ValueError("warm 需要至少 2 个样本")
        n_test = max(1, int(n * test_frac))
        n_test = min(n_test, n - 1)
        test_idx = rng.choice(np.arange(n), size=n_test, replace=False)
        test_mask = np.zeros((n,), dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        print(f"Split=warm | Held-out samples: {int(np.sum(test_mask))}/{n}")
        return train_mask, test_mask
    elif split_mode == "cold_cell":
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise ValueError("cold_cell 需要至少 2 个细胞系")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        test_cells_set = rng.choice(unique_cells, n_test, replace=False)
        test_mask = np.isin(cell_idx, test_cells_set)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(test_cells_set)}/{len(unique_cells)}")
        return train_mask, test_mask
    elif split_mode == "cold_drug":
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise ValueError("cold_drug 需要至少 2 个药物")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        test_drugs = rng.choice(unique_drugs, n_test, replace=False)
        test_mask = np.isin(drug_ids, test_drugs)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(test_drugs)}/{len(unique_drugs)}")
        return train_mask, test_mask
    raise ValueError(f"未知 split_mode: {split_mode}")


def parse_split_modes(raw, fallback):
    valid = {"warm", "cold_drug", "cold_cell"}
    s = str(raw).strip()
    if s == "":
        return [str(fallback)]
    modes = [t.strip() for t in s.split(",") if t.strip() != ""]
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"--split_modes 包含不支持的值: {bad}")
    seen = []
    for m in modes:
        if m not in seen:
            seen.append(m)
    return seen


def append_split_suffix(path, split_mode):
    s = str(path).strip()
    if s == "":
        return ""
    suffix = f".{str(split_mode).strip()}"
    stem, ext = os.path.splitext(s)
    return f"{stem}{suffix}{ext}"


def save_predictions_npz(npz_path, split_mode, y_true, y_pred, sample_pcc=None, drug_ids=None, cell_names=None, trt_distil_ids=None):
    out_dir = os.path.dirname(npz_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "split_mode": np.asarray(str(split_mode)),
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(y_pred, dtype=np.float32),
    }
    if sample_pcc is not None:
        payload["sample_pcc"] = np.asarray(sample_pcc, dtype=np.float32)
    if drug_ids is not None:
        payload["drug_ids"] = np.asarray(drug_ids, dtype=str)
    if cell_names is not None:
        payload["cell_names"] = np.asarray(cell_names, dtype=str)
    if trt_distil_ids is not None:
        payload["trt_distil_ids"] = np.asarray(trt_distil_ids, dtype=str)
    np.savez_compressed(npz_path, **payload)
    print("saved", npz_path)


def samplewise_pcc(y_true, y_pred):
    vals = []
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            vals.append(0.0)
        else:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return np.asarray(vals, dtype=np.float32)


def eval_metrics(model, loader, device):
    model.eval()
    ys = []
    yhats = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb)
            ys.append(yb.numpy())
            yhats.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yhats, axis=0)
    mse = float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    pcc_arr = samplewise_pcc(y_true, y_pred)
    return {
        "mse": mse,
        "pcc": float(np.mean(pcc_arr)),
        "sample_pcc": pcc_arr,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def build_gsnn_graph(node_list, edge_index):
    genes = [str(g) for g in node_list]
    input_nodes_ctl = [f"ctl::{g}" for g in genes]
    input_nodes_drug = [f"drug::{g}" for g in genes]
    input_nodes = input_nodes_ctl + input_nodes_drug
    function_nodes = [f"gene::{g}" for g in genes]
    output_nodes = [f"out::{g}" for g in genes]

    ff = torch.as_tensor(edge_index, dtype=torch.long)
    n = len(genes)
    inp_src = torch.arange(0, 2 * n, dtype=torch.long)
    inp_dst = torch.arange(0, n, dtype=torch.long).repeat(2)
    input_to_function = torch.stack([inp_src, inp_dst], dim=0)
    function_to_output = torch.stack([torch.arange(0, n, dtype=torch.long), torch.arange(0, n, dtype=torch.long)], dim=0)

    edge_index_dict = {
        ("input", "to", "function"): input_to_function,
        ("function", "to", "function"): ff,
        ("function", "to", "output"): function_to_output,
    }
    node_names_dict = {
        "input": input_nodes,
        "function": function_nodes,
        "output": output_nodes,
    }
    return edge_index_dict, node_names_dict


def train_one_split(split_mode, X_input, y, drug_ids, cell_idx, cell_names_arr, trt_distil_ids_arr, args, edge_index_dict, node_names_dict, device, predefined_train_size=None):
    if predefined_train_size is None:
        train_mask, test_mask = build_split_masks(split_mode, drug_ids, cell_idx, args.test_frac, seed=args.seed)
    else:
        n = len(X_input)
        train_n = int(predefined_train_size)
        train_mask = np.zeros((n,), dtype=bool)
        train_mask[:train_n] = True
        test_mask = ~train_mask
    X_train = torch.tensor(X_input[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32)
    X_test = torch.tensor(X_input[test_mask], dtype=torch.float32)
    y_test = torch.tensor(y[test_mask], dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GSNN(
        edge_index_dict=edge_index_dict,
        node_names_dict=node_names_dict,
        channels=args.channels,
        layers=args.layers,
        dropout=args.dropout,
        bias=True,
        share_layers=args.share_layers,
        add_function_self_edges=True,
        checkpoint=False,
        norm=args.norm,
        init="degree_normalized",
        residual=True,
        node_attn=False,
        node_mlp=True,
        node_mlp_hidden=args.node_mlp_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.log_every) == 0:
            train_metrics = eval_metrics(model, train_loader, device)
            test_metrics = eval_metrics(model, test_loader, device)
            print(
                f"[{split_mode}] Epoch {epoch}: "
                f"train_mse={train_metrics['mse']:.4f} train_pcc={train_metrics['pcc']:.4f} | "
                f"test_mse={test_metrics['mse']:.4f} test_pcc={test_metrics['pcc']:.4f}"
            )

    train_metrics = eval_metrics(model, train_loader, device)
    test_metrics = eval_metrics(model, test_loader, device)
    pred_prefix = str(args.save_pred_prefix).strip()
    if pred_prefix == "" and str(args.save_json).strip() != "":
        pred_prefix = os.path.splitext(str(args.save_json).strip())[0] + ".pred"
    pred_npz_path = None
    if pred_prefix != "":
        pred_npz_path = f"{pred_prefix}.{split_mode}.npz"
        save_predictions_npz(
            pred_npz_path,
            split_mode=split_mode,
            y_true=test_metrics["y_true"],
            y_pred=test_metrics["y_pred"],
            sample_pcc=test_metrics["sample_pcc"],
            drug_ids=drug_ids[test_mask],
            cell_names=cell_names_arr[test_mask],
            trt_distil_ids=trt_distil_ids_arr[test_mask],
        )
    out = {
        "split_mode": split_mode,
        "train_n": int(np.sum(train_mask)),
        "test_n": int(np.sum(test_mask)),
        "train_metrics": {"mse": train_metrics["mse"], "pcc": train_metrics["pcc"]},
        "test_metrics": {"mse": test_metrics["mse"], "pcc": test_metrics["pcc"]},
        "pred_npz": pred_npz_path,
    }
    return model, out, train_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="MCF7,LNCAP")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--pairing_mode", choices=["multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"], default="multi_trt_multi_ctl")
    parser.add_argument("--split_mode", choices=["warm", "cold_drug", "cold_cell"], default="cold_drug")
    parser.add_argument("--split_modes", default="warm,cold_drug,cold_cell")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--norm", default="none")
    parser.add_argument("--share_layers", action="store_true", default=False)
    parser.add_argument("--node_mlp_hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_json", default="")
    parser.add_argument("--save_pred_prefix", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    root = args.root
    tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
    ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    cell_lines = args.cell_line
    if cell_lines is not None:
        s = str(cell_lines).strip()
        if s == "" or s.upper() in {"ALL", "NONE", "NULL"}:
            cell_lines = None

    data = load_rfa_data(
        ctl_path,
        trt_path,
        drug_target_path=drug_target_path,
        landmark_path=landmark_path,
        siginfo_path=siginfo_path,
        fingerprint_path=fingerprint_path,
        use_landmark_genes=bool(args.use_landmark_genes),
        full_gene_path=full_gene_path,
        cell_lines=cell_lines,
        ctl_residual_pool_size=3,
        pairing_mode=str(args.pairing_mode),
    )
    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        target_genes=data["target_genes"],
        directed=True,
        symbol_to_entrez=data.get("symbol_to_entrez"),
    )

    anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
    anchor_cell_names_arr = np.asarray(data["anchor_cell_names"], dtype=str)
    anchor_trt_distil_ids_arr = np.asarray(data.get("anchor_trt_distil_ids", [""] * len(anchor_drug_ids)), dtype=str)

    if int(args.max_samples) > 0 and len(anchor_drug_ids) > int(args.max_samples):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(anchor_drug_ids), size=int(args.max_samples), replace=False)
        data = subset_anchor_data(data, idx)
        anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
        anchor_cell_names_arr = np.asarray(data["anchor_cell_names"], dtype=str)
        anchor_trt_distil_ids_arr = np.asarray(data.get("anchor_trt_distil_ids", [""] * len(anchor_drug_ids)), dtype=str)

    le = LabelEncoder()
    cell_idx = le.fit_transform(anchor_cell_names_arr)
    edge_index_dict, node_names_dict = build_gsnn_graph(node_list, edge_index)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device =", device)
    print("anchor_trt =", np.asarray(data["anchor_X_trt"]).shape, "anchor_drug =", np.asarray(data["anchor_X_drug"]).shape)

    results = []
    split_modes = parse_split_modes(args.split_modes, args.split_mode)
    for split_mode in split_modes:
        train_data, test_data, _, _ = build_scheme_a_split_data(
            data=data,
            split_mode=split_mode,
            test_frac=float(args.test_frac),
            seed=int(args.seed),
            train_pairing_mode=str(args.pairing_mode),
            train_ctl_pair_k=3,
            test_pairing_mode="unique_trt_reuse_ctl",
        )
        X_input_train = np.concatenate([np.asarray(train_data["X_ctl"], dtype=np.float32), np.asarray(train_data["X_drug"], dtype=np.float32)], axis=1).astype(np.float32)
        X_input_test = np.concatenate([np.asarray(test_data["X_ctl"], dtype=np.float32), np.asarray(test_data["X_drug"], dtype=np.float32)], axis=1).astype(np.float32)
        y_train = np.asarray(train_data["y_delta"], dtype=np.float32)
        y_test = np.asarray(test_data["y_delta"], dtype=np.float32)
        drug_ids = np.concatenate([np.asarray(train_data["drug_ids"], dtype=str), np.asarray(test_data["drug_ids"], dtype=str)], axis=0)
        cell_names_arr = np.concatenate([np.asarray(train_data["cell_names"], dtype=str), np.asarray(test_data["cell_names"], dtype=str)], axis=0)
        trt_distil_ids_arr = np.concatenate([np.asarray(train_data["trt_distil_ids"], dtype=str), np.asarray(test_data["trt_distil_ids"], dtype=str)], axis=0)
        split_cell_idx = le.transform(cell_names_arr)
        X_input = np.concatenate([X_input_train, X_input_test], axis=0)
        y_delta = np.concatenate([y_train, y_test], axis=0)
        train_n = len(X_input_train)
        _, out, _, _ = train_one_split(
            split_mode=split_mode,
            X_input=X_input,
            y=y_delta,
            drug_ids=drug_ids,
            cell_idx=split_cell_idx,
            cell_names_arr=cell_names_arr,
            trt_distil_ids_arr=trt_distil_ids_arr,
            args=args,
            edge_index_dict=edge_index_dict,
            node_names_dict=node_names_dict,
            device=device,
            predefined_train_size=train_n,
        )
        results.append(out)

    print("\n===== GSNN Summary =====")
    for r in results:
        print(
            f"{r['split_mode']}: "
            f"train_n={r['train_n']} test_n={r['test_n']} | "
            f"test_MSE={r['test_metrics']['mse']:.4f} | "
            f"test_PCC={r['test_metrics']['pcc']:.4f}"
        )

    if str(args.save_json).strip():
        out_path = str(args.save_json).strip()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        print("saved", out_path)


if __name__ == "__main__":
    main()
