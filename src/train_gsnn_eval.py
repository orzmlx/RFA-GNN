import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from train_common import (
    append_split_suffix,
    build_split_masks,
    parse_split_modes,
    samplewise_pcc,
    save_predictions_npz,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GSNN"))

from data_loader import load_rfa_data, build_combined_gnn
from gsnn.models.GSNN import GSNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    mse = float(np.mean((y_true - y_pred) ** 2))
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


def train_one_split(split_mode, X_input, y, drug_ids, cell_idx, cell_names_arr, trt_distil_ids_arr, X_drug, args, edge_index_dict, node_names_dict, device):
    train_mask, test_mask = build_split_masks(
        split_mode,
        drug_ids,
        cell_idx,
        args.test_frac,
        seed=args.seed,
        drug_target_matrix=X_drug,
    )
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
    parser.add_argument("--split_mode", choices=["warm", "cold_drug", "cold_cell", "cold_target_pattern"], default="cold_drug")
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
    )
    adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path=tf_path,
        ppi_path=ppi_path,
        target_genes=data["target_genes"],
        directed=True,
        symbol_to_entrez=data.get("symbol_to_entrez"),
    )

    X_ctl = np.asarray(data["X_ctl"], dtype=np.float32)
    y_delta = np.asarray(data["y_delta"], dtype=np.float32)
    X_drug = np.asarray(data["X_drug"], dtype=np.float32)
    drug_ids = np.asarray(data["drug_ids"], dtype=str)
    cell_names_arr = np.asarray(data["cell_names"], dtype=str)
    trt_distil_ids_arr = np.asarray(data.get("trt_distil_ids", [""] * len(drug_ids)), dtype=str)

    if int(args.max_samples) > 0 and len(X_ctl) > int(args.max_samples):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_ctl), size=int(args.max_samples), replace=False)
        X_ctl = X_ctl[idx]
        y_delta = y_delta[idx]
        X_drug = X_drug[idx]
        drug_ids = drug_ids[idx]
        cell_names_arr = cell_names_arr[idx]
        trt_distil_ids_arr = trt_distil_ids_arr[idx]

    X_input = np.concatenate([X_ctl, X_drug], axis=1).astype(np.float32)
    le = LabelEncoder()
    cell_idx = le.fit_transform(cell_names_arr)
    edge_index_dict, node_names_dict = build_gsnn_graph(node_list, edge_index)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device =", device)
    print("X_input =", X_input.shape, "y_delta =", y_delta.shape)

    results = []
    split_modes = parse_split_modes(args.split_modes, args.split_mode)
    for split_mode in split_modes:
        _, out, _, _ = train_one_split(
            split_mode=split_mode,
            X_input=X_input,
            y=y_delta,
            drug_ids=drug_ids,
            cell_idx=cell_idx,
            cell_names_arr=cell_names_arr,
            trt_distil_ids_arr=trt_distil_ids_arr,
            X_drug=X_drug,
            args=args,
            edge_index_dict=edge_index_dict,
            node_names_dict=node_names_dict,
            device=device,
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
