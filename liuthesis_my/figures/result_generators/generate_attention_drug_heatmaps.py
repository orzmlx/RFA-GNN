from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
OUT_DIR = ROOT / "liuthesis_my/figures"
COLD_DRUG_NPZ = ROOT / "results/cold_target_pattern/cagnn_control_context.eval.cold_target_pattern.attn.npz"
WARM_NPZ = ROOT / "results/gat_hybrid_uncertainty_all_splits.eval.warm.attn.npz"
COLD_CELL_NPZ = ROOT / "results/gat_hybrid_uncertainty_all_splits.eval.cold_cell.attn.npz"
GENE_INFO_PATH = ROOT / "data/GSE92742_Broad_LINCS_gene_info.txt"
COMPOUND_TARGETS_PATH = ROOT / "data/compound_targets.txt"
MIN_N = 10
ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attention_soft_blue",
    ["#F7FBFF", "#D7EFF3", "#92C5DE", "#4393C3", "#2166AC"],
)
TARGET_LINK_CONFIGS = [
    {
        "drug_id": "BRD-K86797399",
        "title": "Cold drug target: BRD-K86797399",
        "focus_edges": [("NFE2L2", "TXNRD1"), ("NFE2L2", "SDHB")],
        "validated_paths": [
            ["HDAC2", "YY1", "MYC", "NFE2L2"],
        ],
    },
    {
        "drug_id": "BRD-K56301217",
        "title": "Cold cell: BRD-K56301217",
        "focus_edges": [("TP53", "DUSP11"), ("TP53", "REEP5")],
        "validated_paths": [
            ["BCL2", "BECN1", "USP10", "TP53"],
        ],
    },
]


def load_run(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    run = {}
    for k in z.files:
        v = z[k]
        if k == "attention":
            run[k] = v[0]
            continue
        if v.dtype == object and v.shape == (1,) and isinstance(v[0], str):
            try:
                run[k] = json.loads(v[0])
                continue
            except Exception:
                pass
        if v.dtype == object and v.shape == (1,) and isinstance(v[0], (dict, list)):
            run[k] = v[0]
            continue
        run[k] = v
    if "sample_pcc" not in run and str(npz_path).endswith(".attn.npz"):
        base_path = Path(str(npz_path).replace(".attn.npz", ".npz"))
        if base_path.exists():
            z_base = np.load(base_path, allow_pickle=True)
            run["_base_drug_ids"] = z_base["drug_ids"] if "drug_ids" in z_base.files else None
            run["_base_trt_distil_ids"] = z_base["trt_distil_ids"] if "trt_distil_ids" in z_base.files else None
            for k in z_base.files:
                if k not in run:
                    run[k] = z_base[k]
    return run


def load_gene_symbol_map():
    gene_info = pd.read_csv(GENE_INFO_PATH, sep="\t", dtype=str)
    return {
        str(row["pr_gene_id"]): str(row["pr_gene_symbol"])
        for _, row in gene_info.iterrows()
    }


def load_drug_targets():
    df = pd.read_csv(
        COMPOUND_TARGETS_PATH,
        sep="\t",
        header=None,
        names=["drug_id", "drug_name", "target", "moa", "smiles", "inchikey", "alias", "uniprot"],
        dtype=str,
    )
    df = df[["drug_id", "target"]].dropna()
    grouped = df.groupby("drug_id")["target"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    return {str(k): v for k, v in grouped.items()}


def build_ranked_drug_table(run):
    pcc = np.asarray(run["sample_pcc"], dtype=float)
    drug_ids = np.asarray(run["drug_ids"], dtype=str)
    if len(drug_ids) != len(pcc) and run.get("_base_drug_ids") is not None:
        drug_ids = np.asarray(run["_base_drug_ids"], dtype=str)
    df = pd.DataFrame({"drug": drug_ids, "pcc": pcc})
    ranked = (
        df.groupby("drug")
        .agg(n=("pcc", "size"), median_pcc=("pcc", "median"), mean_pcc=("pcc", "mean"))
        .reset_index()
        .sort_values(["median_pcc", "mean_pcc"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return ranked


def build_heatmap_matrix(run, selected_drugs, top_k_edges=20):
    att = run["attention"]
    edge_index = np.asarray(att["edge_index"], dtype=np.int32)
    src = edge_index[0]
    dst = edge_index[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]

    # Build a union of top edges across the selected drugs.
    edge_union = []
    seen = set()
    per_drug_vectors = {}
    for drug in selected_drugs:
        alpha = np.asarray(att["group_attention_edge_mean"][drug], dtype=float)[-1][non_self]
        top_idx = np.argsort(-alpha)[: int(top_k_edges)]
        per_drug_vectors[drug] = alpha
        for i in top_idx:
            key = (int(src[i]), int(dst[i]))
            if key not in seen:
                seen.add(key)
                edge_union.append(key)

    mat = np.zeros((len(selected_drugs), len(edge_union)), dtype=np.float32)
    for r, drug in enumerate(selected_drugs):
        alpha = per_drug_vectors[drug]
        edge_to_val = {
            (int(src[i]), int(dst[i])): float(alpha[i]) for i in range(len(alpha))
        }
        vals = np.asarray([edge_to_val.get(e, 0.0) for e in edge_union], dtype=np.float32)
        vmax = float(np.max(vals)) if len(vals) else 0.0
        if vmax > 0:
            vals = vals / vmax
        mat[r] = vals
    return mat


def plot_heatmap(run, selected_df, out_path: Path, title: str, top_k_edges=20):
    drugs = selected_df["drug"].tolist()
    mat = build_heatmap_matrix(run, drugs, top_k_edges=top_k_edges)
    ylabels = [
        f"{row.drug} | {row.median_pcc:.3f}"
        for row in selected_df.itertuples(index=False)
    ]

    plt.figure(figsize=(11, max(4.8, 0.34 * len(drugs))), dpi=220)
    im = plt.imshow(mat, aspect="auto", cmap=ATTN_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=8)
    plt.xticks([])
    plt.ylabel("Drugs ordered by median sample-wise PCC")
    plt.xlabel(f"Union of per-drug top-{int(top_k_edges)} edges")
    plt.title(title)
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Within-drug scaled attention")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def top_edges_for_drug(run, drug, top_k_edges=20):
    att = run["attention"]
    edge_index = np.asarray(att["edge_index"], dtype=np.int32)
    src = edge_index[0]
    dst = edge_index[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    alpha = np.asarray(att["group_attention_edge_mean"][drug], dtype=float)[-1][non_self]
    top_idx = np.argsort(-alpha)[: int(top_k_edges)]
    return [(int(src[i]), int(dst[i])) for i in top_idx], alpha, src, dst


def plot_shared_drug_comparison(warm_run, cold_cell_run, drug, out_path: Path, top_k_edges=20):
    warm_edges, warm_alpha, warm_src, warm_dst = top_edges_for_drug(warm_run, drug, top_k_edges=top_k_edges)
    cold_edges, cold_alpha, cold_src, cold_dst = top_edges_for_drug(cold_cell_run, drug, top_k_edges=top_k_edges)

    edge_union = []
    seen = set()
    for edges in [warm_edges, cold_edges]:
        for e in edges:
            if e not in seen:
                seen.add(e)
                edge_union.append(e)

    def edge_values(alpha, src, dst):
        edge_to_val = {(int(src[i]), int(dst[i])): float(alpha[i]) for i in range(len(alpha))}
        vals = np.asarray([edge_to_val.get(e, 0.0) for e in edge_union], dtype=np.float32)
        vmax = float(np.max(vals)) if len(vals) else 0.0
        if vmax > 0:
            vals = vals / vmax
        return vals

    warm_vals = edge_values(warm_alpha, warm_src, warm_dst)
    cold_vals = edge_values(cold_alpha, cold_src, cold_dst)
    mat = np.stack([warm_vals, cold_vals], axis=0)

    plt.figure(figsize=(11, 3.6), dpi=220)
    im = plt.imshow(mat, aspect="auto", cmap=ATTN_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.yticks([0, 1], ["Warm", "Cold cell"], fontsize=10)
    plt.xticks([])
    plt.ylabel(f"{drug}")
    plt.xlabel(f"Union of top-{int(top_k_edges)} edges for the same drug")
    plt.title("Attention comparison for one drug across cell settings")
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Within-split scaled attention")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def build_graph_from_run(run):
    edge_index = np.asarray(run["attention"]["edge_index"], dtype=np.int32)
    src = edge_index[0]
    dst = edge_index[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    n_nodes = len(np.asarray(run["target_genes"], dtype=str))
    g_dir = nx.DiGraph()
    g_dir.add_nodes_from(range(n_nodes))
    g_dir.add_edges_from([(int(s), int(d)) for s, d in zip(src, dst)])
    g_und = g_dir.to_undirected()
    return g_dir, g_und


def node_labels_for_run(run, id_to_symbol):
    genes = np.asarray(run["target_genes"], dtype=str)
    return [id_to_symbol.get(str(g), str(g)) for g in genes]


def drug_target_node_indices(run, drug_id, id_to_symbol, drug_targets):
    labels = node_labels_for_run(run, id_to_symbol)
    symbol_to_idx = {str(symbol): idx for idx, symbol in enumerate(labels)}
    return sorted(
        {
            int(symbol_to_idx[target])
            for target in drug_targets.get(str(drug_id), [])
            if str(target) in symbol_to_idx
        }
    )


def edge_lookup_by_symbol(run, id_to_symbol):
    labels = node_labels_for_run(run, id_to_symbol)
    edge_index = np.asarray(run["attention"]["edge_index"], dtype=np.int32)
    src = edge_index[0]
    dst = edge_index[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    symbol_edges = {}
    for i in range(len(src)):
        symbol_edges[(labels[int(src[i])], labels[int(dst[i])])] = (int(src[i]), int(dst[i]))
    return symbol_edges


def selected_top_edges(run, drug_id, id_to_symbol, focus_edge_symbols, extra_connected=3):
    labels = node_labels_for_run(run, id_to_symbol)
    symbol_edges = edge_lookup_by_symbol(run, id_to_symbol)
    top_edges, alpha, src, dst = top_edges_for_drug(run, drug_id, top_k_edges=80)
    edge_to_att = {(int(src[i]), int(dst[i])): float(alpha[i]) for i in range(len(alpha))}
    selected = []
    selected_set = set()
    focus_nodes = set()
    for pair in focus_edge_symbols:
        edge = symbol_edges.get(pair)
        if edge is None:
            continue
        selected.append(edge)
        selected_set.add(edge)
        focus_nodes.update(edge)

    ranked = sorted(top_edges, key=lambda e: edge_to_att.get(e, 0.0), reverse=True)
    for edge in ranked:
        if edge in selected_set:
            continue
        if edge[0] in focus_nodes or edge[1] in focus_nodes:
            selected.append(edge)
            selected_set.add(edge)
            if len(selected) >= len(focus_edge_symbols) + int(extra_connected):
                break
    return selected


def symbol_to_index_map(run, id_to_symbol):
    labels = node_labels_for_run(run, id_to_symbol)
    return {str(symbol): idx for idx, symbol in enumerate(labels)}


def validated_path_edges(run, id_to_symbol, validated_paths):
    sym2idx = symbol_to_index_map(run, id_to_symbol)
    kept_paths = []
    path_edges = set()
    path_nodes = set()
    for path in validated_paths:
        idx_path = []
        ok = True
        for symbol in path:
            if symbol not in sym2idx:
                ok = False
                break
            idx_path.append(int(sym2idx[symbol]))
        if not ok or len(idx_path) < 3:
            continue
        kept_paths.append(idx_path)
        path_nodes.update(idx_path)
        for a, b in zip(idx_path[:-1], idx_path[1:]):
            path_edges.add((int(a), int(b)))
    return kept_paths, path_edges, path_nodes


def plot_targets_to_top_edges_panel(
    ax,
    run,
    drug_id,
    panel_title,
    id_to_symbol,
    drug_targets,
    focus_edge_symbols,
    validated_paths,
):
    g_dir, g_und = build_graph_from_run(run)
    labels = node_labels_for_run(run, id_to_symbol)
    top_edges = selected_top_edges(run, drug_id, id_to_symbol, focus_edge_symbols, extra_connected=3)
    endpoints = sorted(set([u for u, v in top_edges] + [v for u, v in top_edges]))
    targets = drug_target_node_indices(run, drug_id, id_to_symbol, drug_targets)
    _, path_edges, path_nodes = validated_path_edges(run, id_to_symbol, validated_paths)

    nodes = set(targets) | set(endpoints) | set(path_nodes)
    for a, b in list(path_edges) + list(top_edges):
        nodes.add(int(a))
        nodes.add(int(b))
    subg = g_dir.subgraph(nodes).copy()

    if len(targets) > 0 and len(path_nodes) > 0:
        dist_map = nx.multi_source_dijkstra_path_length(g_und, sources=targets, weight=None, cutoff=3)
    else:
        dist_map = {}
    maxd = max(dist_map.values()) if len(dist_map) > 0 else 0
    layers = {}
    for node in subg.nodes():
        d = dist_map.get(int(node), maxd + 1)
        layers.setdefault(int(d), []).append(int(node))
    for d in layers:
        layers[d].sort(key=lambda x: labels[int(x)])

    pos = {}
    for d, ns in sorted(layers.items(), key=lambda kv: kv[0]):
        m = len(ns)
        ys = np.linspace(-(m - 1) / 2.0, (m - 1) / 2.0, m) if m > 1 else np.array([0.0])
        for node, y in zip(ns, ys):
            pos[int(node)] = (float(d) * 2.15, float(y) * 1.15)

    top_edge_set = set((int(a), int(b)) for a, b in top_edges)
    other_nodes = [n for n in subg.nodes() if n not in set(targets) and n not in set(endpoints) and n not in path_nodes]
    mid_nodes = [n for n in path_nodes if n not in set(targets) and n not in set(endpoints)]

    if len(path_edges) > 0:
        nx.draw_networkx_edges(subg, pos, edgelist=list(path_edges), width=2.7, alpha=0.95, edge_color="#4C78A8", arrows=False, ax=ax)
    nx.draw_networkx_edges(subg, pos, edgelist=list(top_edge_set), width=3.2, alpha=0.98, edge_color="#E45756", arrows=False, ax=ax)
    if len(other_nodes) > 0:
        nx.draw_networkx_nodes(subg, pos, nodelist=other_nodes, node_size=20, node_color="#D9D9D9", alpha=0.22, ax=ax)
    if len(mid_nodes) > 0:
        nx.draw_networkx_nodes(subg, pos, nodelist=mid_nodes, node_size=120, node_color="#A0CBE8", alpha=0.98, ax=ax)
    if len(endpoints) > 0:
        nx.draw_networkx_nodes(subg, pos, nodelist=endpoints, node_size=165, node_color="#F58518", alpha=0.98, ax=ax)
    if len(targets) > 0:
        nx.draw_networkx_nodes(subg, pos, nodelist=targets, node_size=220, node_color="#54A24B", alpha=0.98, ax=ax)

    label_nodes = set(targets) | set(endpoints) | set(mid_nodes)
    nx.draw_networkx_labels(
        subg,
        pos,
        labels={int(n): labels[int(n)] for n in label_nodes},
        font_size=8,
        font_color="#111111",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        ax=ax,
    )
    ax.set_title(panel_title, fontsize=11)
    ax.axis("off")


def plot_target_links_two_panel(cold_drug_run, cold_cell_run, out_path: Path):
    id_to_symbol = load_gene_symbol_map()
    drug_targets = load_drug_targets()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2), dpi=220)
    plot_targets_to_top_edges_panel(
        axes[0],
        cold_drug_run,
        TARGET_LINK_CONFIGS[0]["drug_id"],
        TARGET_LINK_CONFIGS[0]["title"],
        id_to_symbol,
        drug_targets,
        TARGET_LINK_CONFIGS[0]["focus_edges"],
        TARGET_LINK_CONFIGS[0]["validated_paths"],
    )
    plot_targets_to_top_edges_panel(
        axes[1],
        cold_cell_run,
        TARGET_LINK_CONFIGS[1]["drug_id"],
        TARGET_LINK_CONFIGS[1]["title"],
        id_to_symbol,
        drug_targets,
        TARGET_LINK_CONFIGS[1]["focus_edges"],
        TARGET_LINK_CONFIGS[1]["validated_paths"],
    )
    legend_handles = [
        Line2D([0], [0], color="#54A24B", marker="o", linestyle="", markersize=9, label="Drug targets"),
        Line2D([0], [0], color="#4C78A8", linewidth=3, label="Paths to attention region"),
        Line2D([0], [0], color="#E45756", linewidth=3, label="Top attention edges"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    cold_drug_run = load_run(COLD_DRUG_NPZ)
    ranked = build_ranked_drug_table(cold_drug_run)
    available = set(cold_drug_run["attention"]["group_attention_edge_mean"].keys())
    ranked = ranked[(ranked["drug"].isin(available)) & (ranked["n"] >= int(MIN_N))].reset_index(drop=True)

    best20 = ranked.head(20).copy()
    worst20 = (
        ranked.tail(20)
        .sort_values(["median_pcc", "mean_pcc"], ascending=[True, True])
        .reset_index(drop=True)
        .copy()
    )

    plot_heatmap(
        cold_drug_run,
        best20,
        OUT_DIR / "attention_best_drugs_top20_heatmap.png",
        "Attention heatmap for the 20 best drugs in cold drug",
        top_k_edges=20,
    )
    plot_heatmap(
        cold_drug_run,
        worst20,
        OUT_DIR / "attention_worst_drugs_top20_heatmap.png",
        "Attention heatmap for the 20 worst drugs in cold drug",
        top_k_edges=20,
    )

    warm_run = load_run(WARM_NPZ)
    cold_cell_run = load_run(COLD_CELL_NPZ)
    cold_cell_ranked = build_ranked_drug_table(cold_cell_run)
    cold_cell_available = set(cold_cell_run["attention"]["group_attention_edge_mean"].keys())
    cold_cell_ranked = cold_cell_ranked[
        (cold_cell_ranked["drug"].isin(cold_cell_available)) & (cold_cell_ranked["n"] >= int(MIN_N))
    ].reset_index(drop=True)
    cold_cell_best20 = cold_cell_ranked.head(20).copy()
    plot_heatmap(
        cold_cell_run,
        cold_cell_best20,
        OUT_DIR / "attention_best_drugs_top20_heatmap_cold_cell.png",
        "Attention heatmap for the 20 best drugs in cold cell",
        top_k_edges=20,
    )
    shared_drug = "BRD-K87909389"
    plot_shared_drug_comparison(
        warm_run,
        cold_cell_run,
        shared_drug,
        OUT_DIR / "attention_warm_vs_cold_cell_shared_drug.png",
        top_k_edges=20,
    )
    plot_target_links_two_panel(
        cold_drug_run,
        cold_cell_run,
        OUT_DIR / "attention_target_links_two_panel.png",
    )


if __name__ == "__main__":
    main()
