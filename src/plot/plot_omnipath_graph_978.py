import argparse
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data_loader import build_combined_gnn, combine_full_grapg


def _load_landmark_entrez_ids(landmark_path: str) -> list[str]:
    with open(landmark_path, "r") as f:
        landmark = json.load(f)
    if isinstance(landmark, list) and landmark and isinstance(landmark[0], dict) and "entrez_id" in landmark[0]:
        return [str(d["entrez_id"]).strip() for d in landmark if str(d.get("entrez_id", "")).strip()]
    return [str(x).strip() for x in landmark]


def _load_symbol_to_entrez(full_gene_path: str) -> dict[str, str]:
    df = pd.read_csv(full_gene_path, sep="\t", dtype=str)
    if "pr_gene_symbol" not in df.columns or "pr_gene_id" not in df.columns:
        raise ValueError("full_gene_path missing required columns: pr_gene_symbol, pr_gene_id")
    symbol = df["pr_gene_symbol"].astype(str).str.strip().str.upper()
    entrez = df["pr_gene_id"].astype(str).str.strip()
    m = pd.DataFrame({"symbol": symbol, "entrez": entrez}).dropna()
    m = m[m["symbol"] != ""]
    m = m[m["entrez"] != ""]
    return dict(zip(m["symbol"].tolist(), m["entrez"].tolist()))


def _get_colormap(name: str):
    try:
        import colorbm

        return colorbm.seq(name).as_cmap()
    except Exception:
        return plt.get_cmap("viridis")


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["combined", "full"], default="full")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layout-iter", type=int, default=300)
    parser.add_argument("--cmap", default="batlow")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    cmap = _get_colormap(args.cmap)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    landmark_path = os.path.join(repo_root, "data", "landmark_genes.json")
    full_gene_path = os.path.join(repo_root, "data", "GSE92742_Broad_LINCS_gene_info.txt")

    target_genes = _load_landmark_entrez_ids(landmark_path)
    symbol_to_entrez = _load_symbol_to_entrez(full_gene_path)

    tf_path = os.path.join(repo_root, "data", "omnipath", "omnipath_tf_regulons.csv")
    ppi_path = os.path.join(repo_root, "data", "omnipath", "omnipath_interactions.csv")
    if args.mode == "full":
        _, node_list, _, edge_index = combine_full_grapg(
            tf_path=tf_path,
            ppi_path=ppi_path,
            target_genes=target_genes,
            directed=True,
            symbol_to_entrez=symbol_to_entrez,
        )
    else:
        _, node_list, _, edge_index = build_combined_gnn(
            tf_path=tf_path,
            ppi_path=ppi_path,
            target_genes=target_genes,
            directed=True,
            omnipath_consensus_only=False,
            omnipath_is_directed_only=False,
            symbol_to_entrez=symbol_to_entrez,
        )

    n = len(node_list)
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    deg = np.array([d for _, d in g.degree()], dtype=float)
    deg_min = float(deg.min()) if deg.size else 0.0
    deg_max = float(deg.max()) if deg.size else 1.0
    denom = deg_max - deg_min if deg_max > deg_min else 1.0
    deg_norm = (deg - deg_min) / denom

    k = 0.8 / np.sqrt(max(n, 1))
    pos = nx.spring_layout(g.to_undirected(), seed=args.seed, k=k, iterations=args.layout_iter)
    xy = np.array([pos[i] for i in range(n)])

    sizes = 8.0 + 70.0 * (deg_norm**0.8)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_axis_off()

    if edges:
        segs = np.stack([xy[np.array(edge_index[0])], xy[np.array(edge_index[1])]], axis=1)
        from matplotlib.collections import LineCollection

        lc = LineCollection(segs, colors=[(0.15, 0.15, 0.15, 0.25)], linewidths=0.6, zorder=1)
        ax.add_collection(lc)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=sizes,
        c=deg,
        cmap=cmap,
        linewidths=0.0,
        alpha=0.95,
        zorder=2,
    )

    from matplotlib.colors import Normalize

    sc.set_norm(Normalize(vmin=deg_min, vmax=deg_max))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Node degree")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
