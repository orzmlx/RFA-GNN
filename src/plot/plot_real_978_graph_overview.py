import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def _load_landmark_entrez_ids(path: str) -> list[str]:
    with open(path, "r") as f:
        landmark = json.load(f)
    if isinstance(landmark, list) and landmark and isinstance(landmark[0], dict) and "entrez_id" in landmark[0]:
        return [str(d["entrez_id"]).strip() for d in landmark if str(d.get("entrez_id", "")).strip()]
    return [str(x).strip() for x in landmark]


def _load_symbol_to_entrez(path: str) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    symbol = df["pr_gene_symbol"].astype(str).str.strip().str.upper()
    entrez = df["pr_gene_id"].astype(str).str.strip()
    m = pd.DataFrame({"symbol": symbol, "entrez": entrez}).dropna()
    m = m[(m["symbol"] != "") & (m["entrez"] != "")]
    return dict(zip(m["symbol"].tolist(), m["entrez"].tolist()))


def _invert_symbol_map(symbol_to_entrez: dict[str, str]) -> dict[str, str]:
    out = {}
    for symbol, entrez in symbol_to_entrez.items():
        e = str(entrez).strip()
        if e and e not in out:
            out[e] = str(symbol).strip().upper()
    return out


def _norm_symbol(s: str) -> str:
    return str(s).strip().upper()


def _norm_entrez(x: str) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _resolve_sign_from_row(row, stim_col: str, inhib_col: str) -> float | None:
    stim_i = bool(row.get(stim_col, False))
    inhib_i = bool(row.get(inhib_col, False))
    cs = bool(row.get("consensus_stimulation", False))
    ci = bool(row.get("consensus_inhibition", False))
    if (stim_i and inhib_i) or ((not stim_i) and (not inhib_i)):
        stim_i = cs
        inhib_i = ci
    if inhib_i and not stim_i:
        return -1.0
    if stim_i and not inhib_i:
        return 1.0
    return None


def _collect_edges(df: pd.DataFrame, symbol_to_entrez: dict[str, str], target_genes_set: set[str], stim_col: str, inhib_col: str, directed_only: bool):
    directed_edges = []
    undirected_edges = []

    df = df.copy()
    df["source_genesymbol"] = df["source_genesymbol"].astype(str).map(_norm_symbol)
    df["target_genesymbol"] = df["target_genesymbol"].astype(str).map(_norm_symbol)
    is_directed = df["is_directed"].fillna(False).astype(bool) if "is_directed" in df.columns else pd.Series(False, index=df.index)

    for _, row in df.iterrows():
        src_symbol = row["source_genesymbol"]
        tgt_symbol = row["target_genesymbol"]
        src_entrez = _norm_entrez(symbol_to_entrez.get(src_symbol, ""))
        tgt_entrez = _norm_entrez(symbol_to_entrez.get(tgt_symbol, ""))
        if not src_entrez or not tgt_entrez:
            continue
        if src_entrez not in target_genes_set or tgt_entrez not in target_genes_set:
            continue

        sign = _resolve_sign_from_row(row, stim_col, inhib_col)
        if sign is None:
            continue

        if bool(is_directed.loc[row.name]):
            directed_edges.append((src_entrez, tgt_entrez, sign))
        elif not directed_only:
            undirected_edges.append((src_entrez, tgt_entrez, abs(sign)))

    return directed_edges, undirected_edges


def _build_real_graph():
    landmark_path = os.path.join(REPO_ROOT, "data", "landmark_genes.json")
    full_gene_path = os.path.join(REPO_ROOT, "data", "GSE92742_Broad_LINCS_gene_info.txt")
    tf_path = os.path.join(REPO_ROOT, "data", "omnipath", "omnipath_tf_regulons.csv")
    ppi_path = os.path.join(REPO_ROOT, "data", "omnipath", "omnipath_interactions.csv")

    target_genes = _load_landmark_entrez_ids(landmark_path)
    symbol_to_entrez = _load_symbol_to_entrez(full_gene_path)
    entrez_to_symbol = _invert_symbol_map(symbol_to_entrez)

    node_list = [_norm_entrez(x) for x in target_genes if _norm_entrez(x)]
    node_list = list(dict.fromkeys(node_list))
    gene2idx = {g: i for i, g in enumerate(node_list)}
    target_genes_set = set(node_list)
    n = len(node_list)
    adj_matrix = np.zeros((n, n), dtype=np.float32)

    df_tf = pd.read_csv(tf_path)
    tf_directed, _ = _collect_edges(
        df_tf,
        symbol_to_entrez=symbol_to_entrez,
        target_genes_set=target_genes_set,
        stim_col="is_stimulation",
        inhib_col="is_inhibition",
        directed_only=True,
    )

    df_ppi = pd.read_csv(ppi_path)
    ppi_directed, ppi_undirected = _collect_edges(
        df_ppi,
        symbol_to_entrez=symbol_to_entrez,
        target_genes_set=target_genes_set,
        stim_col="consensus_stimulation",
        inhib_col="consensus_inhibition",
        directed_only=False,
    )

    edge_src = []
    edge_dst = []

    for u, v, w in tf_directed + ppi_directed:
        if u in gene2idx and v in gene2idx:
            i = gene2idx[u]
            j = gene2idx[v]
            if abs(w) > abs(adj_matrix[j, i]):
                adj_matrix[j, i] = w
            edge_src.append(i)
            edge_dst.append(j)

    for u, v, w in ppi_undirected:
        if u in gene2idx and v in gene2idx:
            i = gene2idx[u]
            j = gene2idx[v]
            adj_matrix[i, j] = max(adj_matrix[i, j], w)
            adj_matrix[j, i] = max(adj_matrix[j, i], w)
            edge_src.extend([i, j])
            edge_dst.extend([j, i])

    np.fill_diagonal(adj_matrix, 1.0)
    edge_index = np.array([edge_src, edge_dst], dtype=np.int32)
    return adj_matrix, node_list, edge_index, entrez_to_symbol


def _ring_positions(indices: list[int], radius: float, start_angle: float) -> dict[int, np.ndarray]:
    if not indices:
        return {}
    angles = np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False) + start_angle
    return {
        idx: np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=float)
        for idx, theta in zip(indices, angles)
    }


def _build_layout(total_degree: np.ndarray, out_degree: np.ndarray, in_degree: np.ndarray):
    n = len(total_degree)
    rank = np.argsort(-total_degree)
    n_core = min(max(36, n // 30), 48)
    n_middle = min(max(180, n // 4), 240)
    core = rank[:n_core].tolist()
    middle = rank[n_core : min(n_core + n_middle, n)].tolist()
    outer = rank[min(n_core + n_middle, n) :].tolist()

    def _sort_key(idx: int):
        score = out_degree[idx] - in_degree[idx]
        return (-score, -total_degree[idx], idx)

    core = sorted(core, key=_sort_key)
    middle = sorted(middle, key=_sort_key)
    outer = sorted(outer, key=_sort_key)

    pos = {}
    pos.update(_ring_positions(core, 0.95, np.pi / 2))
    pos.update(_ring_positions(middle, 2.05, np.pi / 2 + 0.08))
    pos.update(_ring_positions(outer, 3.25, np.pi / 2 + 0.04))
    xy = np.vstack([pos[i] for i in range(n)])
    return xy, core, middle, outer


def _node_role_colors(out_degree: np.ndarray, in_degree: np.ndarray):
    total = out_degree + in_degree
    balance = (out_degree - in_degree) / np.maximum(total, 1.0)
    colors = np.empty(total.shape[0], dtype=object)
    colors[balance > 0.18] = "#177e89"
    colors[balance < -0.18] = "#d95f02"
    mask = (balance >= -0.18) & (balance <= 0.18)
    colors[mask] = "#4c78a8"
    return colors


def _draw_shell_guides(ax):
    shell_specs = [
        (0.95, "Core hubs"),
        (2.05, "Middle layer"),
        (3.25, "Peripheral genes"),
    ]
    for radius, label in shell_specs:
        c = Circle((0, 0), radius, facecolor="none", edgecolor=(0.4, 0.4, 0.4, 0.18), linestyle="--", linewidth=0.8, zorder=0)
        ax.add_patch(c)
        ax.text(0, radius + 0.12, label, ha="center", va="bottom", fontsize=8, color="0.35")


def _draw_edges(ax, xy: np.ndarray, src: np.ndarray, dst: np.ndarray, sign: np.ndarray, core_set: set[int]):
    segments = np.stack([xy[src], xy[dst]], axis=1)
    touch_core = np.array([(int(s) in core_set) or (int(t) in core_set) for s, t in zip(src, dst)], dtype=bool)
    pos_mask = sign > 0
    neg_mask = sign < 0

    collections = [
        (pos_mask & (~touch_core), "#2a9d8f", 0.18, 0.020),
        (neg_mask & (~touch_core), "#c1121f", 0.18, 0.020),
        (pos_mask & touch_core, "#2a9d8f", 0.42, 0.090),
        (neg_mask & touch_core, "#c1121f", 0.42, 0.090),
    ]

    for mask, color, lw, alpha in collections:
        if not np.any(mask):
            continue
        lc = LineCollection(segments[mask], colors=[(*plt.matplotlib.colors.to_rgb(color), alpha)], linewidths=lw, zorder=1)
        ax.add_collection(lc)


def _draw_nodes(ax, xy: np.ndarray, total_degree: np.ndarray, colors: np.ndarray, core: list[int]):
    deg_min = float(total_degree.min()) if total_degree.size else 0.0
    deg_max = float(total_degree.max()) if total_degree.size else 1.0
    denom = deg_max - deg_min if deg_max > deg_min else 1.0
    deg_norm = (total_degree - deg_min) / denom
    sizes = 10.0 + 80.0 * (deg_norm ** 0.7)
    sizes[np.array(core, dtype=int)] *= 1.35
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors.tolist(), linewidths=0.25, edgecolors=(1, 1, 1, 0.55), alpha=0.96, zorder=2)


def _annotate_hubs(ax, xy: np.ndarray, node_list: list[str], entrez_to_symbol: dict[str, str], total_degree: np.ndarray, n_labels: int = 18):
    rank = np.argsort(-total_degree)[:n_labels]
    for idx in rank:
        x, y = xy[idx]
        vec = np.array([x, y], dtype=float)
        norm = np.linalg.norm(vec)
        direction = vec / norm if norm > 1e-8 else np.array([1.0, 0.0])
        anchor = vec + 0.24 * direction
        label = entrez_to_symbol.get(str(node_list[idx]), str(node_list[idx]))
        ha = "left" if direction[0] >= 0 else "right"
        text_shift = np.array([0.10 if ha == "left" else -0.10, 0.0])
        label_pos = anchor + text_shift
        ax.plot([x, anchor[0]], [y, anchor[1]], color="0.35", linewidth=0.45, alpha=0.7, zorder=3)
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha=ha,
            va="center",
            fontsize=7.2,
            color="0.12",
            bbox=dict(boxstyle="round,pad=0.14", facecolor=(1, 1, 1, 0.86), edgecolor=(0.7, 0.7, 0.7, 0.75), linewidth=0.5),
            zorder=4,
        )


def _draw_info_panel(fig, n_nodes: int, src: np.ndarray, sign: np.ndarray):
    ax = fig.add_axes([0.77, 0.12, 0.21, 0.76])
    ax.set_axis_off()

    n_edges = int(len(src))
    n_pos = int(np.sum(sign > 0))
    n_neg = int(np.sum(sign < 0))

    ax.text(0.0, 0.98, "Real 978-gene graph", ha="left", va="top", fontsize=11, fontweight="bold", color="0.12")
    ax.text(0.0, 0.92, "Signed directed graph used by the model", ha="left", va="top", fontsize=8.2, color="0.35")

    stats = [
        f"Nodes: {n_nodes}",
        f"Directed edges: {n_edges}",
        f"Positive edges: {n_pos}",
        f"Negative edges: {n_neg}",
    ]
    y = 0.82
    for line in stats:
        ax.text(0.0, y, line, ha="left", va="top", fontsize=8.4, color="0.15")
        y -= 0.07

    ax.text(0.0, 0.50, "Adjacency semantics", ha="left", va="top", fontsize=9, fontweight="bold", color="0.12")
    semantics = [
        "Column = source gene",
        "Row = target gene",
        "Positive = stimulation",
        "Negative = inhibition",
    ]
    y = 0.43
    for line in semantics:
        ax.text(0.02, y, u"\u2022 " + line, ha="left", va="top", fontsize=8.1, color="0.15")
        y -= 0.065

    role_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#177e89", markersize=7, label="Regulator-like node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markersize=7, label="Balanced node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d95f02", markersize=7, label="Receiver-like node"),
        Line2D([0, 1], [0, 0], color="#2a9d8f", linewidth=1.6, label="Positive edge"),
        Line2D([0, 1], [0, 0], color="#c1121f", linewidth=1.6, label="Negative edge"),
    ]
    leg = ax.legend(handles=role_handles, loc="lower left", bbox_to_anchor=(0.0, 0.02), frameon=False, fontsize=7.8, handlelength=1.8, labelspacing=0.8)
    for txt in leg.get_texts():
        txt.set_color("0.15")


def plot_graph(out_path: str, dpi: int):
    _apply_style()
    adj_matrix, node_list, edge_index, entrez_to_symbol = _build_real_graph()

    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    sign = adj_matrix[dst, src]
    keep = sign != 0
    src = src[keep]
    dst = dst[keep]
    sign = sign[keep]

    n = len(node_list)
    out_degree = np.bincount(src, minlength=n).astype(float)
    in_degree = np.bincount(dst, minlength=n).astype(float)
    total_degree = out_degree + in_degree

    xy, core, _, _ = _build_layout(total_degree, out_degree, in_degree)
    colors = _node_role_colors(out_degree, in_degree)

    fig = plt.figure(figsize=(11.2, 8.2))
    ax = fig.add_axes([0.04, 0.05, 0.70, 0.90])
    ax.set_axis_off()
    ax.set_aspect("equal")

    _draw_shell_guides(ax)
    _draw_edges(ax, xy, src, dst, sign, set(core))
    _draw_nodes(ax, xy, total_degree, colors, core)
    _annotate_hubs(ax, xy, node_list, entrez_to_symbol, total_degree, n_labels=18)

    lim = 4.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    _draw_info_panel(fig, n, src, sign)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    plot_graph(args.out, args.dpi)


if __name__ == "__main__":
    main()
