import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def _palette():
    try:
        import colorbm

        return colorbm.pal("lancet").as_hex
    except Exception:
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]


def _box(ax, x, y, w, h, title, lines, facecolor):
    from matplotlib.patches import FancyBboxPatch

    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.9,
        edgecolor="0.2",
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + 0.02, y + h - 0.06, title, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.text(x + 0.02, y + h - 0.11, "\n".join(lines), ha="left", va="top", fontsize=7.2)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25", shrinkA=2, shrinkB=2),
        zorder=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    pal = _palette()

    c_tf = to_rgba(pal[0], 0.12)
    c_ppi = to_rgba(pal[1], 0.12)
    c_map = to_rgba(pal[2], 0.12)
    c_proc = to_rgba(pal[3], 0.12)
    c_out = to_rgba(pal[4], 0.12)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _box(
        ax,
        0.04,
        0.62,
        0.38,
        0.30,
        "OmniPath TF regulons (CSV)",
        [
            "source_genesymbol, target_genesymbol",
            "is_directed (True)",
            "is_stimulation / is_inhibition",
            "optional: consensus_* columns",
        ],
        c_tf,
    )
    _box(
        ax,
        0.04,
        0.22,
        0.38,
        0.30,
        "OmniPath interactions (CSV)",
        [
            "source_genesymbol, target_genesymbol",
            "is_directed (TF edges vs PPI edges)",
            "consensus_stimulation / consensus_inhibition",
            "optional: is_stimulation / is_inhibition",
        ],
        c_ppi,
    )

    _box(
        ax,
        0.48,
        0.42,
        0.22,
        0.36,
        "Identifier mapping",
        [
            "gene symbols  →  Entrez",
            "via LINCS gene info",
            "pr_gene_symbol → pr_gene_id",
            "978 landmark subset (optional)",
        ],
        c_map,
    )

    _box(
        ax,
        0.74,
        0.56,
        0.22,
        0.22,
        "Edge extraction",
        [
            "keep mapped endpoints",
            "optionally filter by gene set",
            "resolve sign (+1/-1)",
            "deduplicate (u,v)",
        ],
        c_proc,
    )

    _box(
        ax,
        0.74,
        0.23,
        0.22,
        0.22,
        "Graph outputs",
        [
            "node_list (ordered genes)",
            "edge_index (2×E)",
            "adjacency / weights",
            "directed edges",
        ],
        c_out,
    )

    _arrow(ax, 0.42, 0.76, 0.48, 0.62)
    _arrow(ax, 0.42, 0.36, 0.48, 0.54)
    _arrow(ax, 0.70, 0.58, 0.74, 0.66)
    _arrow(ax, 0.85, 0.56, 0.85, 0.45)

    ax.text(
        0.04,
        0.06,
        "Each table row encodes a candidate interaction with direction and (when available) sign annotations.\n"
        "Preprocessing aligns identifiers, filters to the modelling gene space, resolves sign, and exports a GNN-ready edge list.",
        fontsize=7,
        ha="left",
        va="bottom",
        color="0.25",
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

