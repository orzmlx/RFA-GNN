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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    pal = _palette()

    header_bg = to_rgba(pal[0], 0.18)
    section_bg = to_rgba(pal[2], 0.14)
    alt_bg = [to_rgba("0.98", 1.0), to_rgba("0.94", 1.0)]

    columns = ["Table", "Key columns", "Meaning", "Used in this thesis"]
    rows = [
        ["OmniPath TF regulons", "", "", ""],
        [
            "omnipath_tf_regulons.csv",
            "source_genesymbol, target_genesymbol",
            "Directed TF→target regulatory edges",
            "Included; mapped to Entrez; kept if endpoints in gene set",
        ],
        [
            "omnipath_tf_regulons.csv",
            "is_stimulation / is_inhibition",
            "Sign of regulation (+1/−1) when available",
            "Used to set signed edge weights",
        ],
        [
            "omnipath_tf_regulons.csv",
            "is_directed",
            "Direction indicator (typically True for TF)",
            "Graph stored as directed edges",
        ],
        ["OmniPath interactions", "", "", ""],
        [
            "omnipath_interactions.csv",
            "source_genesymbol, target_genesymbol",
            "General interaction edges (PPI and others)",
            "Merged with TF regulons to improve coverage",
        ],
        [
            "omnipath_interactions.csv",
            "consensus_stimulation / consensus_inhibition",
            "Consensus sign evidence aggregated across sources",
            "Primary signal for signed PPI edges",
        ],
        [
            "omnipath_interactions.csv",
            "is_directed",
            "Whether an interaction is directed in OmniPath",
            "Optionally filter to directed-only edges",
        ],
        ["Shared preprocessing", "", "", ""],
        [
            "LINCS gene info",
            "pr_gene_symbol → pr_gene_id",
            "Map gene symbols to Entrez identifiers",
            "Align graph nodes to expression gene IDs",
        ],
        [
            "Filtering",
            "endpoints in selected gene set",
            "Keep edges only inside modelling space (978/12,328)",
            "Ensures consistent node ordering",
        ],
        [
            "Deduplication",
            "(u,v) unique",
            "Merge duplicate sources into one ordered pair",
            "Edge list exported as edge_index (2×E)",
        ],
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.set_axis_off()

    tbl = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.18, 0.30, 0.27, 0.25],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.1)
    tbl.scale(1.0, 1.45)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(weight="bold", color="0.15")
        else:
            cell.set_facecolor(alt_bg[(r - 1) % 2])

    section_rows = [1, 5, 9]
    for sr in section_rows:
        for c in range(len(columns)):
            cell = tbl[sr, c]
            cell.set_facecolor(section_bg)
            cell.set_text_props(weight="bold", color="0.15")
            if c > 0:
                cell.get_text().set_text("")

    fig.suptitle("OmniPath tables used for graph construction", y=0.98)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

