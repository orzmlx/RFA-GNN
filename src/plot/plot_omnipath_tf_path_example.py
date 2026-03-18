import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _resolve_sign(row) -> int | None:
    cs = bool(row.get("consensus_stimulation", False))
    ci = bool(row.get("consensus_inhibition", False))
    if cs != ci:
        return 1 if cs else -1
    st = bool(row.get("is_stimulation", False))
    inh = bool(row.get("is_inhibition", False))
    if st != inh:
        return 1 if st else -1
    return None


def _find_signed_path(df: pd.DataFrame, k_edges: int = 3) -> list[str] | None:
    df = df.copy()
    df["sign"] = df.apply(_resolve_sign, axis=1)
    df = df[df["is_directed"].fillna(False).astype(bool) & df["sign"].notna()].copy()
    df["source_genesymbol"] = df["source_genesymbol"].astype(str)
    df["target_genesymbol"] = df["target_genesymbol"].astype(str)

    succ = {}
    for s, t, sg in df[["source_genesymbol", "target_genesymbol", "sign"]].itertuples(index=False):
        succ.setdefault(s, []).append((t, int(sg)))

    tfs = sorted(set(df["source_genesymbol"].tolist()))
    for a in tfs:
        for b, _ in succ.get(a, []):
            if b not in succ:
                continue
            for c, _ in succ.get(b, []):
                if c not in succ:
                    continue
                for d, _ in succ.get(c, []):
                    path = [a, b, c, d]
                    if len(set(path)) != 4:
                        continue
                    return path
    return None


def _extract_rows_for_path(df: pd.DataFrame, path: list[str]) -> pd.DataFrame:
    edges = list(zip(path[:-1], path[1:]))
    sub = df[df["is_directed"].fillna(False).astype(bool)].copy()
    sub["source_genesymbol"] = sub["source_genesymbol"].astype(str)
    sub["target_genesymbol"] = sub["target_genesymbol"].astype(str)
    sub["sign"] = sub.apply(_resolve_sign, axis=1)
    out = []
    for s, t in edges:
        hit = sub[(sub["source_genesymbol"] == s) & (sub["target_genesymbol"] == t)]
        if hit.empty:
            raise RuntimeError(f"Missing edge in TF table: {s} -> {t}")
        hit = hit.sort_values(["n_sources", "n_references"], ascending=False, na_position="last")
        out.append(hit.iloc[0])
    return pd.DataFrame(out)


def _draw_table(ax, rows: pd.DataFrame, pal: list[str]):
    ax.set_axis_off()
    header_bg = to_rgba(pal[0], 0.18)
    section_bg = to_rgba(pal[2], 0.10)
    alt_bg = [to_rgba("0.98", 1.0), to_rgba("0.94", 1.0)]

    cols = ["source_genesymbol", "target_genesymbol", "sign", "n_sources", "n_references"]
    disp = rows[cols].copy()
    disp["sign"] = disp["sign"].map({1: "+1", -1: "-1"}).fillna("NA")
    disp["n_sources"] = disp["n_sources"].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    disp["n_references"] = disp["n_references"].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    disp.columns = ["source", "target", "sign", "n_sources", "n_refs"]

    cell_text = disp.values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.20, 0.22, 0.10, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.05, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(weight="bold", color="0.15")
        else:
            cell.set_facecolor(alt_bg[(r - 1) % 2])
            if c == 2:
                val = disp.iloc[r - 1, 2]
                if val == "+1":
                    cell.set_text_props(color=pal[2], weight="bold")
                elif val == "-1":
                    cell.set_text_props(color=pal[3], weight="bold")
                cell._loc = "center"

    ax.text(
        0.02,
        0.98,
        "Example rows from omnipath_tf_regulons.csv",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="0.15",
    )
    ax.text(
        0.02,
        0.90,
        "Real TF→target regulations used to form the path on the right.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.35",
    )


def _draw_path(ax, path: list[str], rows: pd.DataFrame, pal: list[str]):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    node_bg = [to_rgba(pal[0], 0.12), to_rgba(pal[1], 0.12), to_rgba(pal[4], 0.12), to_rgba(pal[5], 0.12)]

    xs = np.linspace(0.12, 0.88, len(path))
    y = 0.55

    from matplotlib.patches import FancyBboxPatch

    def node_box(x, text, fc):
        w, h = 0.18, 0.13
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=0.9,
            edgecolor="0.2",
            facecolor=fc,
            zorder=2,
        )
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold", color="0.15", zorder=3)

    for i, g in enumerate(path):
        node_box(xs[i], g, node_bg[i % len(node_bg)])

    for i in range(len(path) - 1):
        s, t = path[i], path[i + 1]
        row = rows.iloc[i]
        sign = int(row["sign"]) if pd.notna(row["sign"]) else 1
        edge_color = pal[2] if sign > 0 else pal[3]
        x0, x1 = xs[i] + 0.11, xs[i + 1] - 0.11
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="->", lw=2.0, color=edge_color, shrinkA=2, shrinkB=2),
            zorder=1,
        )
        ax.text((x0 + x1) / 2, y + 0.10, "+1" if sign > 0 else "-1", ha="center", va="bottom", fontsize=8, color=edge_color, fontweight="bold")

    ax.text(
        0.02,
        0.98,
        "Regulatory path (real edges)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="0.15",
    )
    ax.text(
        0.02,
        0.90,
        "A directed TF cascade where each arrow corresponds to a row in the TF regulon table.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.35",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", default="data/omnipath/omnipath_tf_regulons.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    pal = _palette()

    df = pd.read_csv(args.tf)

    preferred = ["HOXB2", "OTX2", "PAX6", "KRT12"]
    try:
        rows = _extract_rows_for_path(df, preferred)
        path = preferred
    except Exception:
        path = _find_signed_path(df, k_edges=3)
        if not path:
            raise RuntimeError("Could not find a signed TF regulatory path in the TF table.")
        rows = _extract_rows_for_path(df, path)

    fig = plt.figure(figsize=(7.8, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    _draw_table(ax0, rows, pal)
    _draw_path(ax1, path, rows, pal)

    fig.suptitle("Example OmniPath TF regulon entries and a derived regulatory path", y=1.02)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
