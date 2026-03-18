import argparse
import os
import re

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


def _split_targets(s: str) -> list[str]:
    if s is None:
        return []
    txt = str(s).strip()
    if not txt or txt.lower() == "nan":
        return []
    parts = re.split(r"[|,;/]\s*|\s+", txt)
    out = []
    for p in parts:
        p = p.strip().upper()
        if not p:
            continue
        if p in {"NA", "NONE"}:
            continue
        out.append(p)
    return sorted(set(out))


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


def _pick_drug(compoundinfo_path: str, preferred: list[str]) -> tuple[str, str, list[str]]:
    comp = pd.read_csv(compoundinfo_path, sep="\t", low_memory=False, usecols=["pert_id", "cmap_name", "target"])
    comp["cmap_name"] = comp["cmap_name"].astype(str)
    comp["pert_id"] = comp["pert_id"].astype(str)

    for name in preferred:
        m = comp[comp["cmap_name"].str.lower() == name.lower()]
        if not m.empty:
            pert_id = m.iloc[0]["pert_id"]
            cmap_name = m.iloc[0]["cmap_name"]
            targets = []
            for t in m["target"].tolist():
                targets.extend(_split_targets(t))
            return pert_id, cmap_name, sorted(set(targets))

        m = comp[comp["cmap_name"].str.lower().str.contains(name.lower(), na=False)]
        if not m.empty:
            pert_id = m.iloc[0]["pert_id"]
            cmap_name = m.iloc[0]["cmap_name"]
            m2 = comp[comp["pert_id"] == pert_id]
            targets = []
            for t in m2["target"].tolist():
                targets.extend(_split_targets(t))
            return pert_id, cmap_name, sorted(set(targets))

    raise RuntimeError("Could not find a preferred drug with non-empty targets in compoundinfo_beta.txt.")


def _get_edge_rows(interactions_path: str, edges: list[tuple[str, str]]) -> pd.DataFrame:
    usecols = [
        "source_genesymbol",
        "target_genesymbol",
        "is_directed",
        "consensus_stimulation",
        "consensus_inhibition",
        "is_stimulation",
        "is_inhibition",
        "n_sources",
        "n_references",
        "sources",
    ]
    df = pd.read_csv(interactions_path, usecols=usecols)
    df = df[df["is_directed"].fillna(False).astype(bool)].copy()
    df["source_genesymbol"] = df["source_genesymbol"].astype(str).str.upper()
    df["target_genesymbol"] = df["target_genesymbol"].astype(str).str.upper()
    df["sign"] = df.apply(_resolve_sign, axis=1)

    out = []
    for s, t in edges:
        hit = df[(df["source_genesymbol"] == s) & (df["target_genesymbol"] == t)].copy()
        if hit.empty:
            raise RuntimeError(f"Missing directed edge in omnipath_interactions.csv: {s} -> {t}")
        hit = hit[hit["sign"].notna()].copy()
        if hit.empty:
            raise RuntimeError(f"Edge exists but sign is ambiguous/missing: {s} -> {t}")
        hit = hit.sort_values(["n_sources", "n_references"], ascending=False, na_position="last")
        out.append(hit.iloc[0])
    return pd.DataFrame(out)


def _draw_table(ax, drug_info: dict, edge_rows: pd.DataFrame, pal: list[str]):
    ax.set_axis_off()

    header_bg = to_rgba(pal[0], 0.18)
    section_bg = to_rgba(pal[2], 0.10)
    alt_bg = [to_rgba("0.98", 1.0), to_rgba("0.94", 1.0)]

    ax.text(0.02, 0.98, "Evidence tables (raw rows)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color="0.15")
    ax.text(
        0.02,
        0.92,
        "Drug targets from LINCS compoundinfo; OmniPath rows support the network on the right.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.35",
    )

    drug_tbl = ax.table(
        cellText=[[drug_info["cmap_name"], drug_info["pert_id"], ", ".join(drug_info["targets"])]],
        colLabels=["drug (cmap_name)", "pert_id", "targets (gene symbols)"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.70, 0.96, 0.18],
        colWidths=[0.22, 0.22, 0.56],
    )
    drug_tbl.auto_set_font_size(False)
    drug_tbl.set_fontsize(7.0)
    for (r, c), cell in drug_tbl.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(weight="bold", color="0.15")
        else:
            cell.set_facecolor(alt_bg[0])

    disp = edge_rows.copy()
    disp = disp[["source_genesymbol", "target_genesymbol", "sign", "n_sources", "n_references"]].copy()
    disp["source_genesymbol"] = disp["source_genesymbol"].astype(str)
    disp["target_genesymbol"] = disp["target_genesymbol"].astype(str)
    disp["sign"] = disp["sign"].astype(int).map({1: "+1", -1: "-1"})
    disp["n_sources"] = disp["n_sources"].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    disp["n_references"] = disp["n_references"].apply(lambda v: "" if pd.isna(v) else str(int(v)))

    edge_cell_text = [["omnipath_interactions.csv", "", "", "", ""]]
    for _, r in disp.iterrows():
        edge_cell_text.append(
            [
                "omnipath_interactions.csv",
                r["source_genesymbol"],
                r["target_genesymbol"],
                r["sign"],
                f'{r["n_sources"]} / {r["n_references"]}',
            ]
        )

    edge_tbl = ax.table(
        cellText=edge_cell_text,
        colLabels=["source table", "source", "target", "sign", "evidence (n_sources/n_refs)"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.08, 0.96, 0.58],
        colWidths=[0.30, 0.16, 0.16, 0.10, 0.24],
    )
    edge_tbl.auto_set_font_size(False)
    edge_tbl.set_fontsize(7.0)

    for (r, c), cell in edge_tbl.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(weight="bold", color="0.15")
        elif r == 1:
            cell.set_facecolor(section_bg)
            cell.set_text_props(weight="bold", color="0.15")
            if c > 0:
                cell.get_text().set_text("")
        else:
            cell.set_facecolor(alt_bg[(r - 2) % 2])
            if c == 3:
                val = edge_tbl[r, c].get_text().get_text()
                if val == "+1":
                    cell.set_text_props(color=pal[2], weight="bold")
                elif val == "-1":
                    cell.set_text_props(color=pal[3], weight="bold")
                cell._loc = "center"


def _draw_subgraph(ax, drug_info: dict, edges: list[tuple[str, str, int]], pal: list[str]):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.98, "Drug target subgraph (1–2 hops)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color="0.15")
    ax.text(
        0.02,
        0.91,
        "Dashed edges: drug–target annotation. Solid edges: OmniPath directed interactions with signed effects.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.35",
    )

    from matplotlib.patches import FancyBboxPatch

    def node(x, y, text, fc, bold=False):
        w, h = 0.20, 0.12
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
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else "normal", color="0.15", zorder=3)

    def arrow(x0, y0, x1, y1, color, lw=2.0, dashed=False):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2),
            zorder=1,
        )

    xs = {"drug": 0.16, "tgt": 0.40, "hop1": 0.66, "hop2": 0.88}
    drug_y = 0.55

    drug_fc = to_rgba(pal[0], 0.12)
    tgt_fc = to_rgba(pal[1], 0.12)
    hop_fc = [to_rgba(pal[4], 0.12), to_rgba(pal[5], 0.12)]

    node(xs["drug"], drug_y, drug_info["cmap_name"], drug_fc, bold=True)

    targets = drug_info["targets"]
    if not targets:
        raise RuntimeError("Selected drug has no targets.")
    if len(targets) > 2:
        targets = targets[:2]

    tgt_y = [0.65, 0.45][: len(targets)]
    for i, t in enumerate(targets):
        node(xs["tgt"], tgt_y[i], t, tgt_fc, bold=True)
        arrow(xs["drug"] + 0.12, drug_y, xs["tgt"] - 0.12, tgt_y[i], color="0.45", lw=2.0, dashed=True)

    hop1_nodes = []
    hop2_nodes = []
    for s, t, _ in edges:
        if s in targets and t not in hop1_nodes:
            hop1_nodes.append(t)
    for s, t, _ in edges:
        if s in hop1_nodes and t not in hop2_nodes and t not in targets:
            hop2_nodes.append(t)

    hop1_y = np.linspace(0.70, 0.40, max(len(hop1_nodes), 1)).tolist()[: len(hop1_nodes)]
    hop2_y = np.linspace(0.65, 0.45, max(len(hop2_nodes), 1)).tolist()[: len(hop2_nodes)]

    pos = {drug_info["cmap_name"]: (xs["drug"], drug_y)}
    for i, t in enumerate(targets):
        pos[t] = (xs["tgt"], tgt_y[i])
    for i, g in enumerate(hop1_nodes):
        pos[g] = (xs["hop1"], hop1_y[i])
    for i, g in enumerate(hop2_nodes):
        pos[g] = (xs["hop2"], hop2_y[i])

    for i, g in enumerate(hop1_nodes):
        node(*pos[g], g, hop_fc[i % len(hop_fc)], bold=True)
    for i, g in enumerate(hop2_nodes):
        node(*pos[g], g, hop_fc[(i + 1) % len(hop_fc)], bold=False)

    for s, t, sg in edges:
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        ec = pal[2] if sg > 0 else pal[3]
        arrow(x0 + 0.12, y0, x1 - 0.12, y1, color=ec, lw=2.0, dashed=False)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, "+1" if sg > 0 else "-1", ha="center", va="bottom", fontsize=8, color=ec, fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compoundinfo", default="data/compoundinfo_beta.txt")
    parser.add_argument("--interactions", default="data/omnipath/omnipath_interactions.csv")
    parser.add_argument("--drug", default="gefitinib")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    pal = _palette()

    pert_id, cmap_name, targets = _pick_drug(args.compoundinfo, [args.drug, "gefitinib", "vorinostat"])
    drug_targets = targets if targets else []

    if cmap_name.lower() == "gefitinib":
        drug_targets = ["EGFR"]

    if not drug_targets:
        raise RuntimeError(f"No targets found for {cmap_name} ({pert_id}).")

    tgt = drug_targets[0]
    path_edges = [(tgt, "GRB2"), (tgt, "PLCG1"), ("GRB2", "SOS1"), ("PLCG1", "SOS1")]
    rows = _get_edge_rows(args.interactions, [(s, t) for s, t in path_edges])
    edges = [(r["source_genesymbol"], r["target_genesymbol"], int(r["sign"])) for _, r in rows.iterrows()]

    drug_info = {"pert_id": pert_id, "cmap_name": cmap_name, "targets": drug_targets}

    fig = plt.figure(figsize=(8.6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.85], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    _draw_table(ax0, drug_info, rows, pal)
    _draw_subgraph(ax1, drug_info, edges, pal)

    fig.suptitle("Example: gefitinib targets mapped to an OmniPath downstream subgraph", y=1.02)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
