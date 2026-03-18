import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def _load_landmark_entrez_ids(landmark_path: str) -> list[str]:
    with open(landmark_path, "r") as f:
        landmark = json.load(f)
    if isinstance(landmark, list) and landmark and isinstance(landmark[0], dict) and "entrez_id" in landmark[0]:
        return [str(d["entrez_id"]).strip() for d in landmark if str(d.get("entrez_id", "")).strip()]
    return [str(x).strip() for x in landmark]


def _read_gctx_subset(gctx_path: str, sample_ids: list[str], genes: list[str]) -> np.ndarray:
    with h5py.File(gctx_path, "r") as f:
        matrix = f["0/DATA/0/matrix"]
        row_ids = f["0/META/ROW/id"][:].astype(str)
        col_ids = f["0/META/COL/id"][:].astype(str)

        row_index = {v: i for i, v in enumerate(row_ids)}
        col_index = {v: i for i, v in enumerate(col_ids)}

        valid_pairs = [(sid, col_index[sid]) for sid in sample_ids if sid in col_index]
        if not valid_pairs:
            return np.zeros((0, len(genes)), dtype=np.float32)

        valid_pairs = sorted(valid_pairs, key=lambda x: x[1])
        seen = set()
        valid_samples = []
        sample_idx = []
        for sid, idx in valid_pairs:
            if idx in seen:
                continue
            seen.add(idx)
            valid_samples.append(sid)
            sample_idx.append(idx)

        gene_idx = [row_index[g] for g in genes if g in row_index]
        gene_present = [g for g in genes if g in row_index]
        gene_pos = {g: i for i, g in enumerate(genes)}

        if matrix.shape[0] == len(row_ids) and matrix.shape[1] == len(col_ids):
            data = matrix[:, sample_idx]
            data = np.asarray(data, dtype=np.float32)
            data = data[gene_idx, :].T
        elif matrix.shape[0] == len(col_ids) and matrix.shape[1] == len(row_ids):
            data = matrix[sample_idx, :]
            data = np.asarray(data, dtype=np.float32)
            data = data[:, gene_idx]
        else:
            raise ValueError("Unexpected GCTX matrix shape.")

    full = np.zeros((len(valid_samples), len(genes)), dtype=np.float32)
    for j, g in enumerate(gene_present):
        full[:, gene_pos[g]] = data[:, j]
    return full


def _categorize_top(series: pd.Series, top_n: int) -> pd.Series:
    s = series.astype(str).fillna("NA")
    top = s.value_counts().head(top_n).index.tolist()
    return s.where(s.isin(top), other="Other")


def _palette(n: int):
    try:
        import colorbm

        pal = colorbm.pal("lancet").as_hex
    except Exception:
        pal = None
    if not pal or len(pal) < n:
        pal = plt.get_cmap("tab10").colors
    return list(pal)[:n]

def _shorten(label: str, max_len: int = 14) -> str:
    s = str(label)
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siginfo", default="data/siginfo_beta.txt")
    parser.add_argument("--ctl-gctx", default="data/cmap/level3_beta_ctl_n188708x12328.gctx")
    parser.add_argument("--landmark", default="data/landmark_genes.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()

    usecols = ["pert_type", "pert_time", "distil_ids", "cell_iname", "bead_batch", "det_plates"]
    sig = pd.read_csv(args.siginfo, sep="\t", low_memory=False, usecols=usecols)
    ctl = sig[(sig["pert_type"] == "ctl_vehicle") & (sig["pert_time"] == 24)].copy()
    ctl["distil_ids"] = ctl["distil_ids"].astype(str).str.split("|")
    ctl = ctl.explode("distil_ids").rename(columns={"distil_ids": "distil_id"})
    ctl = ctl.drop_duplicates(subset=["distil_id"])

    rng = np.random.default_rng(args.seed)
    if len(ctl) > args.n_samples:
        ctl = ctl.iloc[rng.choice(len(ctl), size=args.n_samples, replace=False)]

    ctl = ctl.reset_index(drop=True)
    sample_ids = ctl["distil_id"].astype(str).tolist()
    genes = _load_landmark_entrez_ids(args.landmark)

    x = _read_gctx_subset(args.ctl_gctx, sample_ids, genes)
    if x.shape[0] == 0:
        raise RuntimeError("No matching distil_id found in control GCTX.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    xz = scaler.fit_transform(x)
    pca = PCA(n_components=2, random_state=args.seed)
    emb = pca.fit_transform(xz)

    meta = ctl.iloc[: emb.shape[0]].copy()
    meta["cell_iname"] = _categorize_top(meta["cell_iname"], args.top_n)
    meta["bead_batch"] = _categorize_top(meta["bead_batch"], args.top_n)
    meta["det_plates"] = _categorize_top(meta["det_plates"], args.top_n)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True, sharey=True)
    panels = [("cell_iname", "Cell line"), ("bead_batch", "Bead batch"), ("det_plates", "Detection plate")]
    for ax, (col, title) in zip(axes, panels):
        cats = meta[col].astype(str).tolist()
        uniq = sorted(set(cats), key=lambda z: (z == "Other", z))
        colors = _palette(len(uniq))
        color_map = {u: colors[i] for i, u in enumerate(uniq)}
        c = [color_map[v] for v in cats]
        ax.scatter(emb[:, 0], emb[:, 1], s=6, c=c, alpha=0.85, linewidths=0.0)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        if ax is axes[0]:
            ax.set_ylabel("PC2")

        handles = []
        for u in uniq:
            if u == "Other":
                continue
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=3.5,
                    color=color_map[u],
                    label=_shorten(u) if col == "det_plates" else u,
                )
            )
        if "Other" in uniq:
            handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=color_map["Other"], label="Other"))
        ax.legend(handles=handles, frameon=False, fontsize=6, loc="best", handletextpad=0.3, borderpad=0.2)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
