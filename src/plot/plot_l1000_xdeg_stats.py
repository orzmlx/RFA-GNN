import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _read_gctx_rows(gctx_path: str, sample_ids: list[str], genes: list[str]) -> dict[str, np.ndarray]:
    with h5py.File(gctx_path, "r") as f:
        matrix = f["0/DATA/0/matrix"]
        row_ids = f["0/META/ROW/id"][:].astype(str)
        col_ids = f["0/META/COL/id"][:].astype(str)

        row_index = {v: i for i, v in enumerate(row_ids)}
        col_index = {v: i for i, v in enumerate(col_ids)}

        pairs = [(sid, col_index[sid]) for sid in sample_ids if sid in col_index]
        if not pairs:
            return {}

        pairs = sorted(pairs, key=lambda x: x[1])
        seen = set()
        keep_ids = []
        keep_idx = []
        for sid, idx in pairs:
            if idx in seen:
                continue
            seen.add(idx)
            keep_ids.append(sid)
            keep_idx.append(idx)

        gene_idx = [row_index[g] for g in genes if g in row_index]
        genes_present = [g for g in genes if g in row_index]
        gene_pos = {g: i for i, g in enumerate(genes)}

        if matrix.shape[0] == len(row_ids) and matrix.shape[1] == len(col_ids):
            data = matrix[:, keep_idx]
            data = np.asarray(data, dtype=np.float32)
            data = data[gene_idx, :].T
        elif matrix.shape[0] == len(col_ids) and matrix.shape[1] == len(row_ids):
            data = matrix[keep_idx, :]
            data = np.asarray(data, dtype=np.float32)
            data = data[:, gene_idx]
        else:
            raise ValueError("Unexpected GCTX matrix shape.")

    full = np.zeros((len(keep_ids), len(genes)), dtype=np.float32)
    for j, g in enumerate(genes_present):
        full[:, gene_pos[g]] = data[:, j]
    return {sid: full[i] for i, sid in enumerate(keep_ids)}


def _nonempty_targets(compoundinfo_path: str) -> set[str]:
    comp = pd.read_csv(compoundinfo_path, sep="\t", low_memory=False, usecols=["pert_id", "target"])
    tgt = comp["target"].astype(str)
    ok = tgt.notna() & (tgt.astype(str).str.strip() != "") & (tgt.astype(str) != "nan")
    return set(comp.loc[ok, "pert_id"].astype(str))


def _palette():
    try:
        import colorbm

        pal = colorbm.pal("lancet").as_hex
    except Exception:
        pal = None
    if not pal:
        pal = list(plt.get_cmap("tab10").colors)
    return pal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siginfo", default="data/siginfo_beta.txt")
    parser.add_argument("--compoundinfo", default="data/compoundinfo_beta.txt")
    parser.add_argument("--trt-gctx", default="data/cmap/level3_beta_trt_cp_n1805898x12328.gctx")
    parser.add_argument("--ctl-gctx", default="data/cmap/level3_beta_ctl_n188708x12328.gctx")
    parser.add_argument("--landmark", default="data/landmark_genes.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    pal = _palette()

    usecols = [
        "pert_type",
        "pert_time",
        "pert_dose",
        "is_hiq",
        "distil_ids",
        "pert_id",
        "cell_iname",
        "bead_batch",
        "det_plates",
    ]
    sig = pd.read_csv(args.siginfo, sep="\t", low_memory=False, usecols=usecols)

    trt = sig[
        (sig["pert_type"] == "trt_cp")
        & (sig["pert_time"] == 24)
        & (sig["pert_dose"] == 10.0)
        & (sig["is_hiq"] == 1)
    ].copy()
    has_target = _nonempty_targets(args.compoundinfo)
    trt = trt[trt["pert_id"].astype(str).isin(has_target)].copy()
    trt["distil_ids"] = trt["distil_ids"].astype(str).str.split("|")
    trt = trt.explode("distil_ids").rename(columns={"distil_ids": "distil_id"}).reset_index(drop=True)

    ctl = sig[(sig["pert_type"] == "ctl_vehicle") & (sig["pert_time"] == 24)].copy()
    ctl["distil_ids"] = ctl["distil_ids"].astype(str).str.split("|")
    ctl = ctl.explode("distil_ids").rename(columns={"distil_ids": "distil_id"})
    ctl = ctl.drop_duplicates(subset=["distil_id"]).reset_index(drop=True)

    strict_key = ["cell_iname", "bead_batch", "det_plates"]
    relaxed_key = ["cell_iname", "det_plates"]

    ctl_strict = ctl.groupby(strict_key)["distil_id"].apply(list).to_dict()
    ctl_relaxed = ctl.groupby(relaxed_key)["distil_id"].apply(list).to_dict()

    rng = np.random.default_rng(args.seed)
    trt = trt.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    pairs = []
    for _, row in trt.iterrows():
        k1 = tuple(row[c] for c in strict_key)
        cands = ctl_strict.get(k1, [])
        if not cands:
            k2 = tuple(row[c] for c in relaxed_key)
            cands = ctl_relaxed.get(k2, [])
        if not cands:
            continue
        ctl_id = cands[int(rng.integers(0, len(cands)))]
        pairs.append((str(row["distil_id"]), str(ctl_id)))
        if len(pairs) >= args.n_pairs:
            break

    if not pairs:
        raise RuntimeError("No matched treatment-control pairs found.")

    trt_ids = [p[0] for p in pairs]
    ctl_ids = [p[1] for p in pairs]

    genes = _load_landmark_entrez_ids(args.landmark)
    trt_map = _read_gctx_rows(args.trt_gctx, trt_ids, genes)
    ctl_map = _read_gctx_rows(args.ctl_gctx, ctl_ids, genes)

    xdeg = []
    for trt_id, ctl_id in pairs:
        xt = trt_map.get(trt_id)
        xc = ctl_map.get(ctl_id)
        if xt is None or xc is None:
            continue
        xdeg.append(xt - xc)
    if not xdeg:
        raise RuntimeError("No pairs could be loaded from GCTX.")
    xdeg = np.stack(xdeg, axis=0).astype(np.float32)

    mean = xdeg.mean(axis=1)
    std = xdeg.std(axis=1)
    l2 = np.linalg.norm(xdeg, axis=1)
    up = (xdeg > args.threshold).sum(axis=1).astype(int)
    down = (xdeg < -args.threshold).sum(axis=1).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

    axes[0].hist(mean, bins=40, color=pal[0], alpha=0.35, edgecolor="none")
    axes[0].axvline(0.0, color="0.3", linewidth=1.0)
    axes[0].set_title("Mean of $x_{deg}$")
    axes[0].set_xlabel("Mean")
    axes[0].set_ylabel("Count")

    axes[1].hist(l2, bins=40, color=pal[2], alpha=0.35, edgecolor="none")
    axes[1].set_title(r"$\ell_2$ norm of $x_{deg}$")
    axes[1].set_xlabel(r"$\|x_{deg}\|_2$")
    axes[1].set_ylabel("Count")

    bins = np.arange(0, max(up.max(), down.max()) + 5, 5)
    axes[2].hist(up, bins=bins, color=pal[3], alpha=0.35, edgecolor="none", label=f"Up (>{args.threshold:g})")
    axes[2].hist(down, bins=bins, color=pal[1], alpha=0.35, edgecolor="none", label=f"Down (<-{args.threshold:g})")
    axes[2].set_title("Large changes per sample")
    axes[2].set_xlabel("# genes")
    axes[2].set_ylabel("Count")
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle(f"Distribution of treatment responses $x_{{deg}}=x_{{trt}}-x_{{ctl}}$ (n={xdeg.shape[0]:,})", y=1.02)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
