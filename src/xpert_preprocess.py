import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class L1000Paths:
    siginfo_path: str
    ctl_h5_path: str
    trt_h5_path: str
    landmark_genes_path: str


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5
    seed: int = 42
    test_ratio: float = 0.2


def load_landmark_genes(landmark_genes_path: str) -> List[str]:
    with open(landmark_genes_path, "r") as f:
        genes_meta = json.load(f)
    return [str(g["entrez_id"]) for g in genes_meta if "entrez_id" in g]


def _load_h5_axis(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        axis0 = data["axis0"][:].astype(str)
        axis1 = data["axis1"][:].astype(str)
    return axis0, axis1


def _read_h5_values(h5_path: str, sample_ids: Sequence[str], genes: Sequence[str]) -> np.ndarray:
    axis0, axis1 = _load_h5_axis(h5_path)
    axis0_index = {v: i for i, v in enumerate(axis0)}

    valid_pairs = [(axis0_index[sid], sid) for sid in sample_ids if sid in axis0_index]
    if not valid_pairs:
        return np.zeros((0, len(genes)), dtype=np.float32)

    valid_pairs = sorted(valid_pairs, key=lambda x: x[0])
    idx = [p[0] for p in valid_pairs]
    valid_ids_sorted = [p[1] for p in valid_pairs]

    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        vals = data["block0_values"][:, idx]

    df = pd.DataFrame(vals.T, index=valid_ids_sorted, columns=axis1)
    df = df.reindex(columns=list(genes), fill_value=0.0)
    return df.values.astype(np.float32)


def _read_gctx_df(gctx_path: str, sample_ids: Sequence[str], genes: Sequence[str]) -> pd.DataFrame:
    with h5py.File(gctx_path, "r") as f:
        matrix = f["0/DATA/0/matrix"]
        row_ids = f["0/META/ROW/id"][:].astype(str)
        col_ids = f["0/META/COL/id"][:].astype(str)

        row_index = {v: i for i, v in enumerate(row_ids)}
        col_index = {v: i for i, v in enumerate(col_ids)}

        unique_pairs: Dict[str, int] = {}
        for sid in sample_ids:
            if sid in col_index:
                unique_pairs[sid] = col_index[sid]

        if not unique_pairs:
            return pd.DataFrame(columns=list(genes))

        valid_pairs = sorted(unique_pairs.items(), key=lambda x: x[1])
        valid_samples = [p[0] for p in valid_pairs]
        sample_idx = [p[1] for p in valid_pairs]
        gene_idx = [row_index[g] for g in genes if g in row_index]

        if matrix.shape[0] == len(row_ids) and matrix.shape[1] == len(col_ids):
            data = matrix[:, sample_idx]
            data = data[gene_idx, :].T
        elif matrix.shape[0] == len(col_ids) and matrix.shape[1] == len(row_ids):
            data = matrix[sample_idx, :]
            data = data[:, gene_idx]
        else:
            raise ValueError("Unexpected GCTX matrix shape.")

    full = np.zeros((len(valid_samples), len(genes)), dtype=np.float32)
    gene_pos = {g: i for i, g in enumerate(genes)}
    for j, g in enumerate([g for g in genes if g in row_index]):
        full[:, gene_pos[g]] = data[:, j]

    return pd.DataFrame(full, index=valid_samples, columns=list(genes))


def _save_h5_csv(df: pd.DataFrame, output_prefix: str) -> Tuple[str, str]:
    h5_path = f"{output_prefix}.h5"
    csv_path = f"{output_prefix}.csv"
    df.to_hdf(h5_path, key="data", mode="w")
    df.to_csv(csv_path)
    return h5_path, csv_path


def _filter_siginfo(siginfo: pd.DataFrame) -> pd.DataFrame:
    required = ["pert_id", "cell_iname", "pert_time", "pert_dose", "distil_ids"]
    sig = siginfo.dropna(subset=[c for c in required if c in siginfo.columns]).copy()

    if "pert_time" in sig.columns:
        sig = sig[sig["pert_time"].isin([3, 6, 24])]

    if "is_hiq" in sig.columns:
        sig = sig[sig["is_hiq"] == 1]

    return sig


def _explode_distil_ids(siginfo: pd.DataFrame) -> pd.DataFrame:
    sig = siginfo.copy()
    sig["distil_ids"] = sig["distil_ids"].astype(str).str.split("|")
    sig = sig.explode("distil_ids").rename(columns={"distil_ids": "distil_id"})
    return sig


def _match_random_controls(
    trt: pd.DataFrame,
    ctl: pd.DataFrame,
    plate_col: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctl_grouped = ctl.groupby(plate_col)["distil_id"].apply(list).to_dict()

    def pick_control(plate_id: str) -> Optional[str]:
        ids = ctl_grouped.get(plate_id, [])
        if not ids:
            return None
        return ids[int(rng.integers(0, len(ids)))]

    trt = trt.copy()
    trt["ctl_distil_id"] = trt[plate_col].map(pick_control)
    return trt.dropna(subset=["ctl_distil_id"])


def _collapse_replicates(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_cols: Sequence[str],
) -> pd.DataFrame:
    collapsed = df.groupby(list(group_cols))[list(feature_cols)].mean().reset_index()
    return collapsed


def _bin_dose(dose_values: Iterable[float], bins: Optional[Sequence[float]]) -> pd.Series:
    dose_arr = pd.to_numeric(pd.Series(dose_values), errors="coerce")
    if bins is None:
        valid = dose_arr.dropna().astype(float)
        if valid.empty:
            return pd.Series([np.nan] * len(dose_arr))
        quantiles = np.quantile(np.log10(valid), np.linspace(0, 1, 11))
        bins = np.unique(quantiles)
    return pd.cut(np.log10(dose_arr.astype(float)), bins=bins, include_lowest=True)


def preprocess_l1000(
    paths: L1000Paths,
    plate_col: str = "det_plates",
    dose_bins: Optional[Sequence[float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    siginfo = pd.read_csv(paths.siginfo_path, sep="\t", low_memory=False)
    siginfo = _filter_siginfo(siginfo)

    if plate_col not in siginfo.columns:
        raise ValueError(f"Missing plate column: {plate_col}")

    landmark_genes = load_landmark_genes(paths.landmark_genes_path)

    sig_exp = _explode_distil_ids(siginfo)
    trt = sig_exp[sig_exp["pert_type"].isin(["trt_cp", "trt"])]
    ctl = sig_exp[sig_exp["pert_type"].isin(["ctl_vehicle", "ctl"])]

    paired = _match_random_controls(trt, ctl, plate_col=plate_col, seed=seed)

    trt_expr = _read_h5_values(paths.trt_h5_path, paired["distil_id"], landmark_genes)
    ctl_expr = _read_h5_values(paths.ctl_h5_path, paired["ctl_distil_id"], landmark_genes)

    if trt_expr.shape != ctl_expr.shape:
        raise ValueError("Expression matrices for trt and ctl do not align.")

    xpert = trt_expr
    xdeg = trt_expr - ctl_expr

    meta_cols = ["pert_id", "cell_iname", "pert_time", "pert_dose", plate_col]
    meta = paired[meta_cols].reset_index(drop=True)
    meta["dose_bin"] = _bin_dose(meta["pert_dose"], bins=dose_bins)

    xpert_cols = [f"xpert_{g}" for g in landmark_genes]
    xdeg_cols = [f"xdeg_{g}" for g in landmark_genes]

    full = pd.concat(
        [meta, pd.DataFrame(xpert, columns=xpert_cols), pd.DataFrame(xdeg, columns=xdeg_cols)],
        axis=1,
    )

    group_cols = ["pert_id", "cell_iname", "pert_time", "dose_bin"]
    full = _collapse_replicates(full, xpert_cols + xdeg_cols, group_cols)
    return full


def generate_cleaned_gctx_pairs(
    siginfo_path: str,
    trt_gctx_path: str,
    ctl_gctx_path: str,
    landmark_genes_path: str,
    output_dir: str = "data/cmap",
    output_suffix: str = "landmark_cleaned",
    plate_col: str = "det_plates",
    filter_time: int = 24,
    filter_dose: float = 10.0,
    seed: int = 42,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    siginfo = pd.read_csv(siginfo_path, sep="\t", low_memory=False)

    siginfo = siginfo[siginfo["pert_type"].isin(["trt_cp", "ctl_vehicle"])].copy()
    siginfo = siginfo.dropna(subset=["pert_id", "cell_iname", "pert_time", "distil_ids"])
    if "pert_dose" in siginfo.columns:
        trt_mask = siginfo["pert_type"] == "trt_cp"
        siginfo = siginfo[~trt_mask | siginfo["pert_dose"].notna()]

    if filter_time is not None:
        siginfo = siginfo[siginfo["pert_time"] == filter_time]

    if filter_dose is not None:
        dose_mask = (siginfo["pert_type"] == "ctl_vehicle") | (
            (siginfo["pert_type"] == "trt_cp") & (np.abs(siginfo["pert_dose"] - filter_dose) < 0.1)
        )
        siginfo = siginfo[dose_mask]

    if "is_hiq" in siginfo.columns:
        siginfo = siginfo[(siginfo["pert_type"] == "ctl_vehicle") | (siginfo["is_hiq"] == 1)]

    if plate_col not in siginfo.columns:
        raise ValueError(f"Missing plate column: {plate_col}")

    landmark_genes = load_landmark_genes(landmark_genes_path)

    sig_exp = _explode_distil_ids(siginfo)
    trt = sig_exp[sig_exp["pert_type"] == "trt_cp"]
    ctl = sig_exp[sig_exp["pert_type"] == "ctl_vehicle"]

    paired = _match_random_controls(trt, ctl, plate_col=plate_col, seed=seed)

    trt_df = _read_gctx_df(trt_gctx_path, paired["distil_id"], landmark_genes)
    ctl_df = _read_gctx_df(ctl_gctx_path, paired["ctl_distil_id"], landmark_genes)

    trt_df = trt_df.reindex(paired["distil_id"].values)
    ctl_df = ctl_df.reindex(paired["ctl_distil_id"].values)

    trt_valid = ~trt_df.isna().all(axis=1).to_numpy()
    ctl_valid = ~ctl_df.isna().all(axis=1).to_numpy()
    valid_mask = trt_valid & ctl_valid
    trt_df = trt_df.loc[valid_mask]
    ctl_df = ctl_df.loc[valid_mask]

    base_trt = os.path.splitext(os.path.basename(trt_gctx_path))[0]
    base_ctl = os.path.splitext(os.path.basename(ctl_gctx_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    trt_prefix = os.path.join(output_dir, f"{base_trt}_{output_suffix}")
    ctl_prefix = os.path.join(output_dir, f"{base_ctl}_{output_suffix}")

    trt_paths = _save_h5_csv(trt_df, trt_prefix)
    ctl_paths = _save_h5_csv(ctl_df, ctl_prefix)

    return trt_paths, ctl_paths


def build_l1000_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    subsets: Dict[str, pd.DataFrame] = {}
    subsets["L1000_full"] = df.copy()

    sdst = df[(df["pert_time"] == 24) & (df["pert_dose"].astype(float) == 10.0)]
    subsets["L1000_sdst"] = sdst

    mdmt_groups = df.groupby(["pert_id", "cell_iname"])["dose_bin"].nunique()
    mdmt_pairs = mdmt_groups[mdmt_groups > 1].index
    mdmt_mask = df.set_index(["pert_id", "cell_iname"]).index.isin(mdmt_pairs)
    subsets["L1000_mdmt"] = df[mdmt_mask].copy()

    mdmt_index = subsets["L1000_mdmt"].index
    subsets["L1000_mdmt_pretrain"] = df.drop(index=mdmt_index).copy()
    return subsets


def _kfold_assign(groups: Sequence[str], n_splits: int, seed: int) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    groups = list(groups)
    rng.shuffle(groups)
    fold_ids = {}
    for idx, g in enumerate(groups):
        fold_ids[g] = idx % n_splits
    return fold_ids


def split_warm_start(df: pd.DataFrame, cfg: SplitConfig) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    folds = np.array_split(indices, cfg.n_splits)

    splits = []
    for k in range(cfg.n_splits):
        test_idx = folds[k]
        train_idx = np.setdiff1d(indices, test_idx)
        splits.append((df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)))
    return splits


def split_cold_drug(df: pd.DataFrame, cfg: SplitConfig) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    drug_fold = _kfold_assign(df["pert_id"].unique(), cfg.n_splits, cfg.seed)
    splits = []
    for k in range(cfg.n_splits):
        test_mask = df["pert_id"].map(lambda d: drug_fold[d] == k)
        splits.append((df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)))
    return splits


def split_cold_cell(df: pd.DataFrame, cfg: SplitConfig) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    cell_fold = _kfold_assign(df["cell_iname"].unique(), cfg.n_splits, cfg.seed)
    splits = []
    for k in range(cfg.n_splits):
        test_mask = df["cell_iname"].map(lambda c: cell_fold[c] == k)
        splits.append((df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)))
    return splits


def split_cold_dose_time(df: pd.DataFrame, cfg: SplitConfig) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    key_cols = ["pert_id", "cell_iname", "dose_bin", "pert_time"]
    df = df.copy()
    df["dose_time_key"] = df[key_cols].astype(str).agg("|".join, axis=1)

    splits = []
    rng = np.random.default_rng(cfg.seed)

    for fold_id in range(cfg.n_splits):
        test_mask = np.zeros(len(df), dtype=bool)
        for (drug, cell), group in df.groupby(["pert_id", "cell_iname"]):
            keys = group["dose_time_key"].unique().tolist()
            rng.shuffle(keys)
            if not keys:
                continue
            fold_keys = np.array_split(keys, cfg.n_splits)
            test_keys = set(fold_keys[fold_id])
            test_mask |= group["dose_time_key"].isin(test_keys).values

        splits.append((df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)))

    return splits


def generate_splits(df: pd.DataFrame, strategy: str, cfg: Optional[SplitConfig] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    cfg = cfg or SplitConfig()
    if strategy == "warm-start":
        return split_warm_start(df, cfg)
    if strategy == "cold-drug":
        return split_cold_drug(df, cfg)
    if strategy == "cold-cell":
        return split_cold_cell(df, cfg)
    if strategy == "cold-dose-time":
        return split_cold_dose_time(df, cfg)

    raise ValueError(f"Unknown split strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XPert-style L1000 preprocessing utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gctx = subparsers.add_parser("clean-gctx", help="Generate cleaned TRT/CTL GCTX outputs")
    gctx.add_argument("--siginfo", required=True, help="Path to siginfo_beta.txt")
    gctx.add_argument("--trt-gctx", required=True, help="Path to TRT GCTX file")
    gctx.add_argument("--ctl-gctx", required=True, help="Path to CTL GCTX file")
    gctx.add_argument("--landmark", required=True, help="Path to landmark_genes.json")
    gctx.add_argument("--output-dir", default="data/cmap", help="Output directory")
    gctx.add_argument("--output-suffix", default="landmark_cleaned", help="Output suffix")
    gctx.add_argument("--plate-col", default="det_plates", help="Plate column in siginfo")
    gctx.add_argument("--filter-time", type=int, default=24, help="Filter pert_time")
    gctx.add_argument("--filter-dose", type=float, default=10.0, help="Filter pert_dose for trt_cp")
    gctx.add_argument("--seed", type=int, default=42, help="Random seed")

    h5 = subparsers.add_parser("preprocess-h5", help="Generate XPert features from H5 inputs")
    h5.add_argument("--siginfo", required=True, help="Path to siginfo_beta.txt")
    h5.add_argument("--trt-h5", required=True, help="Path to TRT H5 file")
    h5.add_argument("--ctl-h5", required=True, help="Path to CTL H5 file")
    h5.add_argument("--landmark", required=True, help="Path to landmark_genes.json")
    h5.add_argument("--plate-col", default="det_plates", help="Plate column in siginfo")
    h5.add_argument("--seed", type=int, default=42, help="Random seed")
    h5.add_argument("--output-prefix", required=True, help="Output prefix for H5/CSV")

    args = parser.parse_args()

    if args.command == "clean-gctx":
        trt_paths, ctl_paths = generate_cleaned_gctx_pairs(
            siginfo_path=args.siginfo,
            trt_gctx_path=args.trt_gctx,
            ctl_gctx_path=args.ctl_gctx,
            landmark_genes_path=args.landmark,
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
            plate_col=args.plate_col,
            filter_time=args.filter_time,
            filter_dose=args.filter_dose,
            seed=args.seed,
        )
        print("TRT outputs:", trt_paths)
        print("CTL outputs:", ctl_paths)
        return

    if args.command == "preprocess-h5":
        paths = L1000Paths(
            siginfo_path=args.siginfo,
            ctl_h5_path=args.ctl_h5,
            trt_h5_path=args.trt_h5,
            landmark_genes_path=args.landmark,
        )
        df = preprocess_l1000(paths, plate_col=args.plate_col, seed=args.seed)
        h5_path, csv_path = _save_h5_csv(df, args.output_prefix)
        print("Outputs:", (h5_path, csv_path))
        return


__all__ = [
    "L1000Paths",
    "SplitConfig",
    "load_landmark_genes",
    "generate_cleaned_gctx_pairs",
    "preprocess_l1000",
    "build_l1000_subsets",
    "generate_splits",
    "split_warm_start",
    "split_cold_drug",
    "split_cold_cell",
    "split_cold_dose_time",
    "main",
]


if __name__ == "__main__":
    main()
