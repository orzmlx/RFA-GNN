import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def _count_profiles(distil_ids: pd.Series) -> int:
    s = distil_ids.astype(str)
    return int((s.str.count(r"\|") + 1).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siginfo", default="data/siginfo_beta.txt")
    parser.add_argument("--compoundinfo", default="data/compoundinfo_beta.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()

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

    trt0 = sig[sig["pert_type"] == "trt_cp"].copy()
    trt1 = trt0[trt0["pert_time"] == 24].copy()
    trt2 = trt1[trt1["pert_dose"] == 10.0].copy()
    trt3 = trt2[trt2["is_hiq"] == 1].copy()

    comp = pd.read_csv(args.compoundinfo, sep="\t", low_memory=False, usecols=["pert_id", "target"])
    comp["target"] = comp["target"].astype(str)
    has_target = set(
        comp.loc[
            comp["target"].notna()
            & (comp["target"].astype(str).str.strip() != "")
            & (comp["target"].astype(str) != "nan"),
            "pert_id",
        ].astype(str)
    )
    trt4 = trt3[trt3["pert_id"].astype(str).isin(has_target)].copy()

    ctl = sig[(sig["pert_type"] == "ctl_vehicle") & (sig["pert_time"] == 24)].copy()

    trt4e = trt4.copy()
    trt4e["distil_ids"] = trt4e["distil_ids"].astype(str).str.split("|")
    trt4e = trt4e.explode("distil_ids").rename(columns={"distil_ids": "distil_id"})

    ctle = ctl.copy()
    ctle["distil_ids"] = ctle["distil_ids"].astype(str).str.split("|")
    ctle = ctle.explode("distil_ids").rename(columns={"distil_ids": "distil_id"})

    strict_key = ["cell_iname", "bead_batch", "det_plates"]
    relaxed_key = ["cell_iname", "det_plates"]
    ctl_strict = set(tuple(x) for x in ctle[strict_key].astype(str).to_numpy())
    ctl_relaxed = set(tuple(x) for x in ctle[relaxed_key].astype(str).to_numpy())

    trt_strict_hit = trt4e[strict_key].astype(str).apply(tuple, axis=1).isin(ctl_strict)
    trt_relaxed_hit = trt4e[relaxed_key].astype(str).apply(tuple, axis=1).isin(ctl_relaxed)
    trt5e = trt4e[trt_strict_hit | trt_relaxed_hit].copy()

    counts = [
        _count_profiles(trt0["distil_ids"]),
        _count_profiles(trt1["distil_ids"]),
        _count_profiles(trt2["distil_ids"]),
        _count_profiles(trt3["distil_ids"]),
        len(trt4e),
        len(trt5e),
    ]

    step_labels = [
        "All trt_cp\nprofiles",
        "24 h",
        "10 µM",
        "High quality\n(trt only)",
        "With targets",
        "Treatments with\nmatched control",
    ]

    base = max(counts[0], 1)
    perc = [c / base * 100.0 for c in counts]

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(len(step_labels))

    try:
        import colorbm

        cmap = colorbm.seq("batlow").as_cmap()
        colors = [cmap(v) for v in np.linspace(0.15, 0.85, len(x))]
    except Exception:
        colors = list(plt.get_cmap("tab10").colors)[: len(x)]

    ax.bar(x, perc, width=0.78, color=colors, alpha=0.22, edgecolor="none", zorder=1)
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], perc[i : i + 2], color=colors[i + 1], linewidth=2.0, zorder=3)
    ax.scatter(x, perc, s=22, c=colors, edgecolors="white", linewidths=0.5, zorder=4)

    for i, (p, c) in enumerate(zip(perc, counts)):
        ax.text(i, p + 2.5, f"{c:,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=0)
    ax.set_ylabel("Remaining treatment profiles (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.grid(axis="y", color="0.9", linewidth=0.6)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
