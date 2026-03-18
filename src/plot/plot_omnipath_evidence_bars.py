import argparse
import os

import matplotlib.pyplot as plt
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


def _bin_counts(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    bins = [-0.1, 1, 2, 3, 4, 5, 10, 20, 50, 100, float("inf")]
    labels = ["1", "2", "3", "4", "5", "6–10", "11–20", "21–50", "51–100", ">100"]
    cat = pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=True)
    return cat.value_counts().reindex(labels).fillna(0).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/omnipath/omnipath_interactions.csv")
    parser.add_argument("--tf", default="data/omnipath/omnipath_tf_regulons.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()
    pal = _palette()

    metrics = ["n_sources", "n_primary_sources", "n_references"]
    df_i = pd.read_csv(args.interactions, usecols=metrics)
    df_t = pd.read_csv(args.tf, usecols=metrics)

    counts = {}
    for m in metrics:
        counts[("OmniPath interactions", m)] = _bin_counts(df_i[m])
        counts[("OmniPath TF regulons", m)] = _bin_counts(df_t[m])

    bins = counts[("OmniPath interactions", metrics[0])].index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2), sharey=True)
    fig.subplots_adjust(wspace=0.14, hspace=0.22)
    axes_flat = axes.ravel()

    colors = [pal[0], pal[2]]
    edge = to_rgba("0.2", 0.8)

    for ax, m in zip(axes_flat[:3], metrics):
        x = list(range(len(bins)))
        w = 0.38
        c_i = counts[("OmniPath interactions", m)].values
        c_t = counts[("OmniPath TF regulons", m)].values

        ax.bar([xi - w / 2 for xi in x], c_i, width=w, color=to_rgba(colors[0], 0.75), edgecolor=edge, linewidth=0.6, label="interactions")
        ax.bar([xi + w / 2 for xi in x], c_t, width=w, color=to_rgba(colors[1], 0.75), edgecolor=edge, linewidth=0.6, label="tf_regulons")

        ax.set_title(m.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=35, ha="right")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)

    axes_flat[0].set_ylabel("Number of rows")

    ax_legend = axes_flat[3]
    ax_legend.set_axis_off()

    fig.suptitle("OmniPath evidence statistics (binned counts)")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
