from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
FIG_DIR = ROOT / "liuthesis_my" / "figures"
LANDMARK_PATH = ROOT / "data" / "landmark_genes.json"
SPLIT_PATHS = {
    "Warm": ROOT / "outputs" / "no_cf_0523" / "ugat_no_cf_uncertainty_sparse.eval.warm.npz",
    "Cold drug target": ROOT / "outputs" / "no_cf_0523" / "ugat_no_cf_uncertainty_sparse.eval.cold_target_pattern.npz",
    "Cold cell": ROOT / "outputs" / "no_cf_0523" / "ugat_no_cf_uncertainty_sparse.eval.cold_cell.npz",
}
COLORS = {
    "Warm": {"fill": "#F6D7A7", "line": "#C48A2B"},
    "Cold drug target": {"fill": "#CFE1F5", "line": "#5C8FBD"},
    "Cold cell": {"fill": "#CFE8C7", "line": "#6FAF6A"},
}
MARKERS = {
    "Warm": "o",
    "Cold drug target": "s",
    "Cold cell": "^",
}


def load_symbol_map():
    with LANDMARK_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {str(item["entrez_id"]): str(item["gene_symbol"]) for item in items}


def load_runs():
    runs = {}
    for split_name, path in SPLIT_PATHS.items():
        z = np.load(path, allow_pickle=True)
        runs[split_name] = {key: z[key] for key in z.files}
    return runs


def sample_uncertainty(y_logvar):
    sigma = np.exp(0.5 * np.asarray(y_logvar, dtype=np.float32))
    return np.mean(sigma, axis=1)


def summarize_run(data):
    unc = sample_uncertainty(data["y_logvar"])
    mse = np.asarray(data["sample_mse"], dtype=np.float32)
    pcc = np.asarray(data["sample_pcc"], dtype=np.float32)
    q1, q2 = np.quantile(unc, [1.0 / 3.0, 2.0 / 3.0])
    groups = {
        "Low": unc <= q1,
        "Medium": (unc > q1) & (unc <= q2),
        "High": unc > q2,
    }
    return {
        "unc": unc,
        "mse": mse,
        "pcc": pcc,
        "corr_mse": float(np.corrcoef(unc, mse)[0, 1]),
        "corr_pcc": float(np.corrcoef(unc, pcc)[0, 1]),
        "group_mse": {name: float(np.mean(mse[mask])) for name, mask in groups.items()},
        "group_pcc": {name: float(np.mean(pcc[mask])) for name, mask in groups.items()},
    }


def save_summary_figure(runs):
    summaries = {name: summarize_run(data) for name, data in runs.items()}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), dpi=220)

    ax = axes[0, 0]
    box_data = [summaries[name]["unc"] for name in summaries]
    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "#666666", "linewidth": 1.0},
        boxprops={"edgecolor": "#7A7A7A", "linewidth": 0.8},
        whiskerprops={"color": "#7A7A7A", "linewidth": 0.8},
        capprops={"color": "#7A7A7A", "linewidth": 0.8},
    )
    for patch, name in zip(bp["boxes"], summaries.keys()):
        patch.set_facecolor(COLORS[name]["fill"])
    ax.set_xticks(np.arange(1, len(summaries) + 1))
    ax.set_xticklabels(list(summaries.keys()))
    ax.set_ylabel("Average predicted sigma")
    ax.set_title("Sample-level uncertainty by split")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)

    ax = axes[0, 1]
    for name, summary in summaries.items():
        unc = summary["unc"]
        mse = summary["mse"]
        ax.scatter(
            unc,
            mse,
            s=18,
            alpha=0.55,
            color=COLORS[name]["line"],
            marker=MARKERS[name],
            edgecolors="white",
            linewidths=0.25,
        )
    ax.text(
        0.03,
        0.97,
        "\n".join([f"{name}: r = {summaries[name]['corr_mse']:.2f}" for name in summaries]),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92},
    )
    ax.set_xlabel("Average predicted sigma")
    ax.set_ylabel("Sample-wise MSE")
    ax.set_title("Uncertainty and MSE")
    ax.grid(alpha=0.16, linewidth=0.8)

    ax = axes[1, 0]
    for name, summary in summaries.items():
        unc = summary["unc"]
        pcc = summary["pcc"]
        ax.scatter(
            unc,
            pcc,
            s=18,
            alpha=0.55,
            color=COLORS[name]["line"],
            marker=MARKERS[name],
            edgecolors="white",
            linewidths=0.25,
        )
    ax.text(
        0.03,
        0.97,
        "\n".join([f"{name}: r = {summaries[name]['corr_pcc']:.2f}" for name in summaries]),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92},
    )
    ax.set_xlabel("Average predicted sigma")
    ax.set_ylabel("Sample-wise PCC")
    ax.set_title("Uncertainty and PCC")
    ax.grid(alpha=0.16, linewidth=0.8)

    scatter_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[name],
            linestyle="",
            markersize=7,
            markerfacecolor=COLORS[name]["line"],
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        for name in summaries.keys()
    ]
    axes[0, 1].legend(scatter_handles, list(summaries.keys()), frameon=False, fontsize=9, loc="upper right")
    axes[1, 0].legend(scatter_handles, list(summaries.keys()), frameon=False, fontsize=9, loc="lower right")

    ax = axes[1, 1]
    group_names = ["Low", "Medium", "High"]
    xpos = np.arange(len(group_names))
    width = 0.24
    offsets = [-width, 0.0, width]
    for offset, name in zip(offsets, summaries.keys()):
        vals = [summaries[name]["group_mse"][g] for g in group_names]
        ax.bar(
            xpos + offset,
            vals,
            width=width,
            color=COLORS[name]["fill"],
            edgecolor="#7A7A7A",
            linewidth=0.6,
            label=f"{name} MSE",
        )
    ax.set_ylabel("Mean sample-wise MSE")
    ax.set_xticks(xpos)
    ax.set_xticklabels(group_names)
    ax.set_title("Performance by uncertainty group")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax2 = ax.twinx()
    for name in summaries.keys():
        vals = [summaries[name]["group_pcc"][g] for g in group_names]
        ax2.plot(
            xpos,
            vals,
            color=COLORS[name]["line"],
            marker="o",
            linewidth=1.6,
            label=f"{name} PCC",
        )
    ax2.set_ylabel("Mean sample-wise PCC")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
    )

    for axis in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], ax2]:
        axis.spines["top"].set_visible(False)
    for axis in axes.ravel():
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "uncertainty_summary_placeholder.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "uncertainty_summary_placeholder.pdf", bbox_inches="tight")
    plt.close(fig)


def _select_case_indices(data):
    y_logvar = np.asarray(data["y_logvar"], dtype=np.float32)
    mse = np.asarray(data["sample_mse"], dtype=np.float32)
    pcc = np.asarray(data["sample_pcc"], dtype=np.float32)
    unc = sample_uncertainty(y_logvar)
    q_low, q_high = np.quantile(unc, [0.2, 0.8])
    low_idx = np.where(unc <= q_low)[0]
    high_idx = np.where(unc >= q_high)[0]
    return [
        low_idx[np.argmax(pcc[low_idx])],
        high_idx[np.argmax(mse[high_idx])],
    ]


def save_case_study_figure(runs):
    symbol_map = load_symbol_map()
    titles = ["Lower uncertainty and better fit", "Higher uncertainty and larger error"]
    split_names = list(runs.keys())
    fig, axes = plt.subplots(len(split_names), 2, figsize=(14.0, 11.5), dpi=220, sharey=False)
    if len(split_names) == 1:
        axes = np.asarray([axes])

    for row_idx, split_name in enumerate(split_names):
        data = runs[split_name]
        y_true = np.asarray(data["y_true"], dtype=np.float32)
        y_pred = np.asarray(data["y_pred"], dtype=np.float32)
        y_logvar = np.asarray(data["y_logvar"], dtype=np.float32)
        genes = np.asarray(data["target_genes"], dtype=str)
        mse = np.asarray(data["sample_mse"], dtype=np.float32)
        pcc = np.asarray(data["sample_pcc"], dtype=np.float32)
        unc = sample_uncertainty(y_logvar)
        chosen = _select_case_indices(data)

        for col_idx, (sample_idx, title) in enumerate(zip(chosen, titles)):
            ax = axes[row_idx, col_idx]
            truth = y_true[sample_idx]
            pred = y_pred[sample_idx]
            sigma = np.exp(0.5 * y_logvar[sample_idx])
            band = 2.0 * sigma
            top_idx = np.argsort(-np.abs(truth))[:12]
            labels = [symbol_map.get(genes[i], genes[i]) for i in top_idx]
            xs = np.arange(len(top_idx))
            coverage = np.mean(np.abs(truth[top_idx] - pred[top_idx]) <= band[top_idx])

            ax.plot(xs, truth[top_idx], color="#7A7A7A", marker="o", linewidth=1.2, label="Truth")
            ax.plot(xs, pred[top_idx], color=COLORS[split_name]["line"], marker="o", linewidth=1.2, label="Predicted mean")
            ax.fill_between(
                xs,
                pred[top_idx] - band[top_idx],
                pred[top_idx] + band[top_idx],
                color=COLORS[split_name]["fill"],
                alpha=0.55,
                label=r"$\mu \pm 2\sigma$",
            )
            ax.set_title(
                f"{split_name}: {title}\n"
                f"mean sigma = {unc[sample_idx]:.3f}, coverage = {coverage:.0%}, MSE = {mse[sample_idx]:.3f}, PCC = {pcc[sample_idx]:.3f}",
                fontsize=10,
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(r"$x_{\mathrm{deg}}$")
            ax.grid(axis="y", alpha=0.18, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "uncertainty_case_study_placeholder.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "uncertainty_case_study_placeholder.pdf", bbox_inches="tight")
    plt.close(fig)



def main():
    runs = load_runs()
    save_summary_figure(runs)
    save_case_study_figure(runs)


if __name__ == "__main__":
    main()
