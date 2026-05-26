import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_panel(npz_path, model_name, split_name):
    data = np.load(npz_path, allow_pickle=True)
    y_true = np.asarray(data["y_true"], dtype=np.float32).reshape(-1)
    y_pred = np.asarray(data["y_pred"], dtype=np.float32).reshape(-1)
    return {
        "model_name": str(model_name),
        "split_name": str(split_name),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def draw_figure(panels, out_path, seed=42, max_points=120000, axis_quantile=0.998):
    n_rows = len(panels)
    n_cols = len(panels[0])
    rng = np.random.default_rng(int(seed))
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    sampled_panels = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            panel = panels[i][j]
            y_true = panel["y_true"]
            y_pred = panel["y_pred"]
            n = len(y_true)
            if n > max_points:
                idx = rng.choice(n, size=max_points, replace=False)
                y_true = y_true[idx]
                y_pred = y_pred[idx]
            row.append({"y_true": y_true, "y_pred": y_pred, "panel": panel})
        sampled_panels.append(row)

    # Use one shared global axis range for all panels so every split and model
    # is visually comparable with the same x/y scale.
    pooled = []
    for i in range(n_rows):
        for j in range(n_cols):
            pooled.append(sampled_panels[i][j]["y_true"])
            pooled.append(sampled_panels[i][j]["y_pred"])
    pooled = np.concatenate(pooled, axis=0)
    q = float(axis_quantile)
    if not (0.5 < q < 1.0):
        raise ValueError("axis_quantile must be in (0.5, 1.0)")
    lower_q = 1.0 - q
    data_min = float(np.quantile(pooled, lower_q))
    data_max = float(np.quantile(pooled, q))
    spread = max(float(data_max - data_min), 1e-6)
    pad = 0.08 * spread
    global_lo = float(data_min - pad)
    global_hi = float(data_max + pad)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.6, 10.2),
        dpi=220,
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            payload = sampled_panels[i][j]
            panel = payload["panel"]
            y_true = payload["y_true"]
            y_pred = payload["y_pred"]
            lo, hi = global_lo, global_hi
            ax.hexbin(
                y_true,
                y_pred,
                gridsize=65,
                extent=(lo, hi, lo, hi),
                mincnt=1,
                linewidths=0,
                cmap="viridis",
                bins="log",
            )
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.8, color="#444444")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if i == 0:
                ax.set_title(panel["split_name"], pad=6)
            if i == n_rows - 1:
                ax.set_xlabel(r"$y_{\mathrm{true}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
            ax.tick_params(length=2.5, width=0.8, pad=1.5)
            ax.grid(False)
            ticks = np.linspace(lo, hi, 3)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if j == 0:
                ax.text(
                    -0.38,
                    0.5,
                    panel["model_name"],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    clip_on=False,
                )
    fig.text(0.105, 0.5, r"$y_{\mathrm{pred}}$", rotation=90, va="center", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.075, top=0.95, wspace=0.18, hspace=0.18)
    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    root = "/Users/liuxi/Desktop/RFA_GNN"
    split_labels = {
        "warm": "Warm",
        "cold_target_pattern": "Cold drug target",
        "cold_cell": "Cold cell",
    }
    model_files = [
        (
            "DeepCOP",
            {
                "warm": os.path.join(root, "results", "deepcop.pred.warm.npz"),
                "cold_target_pattern": os.path.join(root, "results", "cold_target_pattern", "deepcop_cold_target_pattern.pred.cold_target_pattern.npz"),
                "cold_cell": os.path.join(root, "results", "deepcop.pred.cold_cell.npz"),
            },
        ),
        (
            "GSNN",
            {
                "warm": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.warm.npz"),
                "cold_target_pattern": os.path.join(root, "results", "cold_target_pattern", "gsnn_cold_target_pattern.pred.cold_target_pattern.npz"),
                "cold_cell": os.path.join(root, "gsnn_res", "gsnn_978_allcells_results.pred.cold_cell.npz"),
            },
        ),
        (
            "UPert no-CF",
            {
                "warm": os.path.join(root, "outputs", "no_cf_0523", "ugat_no_cf_uncertainty_sparse.eval.warm.npz"),
                "cold_target_pattern": os.path.join(root, "outputs", "no_cf_0523", "ugat_no_cf_uncertainty_sparse.eval.cold_target_pattern.npz"),
                "cold_cell": os.path.join(root, "outputs", "no_cf_0523", "ugat_no_cf_uncertainty_sparse.eval.cold_cell.npz"),
            },
        ),
        (
            "UPert with CF",
            {
                "warm": os.path.join(root, "outputs", "with_cf_gat_0523", "cagnn_control_context.eval.warm.npz"),
                "cold_target_pattern": os.path.join(root, "outputs", "with_cf_gat_0523", "cagnn_control_context.eval.cold_target_pattern.npz"),
                "cold_cell": os.path.join(root, "outputs", "with_cf_gat_0523", "cagnn_control_context.eval.cold_cell.npz"),
            },
        ),
    ]
    split_order = ["warm", "cold_target_pattern", "cold_cell"]
    panels = []
    for model_name, split_files in model_files:
        row = []
        for split_mode in split_order:
            row.append(load_panel(split_files[split_mode], model_name, split_labels[split_mode]))
        panels.append(row)
    out_path = os.path.join(root, "liuthesis_my", "figures", "gene_expr_true_pred_all_models_by_split.png")
    draw_figure(panels, out_path=out_path, seed=42, max_points=120000)
    print(out_path)


if __name__ == "__main__":
    main()
