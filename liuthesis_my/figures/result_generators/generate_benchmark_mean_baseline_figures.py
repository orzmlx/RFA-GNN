import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
FIG_DIR = ROOT / "liuthesis_my" / "figures"
RESULT_TEX = ROOT / "liuthesis_my" / "Result.tex"
BASELINE_CSV = ROOT / "tmp" / "pre_eval_latest" / "simple_baseline_metrics.csv"

# Latest CAGNN values extracted from out.log in this session.
CAGNN_OUT_LOG_RESULTS = {
    ("CAGNN", "Warm"): {"test_mse": 0.8182, "test_pcc": 0.5605},
    ("CAGNN", "Cold drug"): {"test_mse": 0.9516, "test_pcc": 0.5424},
    ("CAGNN", "Cold cell"): {"test_mse": 0.8384, "test_pcc": 0.4783},
}


def parse_result_table(metric_name):
    text = RESULT_TEX.read_text(encoding="utf-8")
    table_pattern = re.compile(
        rf"Model & Split & {re.escape(metric_name)} \\\\\n\\hline\n(.*?)\\hline",
        re.DOTALL,
    )
    match = table_pattern.search(text)
    if not match:
        raise RuntimeError(f"Could not find table for {metric_name} in Result.tex")
    rows = {}
    for line in match.group(1).strip().splitlines():
        parts = [x.strip() for x in line.replace("\\\\", "").split("&")]
        if len(parts) != 3:
            continue
        model, split, value = parts
        rows[(model, split)] = float(value)
    return rows


def load_baselines():
    with BASELINE_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def pick_baseline(rows, split, baseline_name, metric_key):
    for row in rows:
        if row["split_mode"] == split and row["baseline"] == baseline_name:
            return float(row[metric_key])
    raise KeyError((split, baseline_name, metric_key))


def build_plot_rows(metric_key, metric_title):
    learned = parse_result_table(metric_title)
    baseline_rows = load_baselines()
    split_map = {"Warm": "warm", "Cold drug": "cold_drug", "Cold cell": "cold_cell"}
    display_splits = ["Warm", "Cold drug", "Cold cell"]

    series = {
        "DeepCOP": [],
        "GSNN": [],
        "CAGNN": [],
        "mean_global": [],
        "mean_cell": [],
        "mean_drug": [],
    }
    for display_split in display_splits:
        split = split_map[display_split]
        for model in ["DeepCOP", "GSNN", "CAGNN"]:
            if model == "CAGNN":
                series[model].append(CAGNN_OUT_LOG_RESULTS[(model, display_split)][metric_key])
            else:
                series[model].append(learned[(model, display_split)])
        series["mean_global"].append(pick_baseline(baseline_rows, split, "global_mean", metric_key))
        series["mean_cell"].append(
            np.nan if split == "cold_cell"
            else pick_baseline(baseline_rows, split, "cell_mean_with_global_fallback", metric_key)
        )
        series["mean_drug"].append(
            np.nan if split == "cold_drug"
            else pick_baseline(baseline_rows, split, "drug_mean_with_global_fallback", metric_key)
        )
    return display_splits, series


def draw(metric_key, metric_title, y_label, out_name):
    display_splits, series = build_plot_rows(metric_key, metric_title)
    x = np.arange(len(display_splits))
    width = 0.13

    colors = {
        "DeepCOP": "#C9DDF2",
        "GSNN": "#F7D8B5",
        "CAGNN": "#CFE8C7",
        "mean_global": "#DEDEDE",
        "mean_cell": "#E3D2EE",
        "mean_drug": "#DCE7BF",
    }
    labels = {
        "DeepCOP": "DeepCOP",
        "GSNN": "GSNN",
        "CAGNN": "CAGNN",
        "mean_global": "mean_global",
        "mean_cell": "mean_cell",
        "mean_drug": "mean_drug",
    }
    order = ["DeepCOP", "GSNN", "CAGNN", "mean_global", "mean_cell", "mean_drug"]
    split_orders = {
        "Warm": ["DeepCOP", "GSNN", "CAGNN", "mean_global", "mean_cell", "mean_drug"],
        "Cold drug": ["DeepCOP", "GSNN", "CAGNN", "mean_global", "mean_cell"],
        "Cold cell": ["DeepCOP", "GSNN", "CAGNN", "mean_global", "mean_drug"],
    }

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=220)
    labeled = set()
    for split_idx, split_name in enumerate(display_splits):
        active_order = split_orders[split_name]
        offsets = (np.arange(len(active_order)) - (len(active_order) - 1) / 2.0) * width
        for off, key in zip(offsets, active_order):
            val = float(series[key][split_idx])
            if np.isnan(val):
                continue
            label = labels[key] if key not in labeled else None
            ax.bar(
                x[split_idx] + off,
                val,
                width=width,
                color=colors[key],
                label=label,
                edgecolor="#8A8A8A",
                linewidth=0.4,
            )
            labeled.add(key)

    ax.set_xticks(x)
    ax.set_xticklabels(display_splits)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()

    png_path = FIG_DIR / f"{out_name}.png"
    pdf_path = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    draw("test_mse", "Test MSE", "Test MSE", "benchmark_mse_with_mean_baselines")
    draw("test_pcc", "Test PCC", "Test PCC", "benchmark_pcc_with_mean_baselines")


if __name__ == "__main__":
    main()
