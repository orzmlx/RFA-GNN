import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import project_paths


PATHS = project_paths()
ROOT = PATHS["root"]
FIG_DIR = PATHS["fig_dir"]
RESULT_TEX = PATHS["result_tex"]
BASELINE_CSV = ROOT / "tmp" / "pre_eval_latest" / "simple_baseline_metrics.csv"

def parse_result_table_by_label(table_label):
    text = RESULT_TEX.read_text(encoding="utf-8")
    table_pattern = re.compile(
        rf"\\label\{{{re.escape(table_label)}\}}.*?\\begin\{{tabular\}}\{{[^}}]+\}}(.*?)\\end\{{tabular\}}",
        re.DOTALL,
    )
    match = table_pattern.search(text)
    if not match:
        raise RuntimeError(f"Could not find table {table_label} in Result.tex")
    rows = {}
    headers = None
    for line in match.group(1).strip().splitlines():
        clean = line.strip()
        if not clean or "\\hline" in clean:
            continue
        parts = [x.strip().replace("\\textbf{", "").replace("}", "") for x in clean.replace("\\\\", "").split("&")]
        if parts and parts[0] == "Split":
            headers = parts[1:]
            continue
        if headers is None or len(parts) != len(headers) + 1:
            continue
        split = parts[0]
        for header, value in zip(headers, parts[1:]):
            try:
                rows[(header, split)] = float(value)
            except ValueError:
                continue
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
    table_label = "tab:mse_results" if metric_key == "test_mse" else "tab:pcc_results"
    learned = parse_result_table_by_label(table_label)
    baseline_rows = load_baselines()
    split_map = {"Warm": "warm", "Cold drug target": "cold_drug", "Cold cell": "cold_cell"}
    learned_split_map = {"Warm": "Warm", "Cold drug target": "Cold drug target", "Cold cell": "Cold cell"}
    display_splits = ["Warm", "Cold drug target", "Cold cell"]

    series = {
        "DeepCOP": [],
        "GSNN": [],
        "UPert no-CF": [],
        "UPert CF": [],
        "mean_global": [],
        "mean_cell": [],
        "mean_drug": [],
    }
    for display_split in display_splits:
        split = split_map[display_split]
        learned_split = learned_split_map[display_split]
        for model in ["DeepCOP", "GSNN", "UPert no-CF", "UPert CF"]:
            series[model].append(learned[(model, learned_split)])
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
    width = 0.11

    colors = {
        "DeepCOP": "#C9DDF2",
        "GSNN": "#F7D8B5",
        "UPert no-CF": "#CFE8C7",
        "UPert CF": "#BFD9F1",
        "mean_global": "#DEDEDE",
        "mean_cell": "#E3D2EE",
        "mean_drug": "#DCE7BF",
    }
    labels = {
        "DeepCOP": "DeepCOP",
        "GSNN": "GSNN",
        "UPert no-CF": "UPert no-CF",
        "UPert CF": "UPert CF",
        "mean_global": "mean_global",
        "mean_cell": "mean_cell",
        "mean_drug": "mean_drug",
    }
    split_orders = {
        "Warm": ["DeepCOP", "GSNN", "UPert no-CF", "UPert CF", "mean_global", "mean_cell", "mean_drug"],
        "Cold drug target": ["DeepCOP", "GSNN", "UPert no-CF", "UPert CF", "mean_global", "mean_cell"],
        "Cold cell": ["DeepCOP", "GSNN", "UPert no-CF", "UPert CF", "mean_global", "mean_drug"],
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
    return png_path, pdf_path


def generate_all():
    outputs = []
    outputs.extend(draw("test_mse", "Test MSE", "Test MSE", "benchmark_mse_with_mean_baselines"))
    outputs.extend(draw("test_pcc", "Test PCC", "Test PCC", "benchmark_pcc_with_mean_baselines"))
    return outputs


def main():
    for path in generate_all():
        print(path)


if __name__ == "__main__":
    main()
