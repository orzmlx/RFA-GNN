from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
NPZ_PATH = ROOT / "tmp" / "gat_cf_loss.eval.npz"
OUT_PNG = ROOT / "liuthesis_my" / "figures" / "attention_best_worst_heatmap.png"
OUT_PDF = ROOT / "liuthesis_my" / "figures" / "attention_best_worst_heatmap.pdf"


def _load_npz():
    data = np.load(NPZ_PATH, allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        if value.shape == (1,) and value.dtype == object:
            out[key] = value[0].item() if hasattr(value[0], "item") else value[0]
        else:
            out[key] = value
    return out


def main():
    run = _load_npz()
    att = run["attention"]
    groups = att["group_attention_edge_mean"]
    edge_index = np.asarray(att["edge_index"], dtype=int)
    src_all = edge_index[0].astype(int)
    dst_all = edge_index[1].astype(int)
    non_self = src_all != dst_all

    drug_ids = np.asarray(run["drug_ids"], dtype=str)
    sample_pcc = np.asarray(run["sample_pcc"], dtype=float)

    rows = []
    for d in sorted(groups.keys()):
        mask = drug_ids == str(d)
        if int(np.sum(mask)) == 0:
            continue
        vals = sample_pcc[mask]
        rows.append((str(d), float(np.median(vals)), int(len(vals))))
    rows.sort(key=lambda x: x[1], reverse=True)

    n_show = min(8, len(rows) // 2)
    best_rows = rows[:n_show]
    worst_rows = rows[-n_show:]
    selected = best_rows + worst_rows

    alpha_rows = []
    for drug, _, _ in selected:
        a = np.asarray(groups[drug], dtype=float)
        if a.ndim == 2:
            a = a[-1]
        alpha_rows.append(a[non_self])
    alpha_mat = np.stack(alpha_rows, axis=0)

    per_drug_top_k = 24
    union_idx = set()
    for row in alpha_mat:
        union_idx.update(np.argsort(-row)[:per_drug_top_k].tolist())
    union_idx = np.asarray(sorted(union_idx), dtype=int)
    heat_abs = alpha_mat[:, union_idx]

    # Group columns so each drug can show a visible band of high-dependency edges.
    primary_owner = np.argmax(heat_abs, axis=0)
    peak_value = np.max(heat_abs, axis=0)
    order = np.lexsort((-peak_value, primary_owner))
    heat_abs = heat_abs[:, order]

    row_min = np.min(heat_abs, axis=1, keepdims=True)
    row_max = np.max(heat_abs, axis=1, keepdims=True)
    heat = (heat_abs - row_min) / (row_max - row_min + 1e-8)

    row_labels = [f"{drug} | {pcc:.3f}" for drug, pcc, _ in selected]

    fig, ax = plt.subplots(figsize=(13.0, 5.2), dpi=220)
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks([])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel("Drugs ordered by median sample-wise PCC", fontsize=10)
    ax.set_xlabel(f"Union of per-drug top-{per_drug_top_k} non-self edges", fontsize=10)
    ax.axhline(n_show - 0.5, color="#666666", linewidth=1.0, linestyle="--")

    # Light group labels instead of cluttered titles.
    ax.text(-0.02, 0.80, "Best", transform=ax.transAxes, fontsize=9, fontweight="bold", color="#4F6D8A",
            ha="right", va="center")
    ax.text(-0.02, 0.23, "Worst", transform=ax.transAxes, fontsize=9, fontweight="bold", color="#9C6644",
            ha="right", va="center")

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Within-drug normalized grouped attention", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
