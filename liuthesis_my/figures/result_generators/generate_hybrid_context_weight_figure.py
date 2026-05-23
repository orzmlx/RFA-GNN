from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
LOG_PATH = ROOT / "out.log"
FIG_DIR = ROOT / "liuthesis_my" / "figures"

SPLITS = ["warm", "cold_drug", "cold_cell"]
SPLIT_TITLES = {
    "warm": "Warm",
    "cold_drug": "Cold drug target",
    "cold_cell": "Cold cell",
}

FALLBACK_HISTORIES = {
    "cold_drug": [
        (1, 0.0511, 0.9489), (2, 0.0522, 0.9478), (3, 0.0532, 0.9468), (4, 0.0547, 0.9453),
        (5, 0.0565, 0.9435), (6, 0.0592, 0.9408), (7, 0.0632, 0.9368), (8, 0.0694, 0.9306),
        (9, 0.0800, 0.9200), (10, 0.0945, 0.9055), (11, 0.1120, 0.8880), (12, 0.1311, 0.8689),
        (13, 0.1500, 0.8500), (14, 0.1709, 0.8291), (15, 0.1932, 0.8068), (16, 0.2114, 0.7886),
        (17, 0.2313, 0.7687), (18, 0.2491, 0.7509), (19, 0.2661, 0.7339), (20, 0.2841, 0.7159),
        (21, 0.3000, 0.7000), (22, 0.3131, 0.6869), (23, 0.3274, 0.6726), (24, 0.3403, 0.6597),
        (25, 0.3512, 0.6488), (26, 0.3616, 0.6384), (27, 0.3726, 0.6274), (28, 0.3806, 0.6194),
        (29, 0.3954, 0.6046), (30, 0.4011, 0.5989), (31, 0.4093, 0.5907), (32, 0.4188, 0.5812),
        (33, 0.4266, 0.5734), (34, 0.4342, 0.5658), (35, 0.4409, 0.5591), (36, 0.4470, 0.5530),
        (37, 0.4530, 0.5470), (38, 0.4593, 0.5407), (39, 0.4651, 0.5349), (40, 0.4712, 0.5288),
        (41, 0.4773, 0.5227), (42, 0.4833, 0.5167), (43, 0.4879, 0.5121), (44, 0.4918, 0.5082),
        (45, 0.4953, 0.5047), (46, 0.5010, 0.4990), (47, 0.5057, 0.4943), (48, 0.5076, 0.4924),
        (49, 0.5123, 0.4877), (50, 0.5146, 0.4854),
    ],
    "cold_cell": [
        (1, 0.0519, 0.9481), (2, 0.0529, 0.9471), (3, 0.0540, 0.9460), (4, 0.0551, 0.9449),
        (5, 0.0566, 0.9434), (6, 0.0588, 0.9412), (7, 0.0623, 0.9377), (8, 0.0670, 0.9330),
        (9, 0.0745, 0.9255), (10, 0.0846, 0.9154), (11, 0.0972, 0.9028), (12, 0.1128, 0.8872),
        (13, 0.1297, 0.8703), (14, 0.1455, 0.8545), (15, 0.1631, 0.8369), (16, 0.1779, 0.8221),
        (17, 0.1950, 0.8050), (18, 0.2077, 0.7923), (19, 0.2213, 0.7787), (20, 0.2331, 0.7669),
        (21, 0.2453, 0.7547), (22, 0.2553, 0.7447), (23, 0.2663, 0.7337), (24, 0.2788, 0.7212),
        (25, 0.2892, 0.7108), (26, 0.3002, 0.6998), (27, 0.3117, 0.6883), (28, 0.3217, 0.6783),
        (29, 0.3308, 0.6692), (30, 0.3414, 0.6586), (31, 0.3501, 0.6499), (32, 0.3595, 0.6405),
        (33, 0.3663, 0.6337), (34, 0.3760, 0.6240), (35, 0.3835, 0.6165), (36, 0.3902, 0.6098),
        (37, 0.3964, 0.6036), (38, 0.4001, 0.5999), (39, 0.4070, 0.5930), (40, 0.4128, 0.5872),
        (41, 0.4177, 0.5823), (42, 0.4241, 0.5759), (43, 0.4304, 0.5696), (44, 0.4345, 0.5655),
        (45, 0.4401, 0.5599), (46, 0.4427, 0.5573), (47, 0.4473, 0.5527), (48, 0.4510, 0.5490),
        (49, 0.4546, 0.5454), (50, 0.4593, 0.5407),
    ],
}

EPOCH_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+):.*?cell_scale=(?P<cell>[0-9.]+)\s+context_scale=(?P<context>[0-9.]+)"
)
SPLIT_RE = re.compile(r"^===== Running split: (?P<split>[a-z_]+) =====$")


def parse_histories():
    histories = {split: {"epoch": [], "cell_scale": [], "context_scale": []} for split in SPLITS}
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()

    # Keep only the latest warm block that belongs to the final hybrid uncertainty run.
    warm_save_marker = "Saved run meta to: /local/data1/liume102/rfa/results/gat_hybrid_uncertainty_all_splits.warm.json"
    warm_end = next((i for i, line in enumerate(lines) if warm_save_marker in line), None)
    if warm_end is not None:
        warm_start = None
        for i in range(warm_end, -1, -1):
            match = SPLIT_RE.match(lines[i].strip())
            if match and match.group("split") == "warm":
                warm_start = i
                break
        if warm_start is not None:
            for line in lines[warm_start:warm_end + 1]:
                epoch_match = EPOCH_RE.match(line.strip())
                if epoch_match:
                    histories["warm"]["epoch"].append(int(epoch_match.group("epoch")))
                    histories["warm"]["cell_scale"].append(float(epoch_match.group("cell")))
                    histories["warm"]["context_scale"].append(float(epoch_match.group("context")))

    for split, hist in histories.items():
        if not hist["epoch"] and split in FALLBACK_HISTORIES:
            hist["epoch"] = [epoch for epoch, _, _ in FALLBACK_HISTORIES[split]]
            hist["cell_scale"] = [cell for _, cell, _ in FALLBACK_HISTORIES[split]]
            hist["context_scale"] = [context for _, _, context in FALLBACK_HISTORIES[split]]
        if not hist["epoch"]:
            raise ValueError(f"No hybrid context history found for split: {split}")
    return histories


def style_axis(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_xlim(1, 50)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    histories = parse_histories()

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), dpi=220, sharey=True)
    cell_color = "#A8D5BA"
    context_color = "#AFCBEA"

    for ax, split in zip(axes, SPLITS):
        hist = histories[split]
        ax.plot(
            hist["epoch"],
            hist["cell_scale"],
            color=cell_color,
            linewidth=2.2,
            label="cell_scale",
        )
        ax.plot(
            hist["epoch"],
            hist["context_scale"],
            color=context_color,
            linewidth=2.2,
            label="context_scale",
        )
        style_axis(ax, SPLIT_TITLES[split])
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#D0D0D0")

    fig.suptitle("Evolution of hybrid context weights across evaluation splits", y=1.02, fontsize=13)
    fig.tight_layout()

    png_path = FIG_DIR / "hybrid_context_weights_all_splits.png"
    pdf_path = FIG_DIR / "hybrid_context_weights_all_splits.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
