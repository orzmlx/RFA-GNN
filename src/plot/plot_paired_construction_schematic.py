import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def _box(ax, x, y, w, h, text, facecolor="white"):
    from matplotlib.patches import FancyBboxPatch

    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="0.2",
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25", shrinkA=2, shrinkB=2),
        zorder=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    _apply_style()

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    try:
        import colorbm

        pal = colorbm.pal("lancet").as_hex
    except Exception:
        pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    c_trt = to_rgba(pal[0], 0.12)
    c_ctl = to_rgba(pal[1], 0.12)
    c_strict = to_rgba(pal[2], 0.12)
    c_fallback = to_rgba(pal[3], 0.12)
    c_out = to_rgba(pal[4], 0.12)

    _box(
        ax,
        0.05,
        0.55,
        0.32,
        0.30,
        "Treatment profile\n(distil_id)\ncell, dose/time,\nbead_batch, det_plates",
        facecolor=c_trt,
    )
    _box(
        ax,
        0.05,
        0.12,
        0.32,
        0.30,
        "Control pool\n(ctl_vehicle)\nmatched context",
        facecolor=c_ctl,
    )

    _box(
        ax,
        0.45,
        0.62,
        0.22,
        0.18,
        "Strict match\ncell + bead_batch\n+ det_plates",
        facecolor=c_strict,
    )
    _box(
        ax,
        0.45,
        0.30,
        0.22,
        0.18,
        "Fallback match\ncell + det_plates",
        facecolor=c_fallback,
    )

    _box(
        ax,
        0.74,
        0.46,
        0.21,
        0.24,
        "Sample K controls\n(K=3)\ncreate paired examples",
        facecolor=c_out,
    )

    _arrow(ax, 0.37, 0.70, 0.45, 0.71)
    _arrow(ax, 0.37, 0.27, 0.45, 0.39)
    _arrow(ax, 0.67, 0.71, 0.74, 0.58)
    _arrow(ax, 0.67, 0.39, 0.74, 0.58)

    ax.text(0.46, 0.52, "if none", ha="left", va="center", fontsize=7, color="0.35")
    ax.text(0.05, 0.02, "Purpose: construct matched treatment–control pairs to estimate response (trt − ctl) while controlling batch/context.", fontsize=7)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
