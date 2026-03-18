import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


def _apply_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["nature"])
    except Exception:
        pass


def _node(ax, xy, label, r=0.08, fc="#FFFFFF", ec="#222222", lw=1.5, fs=11):
    c = Circle(xy, r, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fs, color="#111111")


def _arrow(ax, p0, p1, color="#222222", lw=1.2, ms=14, rad=0.0):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def plot_message_passing(out_path):
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    ax.set_aspect("equal")
    ax.axis("off")

    pos = {
        "v": (0.0, 0.0),
        "u1": (-0.75, 0.55),
        "u2": (-0.85, -0.35),
        "u3": (0.55, 0.75),
        "u4": (0.85, -0.25),
    }
    for u in ["u1", "u2", "u3", "u4"]:
        p0 = pos[u]
        p1 = pos["v"]
        v = np.array(p1) - np.array(p0)
        v = v / (np.linalg.norm(v) + 1e-9)
        _arrow(ax, tuple(np.array(p0) + 0.10 * v), tuple(np.array(p1) - 0.10 * v), rad=0.0)

    _node(ax, pos["v"], r"$v$", r=0.10, fc="#FFF3E0")
    _node(ax, pos["u1"], r"$u_1$")
    _node(ax, pos["u2"], r"$u_2$")
    _node(ax, pos["u3"], r"$u_3$")
    _node(ax, pos["u4"], r"$u_4$")

    ax.text(0.0, -0.78, r"aggregate from neighbours", ha="center", va="center", fontsize=10)
    ax.text(
        0.0,
        -1.05,
        r"$h_v^{(\ell+1)}=\sigma\!\left(\sum_{u\in\mathcal{N}(v)} m_{u\rightarrow v}^{(\ell)}\right)$",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gat_attention(out_path):
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    ax.set_aspect("equal")
    ax.axis("off")

    pos = {
        "v": (0.55, 0.0),
        "u1": (-0.75, 0.55),
        "u2": (-0.85, -0.40),
        "u3": (-0.15, 0.85),
    }

    _node(ax, pos["v"], r"$v$", r=0.10, fc="#E3F2FD")
    _node(ax, pos["u1"], r"$u_1$")
    _node(ax, pos["u2"], r"$u_2$")
    _node(ax, pos["u3"], r"$u_3$")

    weights = {"u1": 0.62, "u2": 0.14, "u3": 0.24}
    for u, w in weights.items():
        p0 = np.array(pos[u])
        p1 = np.array(pos["v"])
        vdir = p1 - p0
        vdir = vdir / (np.linalg.norm(vdir) + 1e-9)
        _arrow(ax, tuple(p0 + 0.10 * vdir), tuple(p1 - 0.10 * vdir), lw=1.2 + 2.2 * w, ms=14)
        mid = 0.58 * p0 + 0.42 * p1
        ax.text(mid[0], mid[1], rf"$\alpha_{{{u}\rightarrow v}}$", fontsize=10, ha="center", va="center", color="#111111")

    ax.text(
        -0.95,
        -1.02,
        r"$\alpha_{u\rightarrow v}=\mathrm{softmax}_u(e_{u\rightarrow v})$",
        ha="left",
        va="center",
        fontsize=11,
    )
    ax.text(
        -0.95,
        -1.28,
        r"$h_v^{(\ell+1)}=\sigma\!\left(\sum_{u\in\mathcal{N}(v)}\alpha_{u\rightarrow v}\,W h_u^{(\ell)}\right)$",
        ha="left",
        va="center",
        fontsize=11,
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.4, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    png1 = os.path.join(out_dir, "gnn_message_passing.png")
    png2 = os.path.join(out_dir, "gat_attention.png")
    plot_message_passing(png1)
    plot_gat_attention(png2)

    try:
        plt.rcParams["savefig.dpi"] = int(args.dpi)
    except Exception:
        pass

    pdf1 = os.path.join(out_dir, "gnn_message_passing.pdf")
    pdf2 = os.path.join(out_dir, "gat_attention.pdf")
    plot_message_passing(pdf1)
    plot_gat_attention(pdf2)


if __name__ == "__main__":
    main()

