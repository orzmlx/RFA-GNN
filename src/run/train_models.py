"""Unified entrypoint for all model variants.

Usage:
    python train_models.py --variant <name> [other args...]

Variants:
    base         BaseLineGAT (dense graph) with counterfactual loss
    control      BaseLineGATControlContext with counterfactual loss + sparse GAT
    hybrid       BaseLineGATHybridContext with counterfactual loss
    attention    BaseLineGATContextAttention with counterfactual loss
    control_nocf BaseLineGATControlContext without counterfactual loss

All other arguments are forwarded to train_gat_run_cf_drug_loss.
"""

import importlib
import sys

from gat_entrypoint_utils import has_cli_arg, run_gat_variant

VARIANTS = {
    "base": {
        "module": "base_gnn",
        "class": "BaseLineGAT",
    },
    "control": {
        "module": "base_gnn_control_context",
        "class": "BaseLineGATControlContext",
    },
    "hybrid": {
        "module": "base_gnn_hybrid_context",
        "class": "BaseLineGATHybridContext",
    },
    "attention": {
        "module": "base_gnn_context_attention",
        "class": "BaseLineGATContextAttention",
    },
    "control_nocf": {
        "module": "base_gnn_control_context",
        "class": "BaseLineGATControlContext",
    },
}


def main():
    variant = None
    forwarded = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--variant="):
            variant = arg.split("=", 1)[1]
            i += 1
        elif arg == "--variant":
            variant = sys.argv[i + 1]
            i += 2
        else:
            forwarded.append(arg)
            i += 1

    if variant is None or variant == "":
        print("Required: --variant <name>")
        print("Available:")
        for name in VARIANTS:
            print(f"  {name}")
        sys.exit(1)

    if variant not in VARIANTS:
        print(f"Unknown variant '{variant}'. Available: {', '.join(VARIANTS.keys())}")
        sys.exit(1)

    cfg = VARIANTS[variant]
    forwarded = ["--split_modes", "warm,cold_target_pattern,cold_cell"] + forwarded

    if variant == "control" and not has_cli_arg(forwarded, "--sparse_gat"):
        forwarded.append("--sparse_gat")

    if variant == "control_nocf" and not has_cli_arg(forwarded, "--cf_lambda"):
        forwarded += ["--cf_lambda", "0.0"]

    mod = importlib.import_module(cfg["module"])
    model_cls = getattr(mod, cfg["class"])
    run_gat_variant(model_cls, "train_gat_run_cf_drug_loss", forwarded)


if __name__ == "__main__":
    main()
