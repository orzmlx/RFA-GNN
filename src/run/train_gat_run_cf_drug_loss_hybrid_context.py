import sys

from gat_entrypoint_utils import inject_default_split_modes, run_gat_variant
from base_gnn_hybrid_context import BaseLineGATHybridContext


def main():

    forwarded_argv = inject_default_split_modes(sys.argv[1:])
    run_gat_variant(BaseLineGATHybridContext, "train_gat_run_cf_drug_loss", forwarded_argv)


if __name__ == "__main__":
    main()
