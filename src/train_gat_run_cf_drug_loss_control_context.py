import sys
from base_gnn_control_context import BaseLineGATControlContext

from gat_entrypoint_utils import has_cli_arg, inject_default_split_modes, run_gat_variant


def main():

    forwarded_argv = inject_default_split_modes(sys.argv[1:])
    if not has_cli_arg(forwarded_argv, "--sparse_gat"):
        forwarded_argv.append("--sparse_gat")
    run_gat_variant(BaseLineGATControlContext, "train_gat_run_cf_drug_loss", forwarded_argv)


if __name__ == "__main__":
    main()
