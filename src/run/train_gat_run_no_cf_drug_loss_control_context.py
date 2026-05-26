import sys

from gat_entrypoint_utils import inject_default_args, run_gat_variant


def main():
    from base_gnn_control_context import BaseLineGATControlContext

    forwarded_argv = inject_default_args(sys.argv[1:], cf_lambda=0.0)
    run_gat_variant(BaseLineGATControlContext, "train_gat_run_cf_drug_loss", forwarded_argv)


if __name__ == "__main__":
    main()
