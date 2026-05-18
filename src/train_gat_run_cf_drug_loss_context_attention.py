import sys

from gat_entrypoint_utils import inject_default_split_modes, run_gat_variant


def main():
    from base_gnn_context_attention import BaseLineGATContextAttention

    forwarded_argv = inject_default_split_modes(sys.argv[1:])
    run_gat_variant(BaseLineGATContextAttention, "train_gat_run_cf_drug_loss", forwarded_argv)


if __name__ == "__main__":
    main()
