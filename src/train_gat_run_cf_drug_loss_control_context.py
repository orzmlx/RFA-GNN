import os
import sys


def _inject_default_split_modes(argv):
    has_split_modes = any(arg == "--split_modes" or arg.startswith("--split_modes=") for arg in argv)
    if has_split_modes:
        return list(argv)
    return list(argv) + ["--split_modes", "warm,cold_drug,cold_cell"]


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import base_gnn
    from base_gnn_control_context import BaseLineGATControlContext
    import train_gat_run_cf_drug_loss as runner

    base_gnn.BaseLineGAT = BaseLineGATControlContext

    forwarded_argv = _inject_default_split_modes(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + forwarded_argv
        runner.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
