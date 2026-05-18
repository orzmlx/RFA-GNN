import os
import sys


DEFAULT_SPLIT_MODES = "warm,cold_drug,cold_cell"


def has_cli_arg(argv, name):
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def inject_default_split_modes(argv, default_split_modes=DEFAULT_SPLIT_MODES):
    forwarded = list(argv)
    if has_cli_arg(forwarded, "--split_modes") or has_cli_arg(forwarded, "--split_mode"):
        return forwarded
    return forwarded + ["--split_modes", default_split_modes]


def inject_default_args(argv, default_split_modes=DEFAULT_SPLIT_MODES, cf_lambda=None):
    forwarded = inject_default_split_modes(argv, default_split_modes=default_split_modes)
    if cf_lambda is not None and not has_cli_arg(forwarded, "--cf_lambda"):
        forwarded += ["--cf_lambda", str(cf_lambda)]
    return forwarded


def run_gat_variant(base_model_cls, runner_module_name, forwarded_argv):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import base_gnn

    runner = __import__(runner_module_name)
    base_gnn.BaseLineGAT = base_model_cls
    runner.BaseLineGAT = base_model_cls

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + list(forwarded_argv)
        runner.main()
    finally:
        sys.argv = old_argv
