import sys

from deepcop_target.train_deepcop_target import main


def _append_default_flag(argv, flag, value=None):
    if any(arg == flag or arg.startswith(flag + "=") for arg in argv):
        return argv
    if value is None:
        return argv + [flag]
    return argv + [flag, value]


def _append_default_split_modes(argv, value):
    if any(arg == "--split_mode" or arg.startswith("--split_mode=") or arg == "--split_modes" or arg.startswith("--split_modes=") for arg in argv):
        return argv
    return argv + ["--split_modes", value]


if __name__ == "__main__":
    argv = sys.argv[1:]
    argv = _append_default_flag(argv, "--drug_feature", "fingerprint")
    argv = _append_default_flag(argv, "--use_go_matrix")
    argv = _append_default_flag(argv, "--original_architecture")
    argv = _append_default_flag(argv, "--no-include_cell_onehot")
    argv = _append_default_flag(argv, "--pairing_mode", "multi_trt_multi_ctl")
    argv = _append_default_split_modes(argv, "warm,cold_drug,cold_cell")
    sys.argv = [sys.argv[0]] + argv
    main()
