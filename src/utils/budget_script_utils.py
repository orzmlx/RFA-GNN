import argparse
import importlib
import os
import sys
from contextlib import contextmanager

from data_budget_split import build_budget_split_data, summarize_budget_split


DEFAULT_BUDGETS = "one_shot,20%,30%,50%,80%"


def parse_budgets(raw):
    text = str(raw).strip()
    if text == "":
        return ["one_shot", "20%", "30%", "50%", "80%"]
    budgets = [token.strip() for token in text.split(",") if token.strip() != ""]
    out = []
    for b in budgets:
        low = b.lower()
        if low in {"zero", "zero_shot", "zero-shot", "0", "0%"}:
            continue
        if low not in out:
            out.append(low if low == "one_shot" else b)
    return out


def budget_token(budget):
    s = str(budget).strip().lower()
    if s in {"one", "one_shot", "one-shot", "1shot", "1-shot"}:
        return "one_shot"
    return s.replace("%", "pct").replace(".", "p").replace("-", "_")


def append_budget_suffix(path, budget):
    s = str(path).strip()
    if s == "":
        return ""
    stem, ext = os.path.splitext(s)
    return f"{stem}.{budget_token(budget)}{ext}"


def rewrite_output_args(argv, budget):
    path_flags = {
        "--save_json",
        "--save_pred_prefix",
        "--save_meta_json",
        "--save_eval_npz",
        "--save_gat_weights",
    }
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in path_flags and i + 1 < len(argv):
            out.extend([token, append_budget_suffix(argv[i + 1], budget)])
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def extract_budget_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--budget_seed", type=int, default=None)
    parser.add_argument("--min_keep_positive_budget", action=argparse.BooleanOptionalAction, default=True)
    args, remaining = parser.parse_known_args(argv)
    budgets = parse_budgets(args.budgets)
    if len(budgets) == 0:
        budgets = ["one_shot", "20%", "30%", "50%", "80%"]
    return args, budgets, remaining


def make_budgeted_split_fn(budget, budget_seed):
    def _budgeted_split(
        data,
        split_mode,
        test_frac,
        seed=42,
        train_pairing_mode="multi_trt_multi_ctl",
        train_ctl_pair_k=3,
        test_pairing_mode="unique_trt_reuse_ctl",
    ):
        train_data, test_data, train_anchor_mask, test_anchor_mask, meta = build_budget_split_data(
            data=data,
            split_mode=split_mode,
            test_frac=test_frac,
            budget=budget,
            seed=seed,
            budget_seed=budget_seed,
            train_pairing_mode=train_pairing_mode,
            train_ctl_pair_k=train_ctl_pair_k,
            test_pairing_mode=test_pairing_mode,
            min_keep_positive_budget=True,
        )
        print("[budget]", summarize_budget_split(meta))
        return train_data, test_data, train_anchor_mask, test_anchor_mask

    return _budgeted_split


@contextmanager
def patched_split(module_name, budget, budget_seed):
    module = importlib.import_module(module_name)
    original = getattr(module, "prepare_split_data")
    setattr(module, "prepare_split_data", make_budgeted_split_fn(budget=budget, budget_seed=budget_seed))
    try:
        yield module
    finally:
        setattr(module, "prepare_split_data", original)


def run_budgeted_main(module_name, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    budget_args, budgets, passthrough = extract_budget_args(argv)
    exit_code = 0
    for budget in budgets:
        run_argv = rewrite_output_args(passthrough, budget)
        print(f"\n===== Budget run: {budget} =====")
        with patched_split(module_name, budget=budget, budget_seed=budget_args.budget_seed):
            old_argv = sys.argv[:]
            try:
                sys.argv = [module_name] + run_argv
                importlib.import_module(module_name).main()
            except SystemExit as exc:
                code = 0 if exc.code is None else int(exc.code)
                exit_code = max(exit_code, code)
            finally:
                sys.argv = old_argv
    return exit_code
