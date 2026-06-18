import math
from typing import Dict, Tuple

import numpy as np


def _dataset_length(dataset: Dict) -> int:
    if "X_ctl" not in dataset:
        raise KeyError("paired dataset 缺少 X_ctl，无法判断样本数")
    return int(len(dataset["X_ctl"]))


def _subset_paired_dataset(dataset: Dict, keep_idx: np.ndarray) -> Dict:
    n = _dataset_length(dataset)
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    out = {}
    for key, value in dataset.items():
        if value is None:
            out[key] = None
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == n:
            out[key] = value[keep_idx]
            continue
        if isinstance(value, list) and len(value) == n:
            out[key] = [value[i] for i in keep_idx.tolist()]
            continue
        out[key] = value
    return out


def parse_budget_spec(budget):
    if budget is None:
        return ("full", 1.0)
    if isinstance(budget, (int, float)):
        frac = float(budget)
        if frac <= 0.0:
            return ("zero_shot", 0.0)
        if frac >= 1.0:
            return ("full", 1.0)
        return ("fraction", frac)

    s = str(budget).strip().lower()
    if s in {"", "full", "all", "100%"}:
        return ("full", 1.0)
    if s in {"zero", "zero_shot", "zero-shot", "0", "0%"}:
        return ("zero_shot", 0.0)
    if s in {"one", "one_shot", "one-shot", "1shot", "1-shot"}:
        return ("one_shot", 1.0)
    if s.endswith("%"):
        frac = float(s[:-1]) / 100.0
        if frac <= 0.0:
            return ("zero_shot", 0.0)
        if frac >= 1.0:
            return ("full", 1.0)
        return ("fraction", frac)
    frac = float(s)
    if frac <= 0.0:
        return ("zero_shot", 0.0)
    if frac >= 1.0:
        return ("full", 1.0)
    return ("fraction", frac)


def build_pair_budget_mask(
    drug_ids,
    cell_names,
    budget,
    seed=42,
    min_keep_positive_budget=True,
):
    drug_ids = np.asarray(drug_ids, dtype=str)
    cell_names = np.asarray(cell_names, dtype=str)
    if len(drug_ids) != len(cell_names):
        raise ValueError("drug_ids 和 cell_names 长度不一致")

    mode, value = parse_budget_spec(budget)
    n = len(drug_ids)
    if mode == "full":
        return np.ones((n,), dtype=bool), {
            "budget_mode": mode,
            "budget_value": 1.0,
            "kept_n": int(n),
            "total_n": int(n),
            "pairs_total": int(len(np.unique(np.char.add(drug_ids, "||" + cell_names)))),
            "pairs_kept": int(len(np.unique(np.char.add(drug_ids, "||" + cell_names)))),
        }
    if mode == "zero_shot":
        return np.zeros((n,), dtype=bool), {
            "budget_mode": mode,
            "budget_value": 0.0,
            "kept_n": 0,
            "total_n": int(n),
            "pairs_total": int(len(np.unique(np.char.add(drug_ids, "||" + cell_names)))),
            "pairs_kept": 0,
        }

    rng = np.random.default_rng(int(seed))
    pair_keys = np.asarray([f"{d}||{c}" for d, c in zip(drug_ids.tolist(), cell_names.tolist())], dtype=object)
    keep_mask = np.zeros((n,), dtype=bool)
    unique_pairs = np.asarray(sorted(set(pair_keys.tolist())), dtype=object)

    for pair in unique_pairs.tolist():
        idx = np.where(pair_keys == pair)[0]
        if len(idx) == 0:
            continue
        if mode == "one_shot":
            keep_n = 1
        else:
            keep_n = int(math.ceil(len(idx) * float(value)))
            if min_keep_positive_budget and float(value) > 0.0:
                keep_n = max(1, keep_n)
        keep_n = min(keep_n, len(idx))
        chosen = rng.choice(idx, size=keep_n, replace=False)
        keep_mask[chosen] = True

    kept_pairs = pair_keys[keep_mask]
    meta = {
        "budget_mode": mode,
        "budget_value": float(value),
        "kept_n": int(np.sum(keep_mask)),
        "total_n": int(n),
        "pairs_total": int(len(unique_pairs)),
        "pairs_kept": int(len(np.unique(kept_pairs))) if len(kept_pairs) > 0 else 0,
    }
    return keep_mask, meta


def apply_pair_budget_to_dataset(
    train_data: Dict,
    budget,
    seed=42,
    min_keep_positive_budget=True,
) -> Tuple[Dict, Dict]:
    if "drug_ids" not in train_data or "cell_names" not in train_data:
        raise KeyError("train_data 必须包含 drug_ids 和 cell_names 才能做 pair budget split")

    keep_mask, meta = build_pair_budget_mask(
        drug_ids=train_data["drug_ids"],
        cell_names=train_data["cell_names"],
        budget=budget,
        seed=seed,
        min_keep_positive_budget=min_keep_positive_budget,
    )
    keep_idx = np.where(keep_mask)[0]
    budget_train = _subset_paired_dataset(train_data, keep_idx)
    meta["dropped_n"] = int(len(keep_mask) - np.sum(keep_mask))
    return budget_train, meta


def build_budget_split_data(
    data,
    split_mode,
    test_frac,
    budget,
    seed=42,
    budget_seed=None,
    train_pairing_mode="multi_trt_multi_ctl",
    train_ctl_pair_k=3,
    test_pairing_mode="unique_trt_reuse_ctl",
    min_keep_positive_budget=True,
    base_split_fn=None,
):
    # Accept the original split function explicitly so wrapper-based patching does not recurse.
    if base_split_fn is None:
        # Lazy import keeps this module usable for budget-mask unit tests without heavy deps.
        from data_loader import prepare_split_data as base_split_fn

    train_data, test_data, train_anchor_mask, test_anchor_mask = base_split_fn(
        data=data,
        split_mode=split_mode,
        test_frac=test_frac,
        seed=seed,
        train_pairing_mode=train_pairing_mode,
        train_ctl_pair_k=train_ctl_pair_k,
        test_pairing_mode=test_pairing_mode,
    )
    budget_seed = int(seed) if budget_seed is None else int(budget_seed)
    budget_train, budget_meta = apply_pair_budget_to_dataset(
        train_data=train_data,
        budget=budget,
        seed=budget_seed,
        min_keep_positive_budget=min_keep_positive_budget,
    )
    meta = {
        "split_mode": str(split_mode),
        "budget": budget,
        "base_train_n": int(_dataset_length(train_data)),
        "budget_train_n": int(_dataset_length(budget_train)),
        "test_n": int(_dataset_length(test_data)),
        "train_anchor_n": int(np.sum(train_anchor_mask)),
        "test_anchor_n": int(np.sum(test_anchor_mask)),
    }
    meta.update(budget_meta)
    return budget_train, test_data, train_anchor_mask, test_anchor_mask, meta


def summarize_budget_split(meta: Dict) -> str:
    mode = meta.get("budget_mode", "unknown")
    if mode == "one_shot":
        budget_desc = "one_shot"
    elif mode == "fraction":
        budget_desc = f"{100.0 * float(meta.get('budget_value', 0.0)):.1f}%"
    else:
        budget_desc = mode
    return (
        f"split={meta.get('split_mode', 'unknown')} "
        f"budget={budget_desc} "
        f"train={meta.get('budget_train_n', meta.get('kept_n', 0))}/{meta.get('base_train_n', meta.get('total_n', 0))} "
        f"test={meta.get('test_n', 0)} "
        f"pairs={meta.get('pairs_kept', 0)}/{meta.get('pairs_total', 0)}"
    )
