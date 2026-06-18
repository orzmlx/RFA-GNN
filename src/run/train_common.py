import json
import os
import numpy as np
from scipy.stats import pearsonr


VALID_SPLIT_MODES = {"warm", "cold_drug", "cold_cell", "cold_target_pattern"}


def split_mean_logvar(pred):
    pred = np.asarray(pred)
    if pred.ndim >= 3 and pred.shape[-1] == 2:
        return pred[..., 0], pred[..., 1]
    return pred, None


def make_target_pattern_keys(drug_target_matrix):
    arr = np.asarray(drug_target_matrix)
    if arr.ndim != 2:
        raise ValueError("drug_target_matrix 必须是二维数组")
    packed = np.packbits(arr > 0, axis=1)
    return np.asarray([row.tobytes() for row in packed], dtype=object)


def build_disjoint_target_split_masks(drug_target_matrix, test_frac, seed=42):
    arr = np.asarray(drug_target_matrix)
    if arr.ndim != 2:
        raise ValueError("drug_target_matrix 必须是二维数组")
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")

    binary = arr > 0
    pattern_keys = make_target_pattern_keys(binary)
    unique_patterns, inverse = np.unique(pattern_keys, return_inverse=True)
    if len(unique_patterns) < 2:
        raise ValueError("cold_target_pattern 需要至少 2 个不同的 target patterns")

    unique_binary = np.zeros((len(unique_patterns), binary.shape[1]), dtype=bool)
    seen = np.zeros((len(unique_patterns),), dtype=bool)
    for row_idx, pat_idx in enumerate(inverse):
        if not seen[pat_idx]:
            unique_binary[pat_idx] = binary[row_idx]
            seen[pat_idx] = True

    parent = np.arange(len(unique_patterns), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    gene_owner = {}
    for pat_idx in range(len(unique_patterns)):
        for gene_idx in np.flatnonzero(unique_binary[pat_idx]):
            prev = gene_owner.get(int(gene_idx))
            if prev is None:
                gene_owner[int(gene_idx)] = pat_idx
            else:
                union(prev, pat_idx)

    component_labels = np.asarray([find(i) for i in range(len(unique_patterns))], dtype=np.int64)
    _, component_inverse = np.unique(component_labels, return_inverse=True)
    n_components = int(component_inverse.max()) + 1
    if n_components < 2:
        raise ValueError(
            "cold_target_pattern 无法切分: 所有药物 target patterns 通过共享 target 连成一个连通块"
        )

    sample_counts = np.bincount(inverse, minlength=len(unique_patterns))
    component_counts = np.bincount(component_inverse, weights=sample_counts, minlength=n_components).astype(np.int64)
    target_test_samples = max(1, int(len(binary) * test_frac))
    target_test_samples = min(target_test_samples, len(binary) - 1)

    rng = np.random.default_rng(int(seed))
    best_components = None
    best_diff = None
    for _ in range(512):
        order = rng.permutation(n_components)
        cum = np.cumsum(component_counts[order])
        candidate_positions = np.where((cum > 0) & (cum < len(binary)))[0]
        if len(candidate_positions) == 0:
            continue
        diffs = np.abs(cum[candidate_positions] - target_test_samples)
        chosen_pos = int(candidate_positions[int(np.argmin(diffs))])
        chosen = np.sort(order[: chosen_pos + 1])
        diff = int(abs(int(component_counts[chosen].sum()) - target_test_samples))
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_components = chosen
            if diff == 0:
                break

    if best_components is None or len(best_components) == 0 or len(best_components) == n_components:
        raise ValueError("cold_target_pattern 无法构造有效的 train/test 切分")

    held_component_mask = np.isin(component_inverse, best_components)
    test_mask = held_component_mask[inverse]
    train_mask = ~test_mask
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("cold_target_pattern 产生了空的 train/test 切分")

    train_targets = np.any(binary[train_mask], axis=0)
    test_targets = np.any(binary[test_mask], axis=0)
    overlap_targets = int(np.sum(train_targets & test_targets))
    train_patterns = set(unique_patterns[np.unique(inverse[train_mask])].tolist())
    test_patterns = set(unique_patterns[np.unique(inverse[test_mask])].tolist())
    stats = {
        "held_out_components": int(len(best_components)),
        "total_components": int(n_components),
        "train_samples": int(np.sum(train_mask)),
        "test_samples": int(np.sum(test_mask)),
        "train_unique_patterns": int(len(train_patterns)),
        "test_unique_patterns": int(len(test_patterns)),
        "target_overlap_count": overlap_targets,
        "pattern_overlap_count": int(len(train_patterns.intersection(test_patterns))),
    }
    return train_mask, test_mask, stats


def build_split_masks(split_mode, drug_ids, cell_idx, test_frac, seed=42, drug_target_matrix=None):
    split_mode = str(split_mode).strip()
    if split_mode not in VALID_SPLIT_MODES:
        raise ValueError(f"未知 split_mode: {split_mode}")
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError("--test_frac 需要在 (0, 1) 之间")

    rng = np.random.default_rng(int(seed))
    n = len(drug_ids)

    if split_mode == "warm":
        if n < 2:
            raise ValueError("warm 需要至少 2 个样本")
        n_test = max(1, int(n * test_frac))
        n_test = min(n_test, n - 1)
        test_idx = rng.choice(np.arange(n), size=n_test, replace=False)
        test_mask = np.zeros((n,), dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        print(f"Split=warm | Held-out samples: {int(np.sum(test_mask))}/{n}")
        return train_mask, test_mask

    if split_mode == "cold_cell":
        unique_cells = np.unique(cell_idx)
        if len(unique_cells) < 2:
            raise ValueError("cold_cell 需要至少 2 个细胞系")
        n_test = max(1, int(len(unique_cells) * test_frac))
        n_test = min(n_test, len(unique_cells) - 1)
        held_cells = rng.choice(unique_cells, n_test, replace=False)
        test_mask = np.isin(cell_idx, held_cells)
        train_mask = ~test_mask
        print(f"Split=cold_cell | Held-out cells: {len(held_cells)}/{len(unique_cells)}")
        return train_mask, test_mask

    if split_mode == "cold_drug":
        unique_drugs = np.unique(drug_ids)
        if len(unique_drugs) < 2:
            raise ValueError("cold_drug 需要至少 2 个药物")
        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_test = min(n_test, len(unique_drugs) - 1)
        held_drugs = rng.choice(unique_drugs, n_test, replace=False)
        test_mask = np.isin(drug_ids, held_drugs)
        train_mask = ~test_mask
        print(f"Split=cold_drug | Held-out drugs: {len(held_drugs)}/{len(unique_drugs)}")
        return train_mask, test_mask

    if drug_target_matrix is None:
        raise ValueError("cold_target_pattern 需要提供 drug_target_matrix")
    train_mask, test_mask, stats = build_disjoint_target_split_masks(
        drug_target_matrix,
        test_frac,
        seed=seed,
    )
    print(
        "Split=cold_target_pattern | "
        f"Held-out overlap components: {stats['held_out_components']}/{stats['total_components']} | "
        f"train_n={stats['train_samples']} test_n={stats['test_samples']} | "
        f"pattern_overlap={stats['pattern_overlap_count']} "
        f"target_overlap={stats['target_overlap_count']}"
    )
    return train_mask, test_mask


def parse_split_modes(raw, fallback):
    s = str(raw).strip()
    if s == "":
        return [str(fallback)]
    modes = [token.strip() for token in s.split(",") if token.strip() != ""]
    bad = [mode for mode in modes if mode not in VALID_SPLIT_MODES]
    if bad:
        raise ValueError(f"--split_modes 包含不支持的值: {bad}")
    seen = []
    for mode in modes:
        if mode not in seen:
            seen.append(mode)
    return seen


def append_split_suffix(path, split_mode):
    s = str(path).strip()
    if s == "":
        return ""
    suffix = f".{str(split_mode).strip()}"
    if s.endswith(".weights.h5"):
        return s[: -len(".weights.h5")] + suffix + ".weights.h5"
    stem, ext = os.path.splitext(s)
    return f"{stem}{suffix}{ext}"


def ensure_parent_dir(path):
    out_dir = os.path.dirname(str(path))
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)


def save_npz(path, **kwargs):
    payload = {}
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            try:
                payload[key] = np.asarray([json.dumps(value)], dtype=object)
            except TypeError:
                payload[key] = np.asarray([value], dtype=object)
        else:
            payload[key] = value
    ensure_parent_dir(path)
    np.savez_compressed(path, **payload)


def save_predictions_npz(
    npz_path,
    split_mode,
    y_true,
    y_pred,
    y_logvar=None,
    sample_pcc=None,
    sample_mse=None,
    drug_ids=None,
    cell_names=None,
    trt_distil_ids=None,
):
    payload = {
        "split_mode": np.asarray(str(split_mode)),
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(y_pred, dtype=np.float32),
    }
    if y_logvar is not None:
        payload["y_logvar"] = np.asarray(y_logvar, dtype=np.float32)
    if sample_pcc is not None:
        payload["sample_pcc"] = np.asarray(sample_pcc, dtype=np.float32)
    if sample_mse is not None:
        payload["sample_mse"] = np.asarray(sample_mse, dtype=np.float32)
    if drug_ids is not None:
        payload["drug_ids"] = np.asarray(drug_ids, dtype=str)
    if cell_names is not None:
        payload["cell_names"] = np.asarray(cell_names, dtype=str)
    if trt_distil_ids is not None:
        payload["trt_distil_ids"] = np.asarray(trt_distil_ids, dtype=str)
    ensure_parent_dir(npz_path)
    np.savez_compressed(npz_path, **payload)
    print(f"Saved predictions to: {npz_path}")


def samplewise_pcc(y_true, y_pred, valid_indices=None):
    if valid_indices is not None:
        y_true = y_true[:, valid_indices]
        y_pred = y_pred[:, valid_indices]
    vals = np.zeros((len(y_true),), dtype=np.float32)
    for i in range(len(y_true)):
        a = y_true[i]
        b = y_pred[i]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            vals[i] = float(pearsonr(a, b)[0])
        else:
            vals[i] = 0.0
    return vals


def samplewise_masked_metrics(y_true, y_pred, loss_mask):
    valid_indices = np.where(np.asarray(loss_mask)[0] > 0)[0]
    yt = y_true[:, valid_indices]
    yp = y_pred[:, valid_indices]
    pcc = samplewise_pcc(yt, yp)
    mse = np.mean((yt - yp) ** 2, axis=1).astype(np.float32)
    return pcc, mse
