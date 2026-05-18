import argparse
import json
import os
import sys
import types

import numpy as np


def resolve_root(root):
    candidate = str(root).strip()
    if os.path.exists(candidate):
        return candidate
    for fallback in ["/local/data1/liume102/rfa", "/local/data1/liume102/src", "/Users/liuxi/Desktop/RFA_GNN"]:
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError("No valid root directory found")


def _normalise_cell_lines(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() in {"ALL", "NONE", "NULL"}:
        return None
    return s


def _pattern_key(row):
    row = np.asarray(row) > 0
    if row.ndim != 1:
        raise ValueError("Target row must be 1D")
    packed = np.packbits(row.astype(np.uint8))
    return packed.tobytes().hex()


def _target_names(row, target_genes):
    idx = np.flatnonzero(np.asarray(row) > 0)
    return [str(target_genes[i]) for i in idx.tolist()]


def _collect_unique_drug_patterns(anchor_drug_ids, anchor_X_drug, target_genes, anchor_mask):
    out = {}
    for i in np.where(np.asarray(anchor_mask, dtype=bool))[0].tolist():
        drug_id = str(anchor_drug_ids[i])
        row = np.asarray(anchor_X_drug[i], dtype=np.float32)
        key = _pattern_key(row)
        payload = out.get(drug_id)
        if payload is None:
            out[drug_id] = {
                "drug_id": drug_id,
                "pattern_key": key,
                "target_genes": _target_names(row, target_genes),
                "anchor_count": 1,
            }
        else:
            payload["anchor_count"] += 1
            if payload["pattern_key"] != key:
                raise ValueError(f"Drug {drug_id} has inconsistent target patterns across anchors")
    return out


def _invert_pattern_map(drug_payloads):
    out = {}
    for drug_id, payload in sorted(drug_payloads.items()):
        out.setdefault(payload["pattern_key"], []).append(drug_id)
    return out


def _shared_targets(train_payloads, test_payloads):
    train_targets = set()
    test_targets = set()
    for payload in train_payloads.values():
        train_targets.update(payload["target_genes"])
    for payload in test_payloads.values():
        test_targets.update(payload["target_genes"])
    return sorted(train_targets & test_targets)


def _cross_split_target_hits(train_payloads, test_payloads):
    train_target_sets = {
        drug_id: set(payload["target_genes"]) for drug_id, payload in train_payloads.items()
    }
    rows = []
    for test_drug, test_payload in sorted(test_payloads.items()):
        test_targets = set(test_payload["target_genes"])
        shared_with = []
        for train_drug, train_targets in sorted(train_target_sets.items()):
            overlap = sorted(test_targets & train_targets)
            if overlap:
                shared_with.append(
                    {
                        "train_drug_id": train_drug,
                        "shared_target_genes": overlap,
                    }
                )
        rows.append(
            {
                "test_drug_id": test_drug,
                "test_target_genes": sorted(test_targets),
                "shared_with_train": shared_with,
            }
        )
    return rows


def _print_drug_section(title, payloads):
    print(f"\n===== {title} =====")
    print(f"Unique drugs: {len(payloads)}")
    for drug_id in sorted(payloads):
        payload = payloads[drug_id]
        targets = payload["target_genes"]
        target_text = ",".join(targets) if targets else "<NO_TARGET>"
        print(
            f"{drug_id}\tanchors={payload['anchor_count']}\t"
            f"pattern={payload['pattern_key']}\ttargets={target_text}"
        )


def _print_pattern_section(title, pattern_map, payloads):
    print(f"\n===== {title} =====")
    print(f"Unique patterns: {len(pattern_map)}")
    for pattern_key in sorted(pattern_map):
        drugs = pattern_map[pattern_key]
        target_text = ",".join(payloads[drugs[0]]["target_genes"]) if drugs else "<NO_TARGET>"
        print(f"{pattern_key}\tdrugs={','.join(drugs)}\ttargets={target_text}")


def _print_overlap_section(shared_targets, shared_patterns, cross_hits):
    print("\n===== Overlap Summary =====")
    print(f"Shared target genes count: {len(shared_targets)}")
    if shared_targets:
        print("Shared target genes:")
        for gene in shared_targets:
            print(gene)
    print(f"Shared target patterns count: {len(shared_patterns)}")
    if shared_patterns:
        print("Shared target patterns:")
        for pattern_key in shared_patterns:
            print(pattern_key)

    print("\n===== Test Drugs Sharing Targets With Train =====")
    any_hits = False
    for row in cross_hits:
        if not row["shared_with_train"]:
            continue
        any_hits = True
        print(f"TEST {row['test_drug_id']} targets={','.join(row['test_target_genes'])}")
        for hit in row["shared_with_train"]:
            print(
                f"  TRAIN {hit['train_drug_id']} "
                f"shared={','.join(hit['shared_target_genes'])}"
            )
    if not any_hits:
        print("No cross-split target overlap detected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/liuxi/Desktop/RFA_GNN")
    parser.add_argument("--cell_line", default="ALL")
    parser.add_argument("--use_landmark_genes", action="store_true", default=True)
    parser.add_argument("--ctl_pair_k", type=int, default=3)
    parser.add_argument(
        "--pairing_mode",
        choices=["multi_trt_multi_ctl", "unique_trt_reuse_ctl", "unique_trt_unique_ctl"],
        default="multi_trt_multi_ctl",
    )
    parser.add_argument(
        "--split_mode",
        choices=["warm", "cold_drug", "cold_cell", "cold_target_pattern"],
        default="cold_target_pattern",
    )
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_json", default="")
    args = parser.parse_args()

    root = resolve_root(args.root)
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = types.ModuleType("tensorflow")

    from data_loader import load_rfa_data, build_pair_split_masks

    ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.h5")
    trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.h5")
    drug_target_path = os.path.join(root, "data/compound_targets.txt")
    siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

    data = load_rfa_data(
        ctl_path,
        trt_path,
        drug_target_path=drug_target_path,
        landmark_path=landmark_path,
        siginfo_path=siginfo_path,
        fingerprint_path=fingerprint_path,
        use_landmark_genes=bool(args.use_landmark_genes),
        full_gene_path=full_gene_path,
        cell_lines=_normalise_cell_lines(args.cell_line),
        ctl_residual_pool_size=int(args.ctl_pair_k),
        pairing_mode=str(args.pairing_mode),
    )

    anchor_drug_ids = np.asarray(data["anchor_drug_ids"], dtype=str)
    anchor_cell_names = np.asarray(data["anchor_cell_names"], dtype=str)
    anchor_X_drug = np.asarray(data["anchor_X_drug"], dtype=np.float32)
    target_genes = np.asarray(data["target_genes"], dtype=object)

    train_mask, test_mask = build_pair_split_masks(
        split_mode=str(args.split_mode),
        drug_ids=anchor_drug_ids,
        cell_names=anchor_cell_names,
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        drug_target_matrix=anchor_X_drug,
    )

    train_payloads = _collect_unique_drug_patterns(anchor_drug_ids, anchor_X_drug, target_genes, train_mask)
    test_payloads = _collect_unique_drug_patterns(anchor_drug_ids, anchor_X_drug, target_genes, test_mask)
    train_pattern_map = _invert_pattern_map(train_payloads)
    test_pattern_map = _invert_pattern_map(test_payloads)
    shared_targets = _shared_targets(train_payloads, test_payloads)
    shared_patterns = sorted(set(train_pattern_map) & set(test_pattern_map))
    cross_hits = _cross_split_target_hits(train_payloads, test_payloads)

    print(f"Split mode: {args.split_mode}")
    print(f"Train anchors: {int(np.sum(train_mask))}")
    print(f"Test anchors: {int(np.sum(test_mask))}")
    _print_drug_section("Train Drug IDs", train_payloads)
    _print_drug_section("Test Drug IDs", test_payloads)
    _print_pattern_section("Train Target Patterns", train_pattern_map, train_payloads)
    _print_pattern_section("Test Target Patterns", test_pattern_map, test_payloads)
    _print_overlap_section(shared_targets, shared_patterns, cross_hits)

    if str(args.save_json).strip() != "":
        out_path = os.path.abspath(str(args.save_json))
        out_dir = os.path.dirname(out_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "split_mode": str(args.split_mode),
            "train_anchor_count": int(np.sum(train_mask)),
            "test_anchor_count": int(np.sum(test_mask)),
            "train_unique_drugs": sorted(train_payloads.values(), key=lambda x: x["drug_id"]),
            "test_unique_drugs": sorted(test_payloads.values(), key=lambda x: x["drug_id"]),
            "train_target_patterns": [
                {
                    "pattern_key": key,
                    "drug_ids": sorted(drugs),
                    "target_genes": train_payloads[drugs[0]]["target_genes"] if drugs else [],
                }
                for key, drugs in sorted(train_pattern_map.items())
            ],
            "test_target_patterns": [
                {
                    "pattern_key": key,
                    "drug_ids": sorted(drugs),
                    "target_genes": test_payloads[drugs[0]]["target_genes"] if drugs else [],
                }
                for key, drugs in sorted(test_pattern_map.items())
            ],
            "shared_target_genes": shared_targets,
            "shared_target_patterns": shared_patterns,
            "cross_split_target_hits": cross_hits,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
