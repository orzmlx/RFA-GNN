import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests


def _norm_symbol(s: str) -> str:
    return str(s).strip().upper()


def _norm_inchikey(k: str) -> str:
    k = str(k).strip().upper()
    if k == "" or k == "NAN":
        return ""
    return k


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def _affinity_to_um(value: float, units: str) -> Optional[float]:
    u = str(units).strip()
    if u == "":
        return None
    u = u.replace("µ", "u")
    scale = {
        "pM": 1e-6,
        "nM": 1e-3,
        "uM": 1.0,
        "mM": 1e3,
        "M": 1e6,
    }.get(u)
    if scale is None:
        return None
    return float(value) * float(scale)


@dataclass
class CompoundRow:
    pert_id: str
    inchi_key: str
    canonical_smiles: str


class HttpClient:
    def __init__(self, base_headers: Optional[Dict[str, str]] = None, sleep_s: float = 0.0, timeout_s: float = 30.0):
        self.session = requests.Session()
        self.base_headers = dict(base_headers or {})
        self.sleep_s = float(sleep_s)
        self.timeout_s = float(timeout_s)

    def get_json(self, url: str, params: Optional[dict] = None) -> dict:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        r = self.session.get(url, params=params, headers=self.base_headers, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def post_json(self, url: str, data: dict) -> dict:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        r = self.session.post(url, data=data, headers=self.base_headers, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


def load_compound_table(compound_targets_in: str) -> List[CompoundRow]:
    df = pd.read_csv(compound_targets_in, sep="\t", dtype=str, low_memory=False)
    cols = set(df.columns)
    if "pert_id" not in cols:
        raise ValueError("compound_targets_in 缺少 pert_id 列")
    if "inchi_key" not in cols and "inchi_key" not in cols and "inchi_key" not in cols:
        pass
    inchi_col = "inchi_key" if "inchi_key" in cols else ("inchi_key" if "inchi_key" in cols else None)
    if inchi_col is None:
        inchi_col = "inchi_key" if "InChIKey" in cols else None
    if inchi_col is None:
        raise ValueError("compound_targets_in 缺少 inchi_key/InChIKey 列")
    smiles_col = "canonical_smiles" if "canonical_smiles" in cols else ("smiles" if "smiles" in cols else "")

    rows = []
    for _, r in df.iterrows():
        pid = str(r.get("pert_id", "")).strip()
        if pid == "" or pid.lower() == "nan":
            continue
        ik = _norm_inchikey(r.get(inchi_col, ""))
        smi = str(r.get(smiles_col, "")).strip() if smiles_col else ""
        rows.append(CompoundRow(pert_id=pid, inchi_key=ik, canonical_smiles=smi))
    return rows


def load_uniprot_gene_mapping(mapping_path: str) -> Dict[str, str]:
    if not mapping_path or not os.path.exists(mapping_path):
        return {}
    import gzip

    opener = gzip.open if mapping_path.endswith(".gz") else open
    mapping = {}
    with opener(mapping_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            uniprot_id, id_type, value = parts
            if id_type in ("Gene_Name", "Gene_Name (primary)", "Gene_Name_primary"):
                if uniprot_id and value and uniprot_id not in mapping:
                    mapping[uniprot_id] = value
    return mapping


def map_uniprot_to_symbol_mygene(client: HttpClient, uniprot_ids: Sequence[str]) -> Dict[str, str]:
    ids = [str(x).strip() for x in uniprot_ids if str(x).strip() != ""]
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}
    url = "https://mygene.info/v3/querymany"
    mapping = {}
    batch = 1000
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        payload = {"ids": ",".join(chunk), "scopes": "uniprot", "fields": "symbol", "species": "human"}
        try:
            resp = client.post_json(url, payload)
        except Exception:
            continue
        if isinstance(resp, list):
            for hit in resp:
                q = hit.get("query")
                s = hit.get("symbol")
                if q and s:
                    mapping[str(q).strip()] = _norm_symbol(s)
    return mapping


def chembl_find_molecule_ids_by_inchikey(client: HttpClient, inchi_key: str) -> List[str]:
    ik = _norm_inchikey(inchi_key)
    if not ik:
        return []
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    params = {"molecule_structures__standard_inchi_key": ik, "limit": 1000, "offset": 0}
    out = []
    while True:
        js = client.get_json(url, params=params)
        mos = js.get("molecules") or []
        for m in mos:
            mid = m.get("molecule_chembl_id")
            if mid:
                out.append(str(mid))
        meta = js.get("page_meta") or {}
        if meta.get("next") is None:
            break
        params["offset"] = int(params["offset"]) + int(params["limit"])
    return list(dict.fromkeys(out))


def chembl_fetch_activities(
    client: HttpClient,
    molecule_chembl_id: str,
    types: Sequence[str],
    max_um: float,
    assay_types: Optional[Set[str]] = None,
    organism: str = "Homo sapiens",
    limit: int = 1000,
) -> List[dict]:
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {"molecule_chembl_id": str(molecule_chembl_id), "limit": int(limit), "offset": 0}
    acts = []
    types_set = {str(t).upper() for t in types}
    while True:
        js = client.get_json(url, params=params)
        rows = js.get("activities") or []
        for a in rows:
            if organism and str(a.get("target_organism") or "").strip() != organism:
                continue
            if assay_types is not None and len(assay_types) > 0:
                at = str(a.get("assay_type") or "").strip()
                if at not in assay_types:
                    continue
            st = str(a.get("standard_type") or "").strip().upper()
            if st not in types_set:
                continue
            rel = str(a.get("standard_relation") or "").strip()
            if rel not in {"=", "<", "<=", "~"}:
                continue
            val = _to_float(a.get("standard_value"))
            if val is None:
                continue
            um = _affinity_to_um(val, a.get("standard_units") or "")
            if um is None:
                continue
            if float(um) > float(max_um):
                continue
            tid = a.get("target_chembl_id")
            if not tid:
                continue
            acts.append(
                {
                    "target_chembl_id": str(tid),
                    "standard_type": st,
                    "standard_value_um": float(um),
                }
            )
        meta = js.get("page_meta") or {}
        if meta.get("next") is None:
            break
        params["offset"] = int(params["offset"]) + int(params["limit"])
    return acts


def chembl_target_to_uniprot_accessions(client: HttpClient, target_chembl_id: str) -> List[str]:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_chembl_id}.json"
    js = client.get_json(url)
    comps = js.get("target_components") or []
    accs = []
    for c in comps:
        acc = c.get("accession")
        if acc:
            accs.append(str(acc).strip())
    return list(dict.fromkeys([a for a in accs if a]))


def build_targets_from_chembl(
    compounds: List[CompoundRow],
    threshold_um: float,
    types: Sequence[str],
    assay_types: Sequence[str],
    sleep_s: float,
    uniprot_mapping_path: str,
    max_drugs: int,
    verbose_every: int,
) -> Dict[str, Set[str]]:
    client = HttpClient(sleep_s=sleep_s)
    uniprot_local = load_uniprot_gene_mapping(uniprot_mapping_path)

    pert2mids: Dict[str, List[str]] = {}
    all_mids: Set[str] = set()
    for i, c in enumerate(compounds):
        if max_drugs > 0 and i >= max_drugs:
            break
        if not c.inchi_key:
            continue
        mids = chembl_find_molecule_ids_by_inchikey(client, c.inchi_key)
        if mids:
            pert2mids[c.pert_id] = mids
            all_mids.update(mids)
        if verbose_every > 0 and (i + 1) % verbose_every == 0:
            print(f"ChEMBL molecule lookup: {i+1}/{min(len(compounds), max_drugs or len(compounds))} | mapped={len(pert2mids)}")

    assay_types_set = set([x.strip() for x in assay_types if str(x).strip() != ""])
    mid2acts: Dict[str, List[dict]] = {}
    target_ids: Set[str] = set()
    for j, mid in enumerate(sorted(all_mids)):
        acts = chembl_fetch_activities(
            client,
            molecule_chembl_id=mid,
            types=types,
            max_um=float(threshold_um),
            assay_types=(assay_types_set if assay_types_set else None),
            organism="Homo sapiens",
        )
        mid2acts[mid] = acts
        for a in acts:
            target_ids.add(a["target_chembl_id"])
        if verbose_every > 0 and (j + 1) % verbose_every == 0:
            print(f"ChEMBL activity: {j+1}/{len(all_mids)} | targets={len(target_ids)}")

    tid2acc: Dict[str, List[str]] = {}
    all_accs: Set[str] = set()
    for k, tid in enumerate(sorted(target_ids)):
        accs = chembl_target_to_uniprot_accessions(client, tid)
        tid2acc[tid] = accs
        all_accs.update(accs)
        if verbose_every > 0 and (k + 1) % verbose_every == 0:
            print(f"ChEMBL target map: {k+1}/{len(target_ids)} | accs={len(all_accs)}")

    acc2sym = {}
    if uniprot_local:
        for a in all_accs:
            s = uniprot_local.get(a)
            if s:
                acc2sym[a] = _norm_symbol(s)
    missing = [a for a in all_accs if a not in acc2sym]
    if missing:
        acc2sym.update(map_uniprot_to_symbol_mygene(client, missing))

    pert2symbols: Dict[str, Set[str]] = {}
    for pid, mids in pert2mids.items():
        syms: Set[str] = set()
        for mid in mids:
            for a in mid2acts.get(mid, []):
                accs = tid2acc.get(a["target_chembl_id"], [])
                for acc in accs:
                    s = acc2sym.get(acc)
                    if s:
                        syms.add(s)
        if syms:
            pert2symbols[pid] = syms
    return pert2symbols


def parse_bindingdb(
    bindingdb_path: str,
    threshold_um: float,
    inchi_key_to_pert: Dict[str, str],
    max_rows: int,
) -> Dict[str, Set[str]]:
    if not bindingdb_path or not os.path.exists(bindingdb_path):
        return {}

    usecols = None
    df = pd.read_csv(bindingdb_path, sep="\t", dtype=str, low_memory=False, nrows=(None if max_rows <= 0 else int(max_rows)))
    cols = {c.lower(): c for c in df.columns}

    def col(name_candidates: Sequence[str]) -> Optional[str]:
        for n in name_candidates:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    inchi_col = col(["InChIKey", "InChI Key", "Ligand InChIKey", "InChIKey (Ligand)"])
    target_gene_col = col(["Target Gene Name", "Target Gene Symbol", "Gene Name", "Gene Symbol"])
    organism_col = col(["Target Source Organism", "Target Organism", "Organism"])

    meas_col = col(["Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)", "AC50 (nM)"])
    if inchi_col is None or target_gene_col is None:
        return {}

    pert2sym: Dict[str, Set[str]] = {}
    for _, r in df.iterrows():
        ik = _norm_inchikey(r.get(inchi_col, ""))
        if not ik:
            continue
        pid = inchi_key_to_pert.get(ik)
        if not pid:
            continue
        if organism_col is not None:
            org = str(r.get(organism_col, "")).strip()
            if org and org not in {"Homo sapiens", "Human"}:
                continue
        val_um = None
        if meas_col is not None:
            v = _to_float(r.get(meas_col))
            if v is not None:
                val_um = float(v) * 1e-3
        if val_um is not None and float(val_um) > float(threshold_um):
            continue
        sym = _norm_symbol(r.get(target_gene_col, ""))
        if sym:
            pert2sym.setdefault(pid, set()).add(sym)
    return pert2sym


def write_out(
    out_path: str,
    compounds: List[CompoundRow],
    chembl_targets: Dict[str, Set[str]],
    bindingdb_targets: Dict[str, Set[str]],
):
    rows = []
    for c in compounds:
        pid = c.pert_id
        a = chembl_targets.get(pid, set())
        b = bindingdb_targets.get(pid, set())
        merged = sorted(set(a) | set(b))
        rows.append(
            {
                "pert_id": pid,
                "inchi_key": c.inchi_key,
                "canonical_smiles": c.canonical_smiles,
                "targets_chembl": ",".join(sorted(a)),
                "targets_bindingdb": ",".join(sorted(b)),
                "target": ",".join(merged),
                "n_targets_chembl": int(len(a)),
                "n_targets_bindingdb": int(len(b)),
                "n_targets_total": int(len(merged)),
            }
        )
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, sep="\t", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compound_targets_in", default="data/compound_targets.txt")
    p.add_argument("--out", default="data/compound_targets_public.tsv")
    p.add_argument("--max_drugs", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.05)
    p.add_argument("--verbose_every", type=int, default=200)

    p.add_argument("--chembl", action="store_true", default=True)
    p.add_argument("--chembl_threshold_um", type=float, default=10.0)
    p.add_argument("--chembl_types", default="KI,KD,IC50,EC50,AC50,POTENCY")
    p.add_argument("--chembl_assay_types", default="B,F")
    p.add_argument("--uniprot_mapping_path", default="")

    p.add_argument("--bindingdb_path", default="")
    p.add_argument("--bindingdb_threshold_um", type=float, default=10.0)
    p.add_argument("--bindingdb_max_rows", type=int, default=0)
    args = p.parse_args()

    compounds = load_compound_table(args.compound_targets_in)
    if int(args.max_drugs) > 0:
        compounds = compounds[: int(args.max_drugs)]

    chembl_targets: Dict[str, Set[str]] = {}
    if bool(args.chembl):
        chembl_targets = build_targets_from_chembl(
            compounds=compounds,
            threshold_um=float(args.chembl_threshold_um),
            types=[x.strip() for x in str(args.chembl_types).split(",") if x.strip()],
            assay_types=[x.strip() for x in str(args.chembl_assay_types).split(",") if x.strip()],
            sleep_s=float(args.sleep_s),
            uniprot_mapping_path=str(args.uniprot_mapping_path),
            max_drugs=int(args.max_drugs),
            verbose_every=int(args.verbose_every),
        )

    inchi_key_to_pert = {c.inchi_key: c.pert_id for c in compounds if c.inchi_key}
    bindingdb_targets = parse_bindingdb(
        bindingdb_path=str(args.bindingdb_path),
        threshold_um=float(args.bindingdb_threshold_um),
        inchi_key_to_pert=inchi_key_to_pert,
        max_rows=int(args.bindingdb_max_rows),
    )

    out_path = str(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_out(out_path, compounds, chembl_targets, bindingdb_targets)
    print(f"Saved: {out_path}")
    print(f"ChemBL mapped drugs: {len(chembl_targets)}")
    print(f"BindingDB mapped drugs: {len(bindingdb_targets)}")


if __name__ == "__main__":
    main()

