"""
fetch_drug_smiles.py
====================
Maps CCLE drug names → SMILES via PubChem REST API.

CCLE drug ID format: "DrugName-N" where N is a replicate/concentration index.
The suffix is stripped before querying (e.g. "Afatinib-1" → "Afatinib").

PubChem returns SMILES under the key "SMILES" (isomeric) and
"ConnectivitySMILES" (canonical) — NOT "IsomericSMILES"/"CanonicalSMILES".

Usage:
    python fetch_drug_smiles.py [--ccle_dir Dataset/ccle_broad_2019]
                                [--out Dataset/ccle_drug_smiles.csv]

Output CSV columns: drug_name, query_name, smiles, source
"""

import argparse
import os
import re
import time

import pandas as pd
import requests

# ── PubChem REST endpoint ─────────────────────────────────────────────────────
PUBCHEM_NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    "/{name}/property/SMILES,ConnectivitySMILES/JSON"
)
TIMEOUT   = 10    # seconds per request
DELAY     = 0.25  # seconds between requests (PubChem rate limit ~5 req/s)
MAX_RETRY = 1     # one retry on connection error


def _strip_replicate_suffix(name: str) -> str:
    """Remove trailing replicate index: 'Afatinib-1' → 'Afatinib', 'Drug-2' → 'Drug'.
    Only removes a pure integer suffix, not hyphens that are part of the drug name
    (e.g. 'BMS-536924' stays as-is, 'BMS-536924-1' → 'BMS-536924').
    """
    return re.sub(r'-\d+$', '', name.strip())


def _normalize(name: str) -> str:
    """Additional normalization applied after suffix stripping."""
    name = re.sub(r'\s+uM\s*$', '', name, flags=re.IGNORECASE)
    name = name.strip().rstrip('-').strip()
    return name


def _query_pubchem(name: str) -> tuple[str | None, str | None]:
    """
    Query PubChem name search for a single compound name.
    Returns (smiles, source) where source is 'isomeric' | 'canonical' | None.

    PubChem JSON response keys (verified 2026-05):
      "SMILES"             → isomeric SMILES
      "ConnectivitySMILES" → canonical / connectivity SMILES
    """
    url = PUBCHEM_NAME_URL.format(name=requests.utils.quote(name))
    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                iso = props.get("SMILES")
                can = props.get("ConnectivitySMILES")
                if iso:
                    return iso, "isomeric"
                if can:
                    return can, "canonical"
                return None, None
            elif resp.status_code == 404:
                return None, None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < MAX_RETRY:
                time.sleep(DELAY * 2)
    return None, None


def fetch_smiles(ccle_name: str) -> tuple[str | None, str | None, str]:
    """
    Try up to three query variants for a CCLE drug name:
      1. Stripped name (replicate suffix removed)
      2. Further normalized name (uM suffix, trailing hyphens)
      3. Name with spaces inserted before uppercase runs (CamelCase split)

    Returns (smiles, source, query_name_used).
    """
    stripped  = _strip_replicate_suffix(ccle_name)
    normalized = _normalize(stripped)

    # CamelCase → spaced (e.g. "AKTinhibitorVIII" → "AKT inhibitor VIII")
    camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized)
    camel_split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', camel_split).strip()

    candidates = []
    if stripped:
        candidates.append(stripped)
    if normalized != stripped:
        candidates.append(normalized)
    if camel_split not in candidates and camel_split != stripped:
        candidates.append(camel_split)

    for query in candidates:
        smiles, source = _query_pubchem(query)
        time.sleep(DELAY)
        if smiles is not None:
            return smiles, source, query

    return None, None, stripped


def load_drug_names(ccle_dir: str) -> list[str]:
    """Extract drug names from CCLE IC50 file index column."""
    ic50_path = os.path.join(ccle_dir, "data_drug_treatment_ic50.txt")
    if not os.path.exists(ic50_path):
        raise FileNotFoundError(f"IC50 file not found: {ic50_path}")
    df_full = pd.read_csv(ic50_path, sep="\t", index_col=0, usecols=[0])
    drug_names = df_full.index.tolist()
    print(f"[fetch_drug_smiles] {len(drug_names)} drug entries in IC50 file.")
    return drug_names


def main(ccle_dir: str = "Dataset/ccle_broad_2019",
         out_path: str = "Dataset/ccle_drug_smiles.csv"):

    drug_names = load_drug_names(ccle_dir)
    n_total    = len(drug_names)

    results  = []
    n_mapped = 0

    for i, name in enumerate(drug_names, 1):
        smiles, source, query_used = fetch_smiles(name)

        results.append({
            "drug_name":  name,
            "query_name": query_used,
            "smiles":     smiles,
            "source":     source,
        })

        if smiles is not None:
            n_mapped += 1
            status = f"OK  ({source})  [{query_used}]"
        else:
            status = f"MISS  [{query_used}]"

        print(f"  [{i:3d}/{n_total}] {name:<40s} {status}")

    df_out = pd.DataFrame(results)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Résumé : {n_mapped}/{n_total} drogues mappées ({100*n_mapped/n_total:.1f}%)")
    print(f"Fichier sauvegardé : {out_path}")

    missed = df_out[df_out["smiles"].isna()]["drug_name"].tolist()
    if missed:
        print(f"\nDrogues non trouvées ({len(missed)}) :")
        for m in missed:
            print(f"  - {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map CCLE drug names to SMILES via PubChem.")
    parser.add_argument("--ccle_dir", default="Dataset/ccle_broad_2019")
    parser.add_argument("--out", default="Dataset/ccle_drug_smiles.csv")
    args = parser.parse_args()
    main(ccle_dir=args.ccle_dir, out_path=args.out)
