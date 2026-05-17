"""
fetch_drug_smiles.py
====================
Maps CCLE drug names → SMILES via PubChem REST API.

Usage:
    python fetch_drug_smiles.py [--ccle_dir Dataset/ccle_broad_2019]
                                [--out Dataset/ccle_drug_smiles.csv]

Output CSV columns: drug_name, smiles, source
"""

import argparse
import os
import re
import time

import pandas as pd
import requests

# ── PubChem REST endpoint ─────────────────────────────────────────────────────
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    "/{name}/property/CanonicalSMILES,IsomericSMILES/JSON"
)
TIMEOUT    = 10   # seconds per request
DELAY      = 0.25 # seconds between requests (PubChem rate limit)
MAX_RETRY  = 1    # one retry on failure


def _clean_name(name: str) -> str:
    """Light normalisation: strip trailing hyphens/spaces, remove 'uM' suffix."""
    name = name.strip()
    name = re.sub(r'\s+uM\s*$', '', name, flags=re.IGNORECASE)
    name = name.rstrip('-').strip()
    return name


def _query_pubchem(name: str) -> tuple[str | None, str | None]:
    """
    Query PubChem for a single name.
    Returns (smiles, source) where source is 'isomeric' | 'canonical' | None.
    Retries once on connection error / timeout.
    """
    url = PUBCHEM_URL.format(name=requests.utils.quote(name))
    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                iso = props.get("IsomericSMILES")
                can = props.get("CanonicalSMILES")
                if iso:
                    return iso, "isomeric"
                if can:
                    return can, "canonical"
                return None, None
            elif resp.status_code == 404:
                return None, None  # not found — no point retrying
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < MAX_RETRY:
                time.sleep(DELAY * 2)
    return None, None


def fetch_smiles(drug_name: str) -> tuple[str | None, str | None]:
    """
    Try original name first, then cleaned name if the original fails.
    Returns (smiles, source).
    """
    smiles, source = _query_pubchem(drug_name)
    if smiles is not None:
        return smiles, source

    cleaned = _clean_name(drug_name)
    if cleaned != drug_name:
        time.sleep(DELAY)
        smiles, source = _query_pubchem(cleaned)
        if smiles is not None:
            return smiles, source

    return None, None


def load_drug_names(ccle_dir: str) -> list[str]:
    """
    Reads data_drug_treatment_ic50.txt and extracts drug names (index column).
    The file is tab-separated; rows = drugs, columns = cell lines.
    """
    ic50_path = os.path.join(ccle_dir, "data_drug_treatment_ic50.txt")
    if not os.path.exists(ic50_path):
        raise FileNotFoundError(f"IC50 file not found: {ic50_path}")

    df = pd.read_csv(ic50_path, sep="\t", index_col=0, nrows=0)  # only header
    # Reload to get all index values
    df_full = pd.read_csv(ic50_path, sep="\t", index_col=0, usecols=[0])
    drug_names = df_full.index.tolist()
    print(f"[fetch_drug_smiles] Found {len(drug_names)} drug names in IC50 file.")
    return drug_names


def main(ccle_dir: str = "Dataset/ccle_broad_2019",
         out_path: str = "Dataset/ccle_drug_smiles.csv"):

    drug_names = load_drug_names(ccle_dir)
    n_total = len(drug_names)

    results = []
    n_mapped = 0

    for i, name in enumerate(drug_names, 1):
        smiles, source = fetch_smiles(name)
        results.append({"drug_name": name, "smiles": smiles, "source": source})

        if smiles is not None:
            n_mapped += 1
            status = f"OK  ({source})"
        else:
            status = "MISS"

        print(f"  [{i:3d}/{n_total}] {name:<40s} {status}")
        time.sleep(DELAY)  # respect PubChem rate limit after every successful call

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Résumé : {n_mapped}/{n_total} drogues mappées avec succès")
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
