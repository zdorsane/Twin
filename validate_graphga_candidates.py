#!/usr/bin/env python3
import csv
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from smiles_sanitizer import sanitize_smiles

# Supprimer les warnings RDKit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from rdkit.Chem import RDConfig
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SASCORER_AVAILABLE = True
except Exception:
    SASCORER_AVAILABLE = False
    print("[WARN] SA scorer not available — SA will be None.")


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

CANDIDATES = []


def load_candidates_from_file(filename="graphga_population.txt"):
    """Load candidates from generated population file."""
    candidates = []
    try:
        with open(filename, "r") as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    candidates.append(smiles)
    except FileNotFoundError:
        print(f"Warning: {filename} not found, using default candidates")
        # Fallback to hardcoded list if file doesn't exist
        candidates = [
            "CCON1NCCOCNN1OC(=O)ON(C)C",
            "CCN1COCCOCCOCCCCCCOCCNN1OC(=O)O",
            "CCOC(=O)OON1NCOCOCCNN1ON(C)C",
            "CCCON1NCCCCCONNCOCNN1OC(=O)OCC",
            "CCNOC(=O)ON1NCOCCCCOCN1CC",
            "CN(C)ON1NCCOCOCNN1OC(=O)O",
            "CCOC(=O)ON1NCCOCOCCCCCOCCCN1CC",
            "CCNOC(=O)ON1NCCOCCOCCCOCCCOCN1CC",
            "CN(C)CN1CCCOCCOCCCCCOCCNN1OC(=O)ON(C)C",
            "CCN1CCCOCCOCCNN1OC(=O)O",
        ]
    return candidates


def load_top_candidates_from_csv(filename="graphga_ranked_population.csv", top_n=10):
    """Load top candidates from ranked CSV file."""
    candidates = []
    try:
        import csv
        with open(filename, "r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= top_n:
                    break
                smiles = row.get("smiles", "").strip()
                if smiles:
                    candidates.append(smiles)
    except (FileNotFoundError, KeyError):
        print(f"Warning: Could not load from {filename}, falling back to population file")
        return load_candidates_from_file()
    return candidates


def synthetic_accessibility_score(mol) -> float | None:
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        sa = sascorer.calculateScore(mol)
        sa_norm = (10 - sa) / 9   # normalise 1→10 en 0→1 (1.0 = facile à synthétiser)
        return float(sa_norm)
    except Exception:
        return None  # ne pas mettre 0 silencieusement


def is_pains(mol) -> bool:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    return catalog.HasMatch(mol)


def logp_penalty(logp: float) -> float:
    return max(0.0, abs(logp) - 2.0) * 0.15


def chem_quality(mol) -> tuple[float, float | None, float, float | None]:
    qed = float(QED.qed(mol))
    sa = synthetic_accessibility_score(mol)
    logp = float(Crippen.MolLogP(mol))
    penalty = logp_penalty(logp)
    composite = qed + (sa if sa is not None else 0) - penalty if sa is not None else None
    return qed, sa, logp, composite


def validate(smiles: str) -> dict:
    clean_smi = sanitize_smiles(smiles)
    if clean_smi is None:
        return {
            "smiles": smiles,
            "valid": False,
            "canonical": None,
            "qed": None,
            "sa": None,
            "mw": None,
            "logp": None,
            "composite": None,
            "pains": None,
        }
    
    mol = Chem.MolFromSmiles(clean_smi)  # Already sanitized
    canonical = clean_smi  # Already canonical
    qed_value, sa_value, logp_value, composite_value = chem_quality(mol)
    pains_value = is_pains(mol)
    return {
        "smiles": smiles,
        "valid": True,
        "canonical": canonical,
        "qed": qed_value,
        "sa": sa_value,
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(logp_value),
        "composite": composite_value,
        "pains": pains_value,
    }


def analyser_candidats(smiles_list):
    for i, smi in enumerate(smiles_list, 1):   # parcourt la liste (index commence à 1)
        clean_smi = sanitize_smiles(smi)
        if clean_smi:
            mol = Chem.MolFromSmiles(clean_smi)
            print(f"{i:02d}. QED={QED.qed(mol):.3f} | "
                  f"MW={Descriptors.MolWt(mol):.1f} | "
                  f"LogP={Crippen.MolLogP(mol):.2f} | "
                  f"{clean_smi}")
        else:
            print(f"{i:02d}. ❌ Invalid: {smi[:40]}...")


def save_results(results, filename="graphga_validated_candidates.csv"):
    fieldnames = ["rank", "smiles", "canonical", "valid", "qed", "sa", "mw", "logp", "composite", "pains"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, res in enumerate(results, 1):
            row = {**res, "rank": idx}
            writer.writerow(row)


def save_top_candidates(results, filename="graphga_top_candidates.csv", top_n=5):
    fieldnames = ["rank", "smiles", "canonical", "qed", "sa", "mw", "logp", "composite", "pains"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, res in enumerate(results[:top_n], 1):
            row = {
                "rank": idx,
                "smiles": res["smiles"],
                "canonical": res["canonical"],
                "qed": res["qed"],
                "sa": res["sa"],
                "mw": res["mw"],
                "logp": res["logp"],
                "composite": res["composite"],
                "pains": res["pains"],
            }
            writer.writerow(row)


def main():
    print("Validating GraphGA top candidates:\n")
    
    # Load candidates from generated files
    CANDIDATES = load_top_candidates_from_csv("graphga_ranked_population.csv", top_n=10)
    
    if not CANDIDATES:
        print("No candidates found to validate!")
        return
    
    results = [validate(smiles) for smiles in CANDIDATES]

    filtered = [res for res in results if res["valid"] and res["composite"] is not None and not res["pains"]]
    filtered.sort(key=lambda x: x["composite"], reverse=True)

    # Use analyser_candidats for clean output of valid molecules
    valid_smiles = [res["smiles"] for res in filtered]
    analyser_candidats(valid_smiles)

    save_results(results)
    print(f"Saved full validation results to graphga_validated_candidates.csv")

    if filtered:
        save_top_candidates(filtered, filename="graphga_top_candidates.csv", top_n=len(filtered))
        print(f"Saved filtered top candidates to graphga_top_candidates.csv")
        print("Best filtered candidates (sorted by composite quality):")
        for idx, res in enumerate(filtered, 1):
            sa_str = f"{res['sa']:.3f}" if res['sa'] is not None else "N/A"
            pains_str = "Yes" if res['pains'] else "No"
            print(
                f"{idx:02d}. {res['canonical']} | composite={res['composite']:.3f} | "
                f"QED={res['qed']:.3f} | SA={sa_str} | MW={res['mw']:.1f} | logP={res['logp']:.3f} | PAINS={pains_str}"
            )
    else:
        print("No valid candidates available for composite scoring.")


if __name__ == '__main__':
    main()
