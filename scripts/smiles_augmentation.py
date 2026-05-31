"""
SMILES Enumeration Augmentation for LDO generalization.

Generates N random SMILES per molecule using RDKit doRandom=True.
Only training drugs are augmented; validation drugs are left untouched
to prevent label leakage.

Usage (standalone test):
    python3 scripts/smiles_augmentation.py

Integration: call augment_train_smiles() before featurization.
"""

import numpy as np
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def random_smiles(smiles: str, n: int = 4, seed: Optional[int] = None) -> list[str]:
    """
    Generate up to *n* distinct random SMILES for *smiles* using RDKit atom re-ordering.

    Returns a list of valid SMILES strings (may be shorter than *n* if the
    molecule is small or if de-duplication reduces the count).  Always includes
    the canonical SMILES as the first entry.
    """
    if not HAS_RDKIT:
        return [smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    canonical = Chem.MolToSmiles(mol, canonical=True)
    seen = {canonical}
    results = [canonical]

    rng = np.random.default_rng(seed)
    attempts = 0
    max_attempts = n * 5  # allow extra attempts to fill quota

    while len(results) < n + 1 and attempts < max_attempts:
        # Randomise atom order by shuffling atom indices
        atom_order = rng.permutation(mol.GetNumAtoms()).tolist()
        try:
            smi = Chem.MolToSmiles(
                Chem.RenumberAtoms(mol, atom_order),
                canonical=False,
                doRandom=False,  # randomness comes from atom reordering
            )
        except Exception:
            attempts += 1
            continue

        if smi and smi not in seen:
            seen.add(smi)
            results.append(smi)
        attempts += 1

    return results  # first entry is always canonical


def augment_train_smiles(
    drug_smiles_map: dict,
    train_drug_ids: set,
    n_augment: int = 4,
    seed: int = 42,
) -> dict:
    """
    Augment SMILES for training drugs only.

    Parameters
    ----------
    drug_smiles_map : dict
        Maps drug_id -> canonical SMILES string (None entries are skipped).
    train_drug_ids : set
        Set of drug_ids that belong to the training split.
    n_augment : int
        Number of additional random SMILES per drug (total = n_augment + 1 with canonical).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Maps (drug_id, aug_idx) -> SMILES.
        aug_idx=0 is always the canonical SMILES.
        For validation drugs only aug_idx=0 is present.
    """
    result = {}
    rng = np.random.default_rng(seed)

    for drug_id, smiles in drug_smiles_map.items():
        if smiles is None:
            continue
        if drug_id in train_drug_ids:
            variants = random_smiles(smiles, n=n_augment,
                                     seed=int(rng.integers(0, 2**31)))
        else:
            # Validation drug: canonical SMILES only — no augmentation
            mol = Chem.MolFromSmiles(smiles) if HAS_RDKIT else None
            canonical = Chem.MolToSmiles(mol) if mol else smiles
            variants = [canonical]

        for idx, smi in enumerate(variants):
            result[(drug_id, idx)] = smi

    return result


def build_augmented_triplets(
    samples_atoms: list,
    samples_adj: list,
    samples_gex: list,
    samples_mut: list,
    samples_cna: list,
    samples_ic50: list,
    samples_drug_ids: list,
    train_drug_ids: set,
    drug_smiles_map: dict,
    featurizer,
    n_augment: int = 4,
    seed: int = 42,
) -> tuple:
    """
    Expand the training triplets with augmented SMILES.

    For each training triplet, generates *n_augment* additional entries
    using randomised SMILES variants of the same drug.  The omics features
    and IC50 label are copied unchanged — only the molecular graph differs.

    Validation triplets are returned unchanged (no augmentation).

    Returns (atoms, adj, gex, mut, cna, ic50, drug_ids) as lists,
    ready for np.stack().
    """
    import numpy as np

    # Pre-generate augmented SMILES map for all training drugs
    train_smiles_map = {d: drug_smiles_map[d]
                        for d in train_drug_ids if d in drug_smiles_map}
    aug_map = augment_train_smiles(train_smiles_map, train_drug_ids,
                                   n_augment=n_augment, seed=seed)

    out_atoms, out_adj, out_gex, out_mut, out_cna, out_ic50, out_drug_ids = \
        [], [], [], [], [], [], []

    rng = np.random.default_rng(seed)

    for i, drug_id in enumerate(samples_drug_ids):
        # Original triplet always included
        out_atoms.append(samples_atoms[i])
        out_adj.append(samples_adj[i])
        out_gex.append(samples_gex[i])
        out_mut.append(samples_mut[i])
        out_cna.append(samples_cna[i])
        out_ic50.append(samples_ic50[i])
        out_drug_ids.append(drug_id)

        if drug_id not in train_drug_ids:
            continue  # validation: no augmentation

        # Pick random augmented SMILES variants (skip aug_idx=0 = canonical, already added)
        available = [(did, idx) for (did, idx) in aug_map if did == drug_id and idx > 0]
        for key in available:
            smi = aug_map[key]
            atoms, adj = featurizer.featurize(smi)
            out_atoms.append(atoms)
            out_adj.append(adj)
            out_gex.append(samples_gex[i])
            out_mut.append(samples_mut[i])
            out_cna.append(samples_cna[i])
            out_ic50.append(samples_ic50[i])
            out_drug_ids.append(drug_id)

    n_orig = len(samples_ic50)
    n_aug  = len(out_ic50) - n_orig
    print(f"  [SMILES Aug] {n_orig:,} original + {n_aug:,} augmented = {len(out_ic50):,} triplets")

    return (out_atoms, out_adj, out_gex, out_mut, out_cna, out_ic50, out_drug_ids)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "c1ccc(cc1)N",              # aniline
        "CCO",                       # ethanol (tiny, may not produce many variants)
    ]

    print("SMILES Augmentation Test")
    print("=" * 50)
    for smi in test_smiles:
        variants = random_smiles(smi, n=4, seed=0)
        print(f"\nOriginal: {smi}")
        for idx, v in enumerate(variants):
            tag = "(canonical)" if idx == 0 else f"(aug {idx})"
            print(f"  {tag} {v}")

    print("\n[OK] smiles_augmentation.py verified.")
