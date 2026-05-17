"""
baseline_models.py
==================
Classical baselines for IC50 prediction on CCLE data.

Models:
  - Ridge regression (ECFP4 + omics)
  - Ridge regression (omics only)
  - Random Forest (ECFP4 + omics)

Splits:
  A — Random 85/15
  B — Leave-Drug-Out (20% least frequent drugs → val)
  C — Leave-Cell-Out (20% of cell lines → val)

Outputs:
  Dataset/baseline_results.csv

Usage:
    python baseline_models.py [--ccle_dir Dataset/ccle_broad_2019]
                               [--smiles_csv Dataset/ccle_drug_smiles.csv]
                               [--out Dataset/baseline_results.csv]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit not available — ECFP4 fingerprints will be zeros.")

# ── Constants matching fullPipeline.py HP ─────────────────────────────────────
GEX_DIM = 978
CNV_DIM = 426
ECFP_BITS = 1024
ECFP_RADIUS = 2
VAL_FRAC = 0.20
RANDOM_SEED = 42


# ─── 1. ECFP4 fingerprint ─────────────────────────────────────────────────────

def smiles_to_ecfp4(smiles: str) -> np.ndarray:
    """Returns ECFP4 bit vector (1024 dims) or zeros on failure."""
    fp = np.zeros(ECFP_BITS, dtype=np.float32)
    if not HAS_RDKIT or not isinstance(smiles, str):
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_BITS)
    fp = np.array(bitvec, dtype=np.float32)
    return fp


# ─── 2. Data loading ──────────────────────────────────────────────────────────

def load_data(ccle_dir: str, smiles_csv: str):
    """
    Loads and aligns CCLE IC50, GEx, CNA, and SMILES data.
    Returns a dict with all arrays and metadata.
    """
    ic50_path = os.path.join(ccle_dir, "data_drug_treatment_ic50.txt")
    gex_path  = os.path.join(ccle_dir, "data_mrna_seq_rpkm.txt")
    cna_path  = os.path.join(ccle_dir, "data_cna.txt")

    for p in [ic50_path, gex_path, cna_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    print("[baseline] Loading IC50...")
    ic50_df = pd.read_csv(ic50_path, sep="\t", index_col=0)
    meta_cols = [c for c in ic50_df.columns if ic50_df[c].dtype == object]
    ic50_df = ic50_df.drop(columns=meta_cols, errors="ignore")
    ic50_df = ic50_df.apply(pd.to_numeric, errors="coerce")
    drug_ids   = list(ic50_df.index)
    cell_lines = list(ic50_df.columns)
    print(f"  IC50: {len(drug_ids)} drugs × {len(cell_lines)} cell lines")

    print("[baseline] Loading GEx...")
    gex_df = pd.read_csv(gex_path, sep="\t", index_col=0)
    gex_df = gex_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print("[baseline] Loading CNA...")
    cna_df = pd.read_csv(cna_path, sep="\t", index_col=0)
    cna_df = cna_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Common cell lines across all three matrices
    common_cells = list(
        set(cell_lines) & set(gex_df.columns) & set(cna_df.columns)
    )
    print(f"  Common cell lines (IC50 ∩ GEx ∩ CNA): {len(common_cells)}")

    # GEx: top GEX_DIM genes by variance
    gex_sub = gex_df[common_cells].T
    top_gex = gex_sub.var(axis=0).sort_values(ascending=False).index[:GEX_DIM].tolist()
    gex_mat = gex_sub[top_gex].values.astype(np.float32)          # (cells, 978)
    scaler_gex = StandardScaler()
    gex_mat = scaler_gex.fit_transform(gex_mat).astype(np.float32)

    # CNA: top CNV_DIM genes by variance
    cna_sub = cna_df[common_cells].T
    top_cna = cna_sub.var(axis=0).sort_values(ascending=False).index[:CNV_DIM].tolist()
    cna_mat = cna_sub[top_cna].values.astype(np.float32)           # (cells, 426)
    scaler_cna = StandardScaler()
    cna_mat = scaler_cna.fit_transform(cna_mat).astype(np.float32)

    omics_mat = np.concatenate([gex_mat, cna_mat], axis=1)         # (cells, 1404)
    print(f"  Omics matrix: {omics_mat.shape}")

    # Load SMILES
    smiles_map = {}
    if os.path.exists(smiles_csv):
        df_smi = pd.read_csv(smiles_csv)
        for _, row in df_smi.iterrows():
            if pd.notna(row.get("smiles")):
                smiles_map[row["drug_name"]] = row["smiles"]
        print(f"  SMILES loaded: {len(smiles_map)} drugs mapped")
    else:
        print(f"  [WARN] {smiles_csv} not found — all drugs will be excluded from ECFP4 runs.")

    # Filter drugs that have SMILES
    drugs_with_smiles = [d for d in drug_ids if d in smiles_map]
    n_excluded = len(drug_ids) - len(drugs_with_smiles)
    print(f"  Drugs excluded (no SMILES): {n_excluded} / {len(drug_ids)}")

    # Build triplets
    cell_to_idx = {c: i for i, c in enumerate(common_cells)}

    drug_indices, cell_indices, ic50_vals = [], [], []

    for drug_id in drugs_with_smiles:
        for cell in common_cells:
            if cell not in ic50_df.columns:
                continue
            val = ic50_df.loc[drug_id, cell]
            if pd.isna(val):
                continue
            ic50_log = np.log1p(max(float(val), 0.001))
            drug_indices.append(drug_id)
            cell_indices.append(cell_to_idx[cell])
            ic50_vals.append(ic50_log)

    drug_indices = np.array(drug_indices)
    cell_indices = np.array(cell_indices, dtype=np.int32)
    ic50_arr     = np.array(ic50_vals, dtype=np.float32)

    # Z-score IC50 (same as fullPipeline)
    ic50_mean = ic50_arr.mean()
    ic50_std  = ic50_arr.std() + 1e-6
    ic50_z    = (ic50_arr - ic50_mean) / ic50_std

    print(f"  Total valid triplets: {len(ic50_z):,}")

    return {
        "drug_indices": drug_indices,      # array of drug name strings
        "cell_indices": cell_indices,      # int indices into common_cells
        "ic50_z": ic50_z,
        "ic50_mean": ic50_mean,
        "ic50_std": ic50_std,
        "omics_mat": omics_mat,            # (n_cells, 1404)
        "smiles_map": smiles_map,
        "drugs_with_smiles": drugs_with_smiles,
        "common_cells": common_cells,
    }


# ─── 3. Feature construction ──────────────────────────────────────────────────

def build_features(data: dict, use_ecfp: bool = True):
    """
    Builds the full feature matrix X and target y.
    X = [ECFP4 (1024) | omics (1404)] if use_ecfp else [omics (1404)]
    """
    drug_indices = data["drug_indices"]
    cell_indices = data["cell_indices"]
    omics_mat    = data["omics_mat"]
    smiles_map   = data["smiles_map"]
    y            = data["ic50_z"]

    print(f"[baseline] Building features (ECFP4={use_ecfp})...")
    n = len(drug_indices)
    omics_feats = omics_mat[cell_indices]  # (n, 1404)

    if use_ecfp:
        fps = np.stack([smiles_to_ecfp4(smiles_map.get(d, "")) for d in drug_indices])
        X = np.concatenate([fps, omics_feats], axis=1).astype(np.float32)
    else:
        X = omics_feats.astype(np.float32)

    print(f"  Feature matrix: {X.shape}")
    return X, y


# ─── 4. Splits ────────────────────────────────────────────────────────────────

def split_random(drug_indices, cell_indices, y, val_frac=VAL_FRAC, seed=RANDOM_SEED):
    """Split A: random 80/20."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int((1 - val_frac) * n)
    return idx[:split], idx[split:]


def split_leave_drug_out(drug_indices, cell_indices, y, val_frac=VAL_FRAC):
    """
    Split B: last val_frac of sorted drug names go to val.
    All triplets involving those drugs → val set.
    """
    unique_drugs = sorted(set(drug_indices))
    n_val_drugs  = max(1, int(len(unique_drugs) * val_frac))
    train_drugs  = set(unique_drugs[:-n_val_drugs])
    val_drugs    = set(unique_drugs[-n_val_drugs:])

    train_idx = np.where(np.isin(drug_indices, list(train_drugs)))[0]
    val_idx   = np.where(np.isin(drug_indices, list(val_drugs)))[0]

    print(f"  Leave-Drug-Out: {len(train_drugs)} train drugs | {len(val_drugs)} val drugs")
    return train_idx, val_idx


def split_leave_cell_out(drug_indices, cell_indices, y, val_frac=VAL_FRAC, seed=RANDOM_SEED):
    """
    Split C: last val_frac of sorted cell indices go to val.
    All triplets involving those cell lines → val set.
    """
    unique_cells = sorted(set(cell_indices))
    n_val_cells  = max(1, int(len(unique_cells) * val_frac))
    train_cells  = set(unique_cells[:-n_val_cells])
    val_cells    = set(unique_cells[-n_val_cells:])

    train_idx = np.where(np.isin(cell_indices, list(train_cells)))[0]
    val_idx   = np.where(np.isin(cell_indices, list(val_cells)))[0]

    print(f"  Leave-Cell-Out: {len(train_cells)} train cells | {len(val_cells)} val cells")
    return train_idx, val_idx


# ─── 5. Evaluation ────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {"rmse": rmse, "pearson_r": float(pr), "spearman_r": float(sr)}


# ─── 6. Main training loop ────────────────────────────────────────────────────

def run_baselines(ccle_dir: str, smiles_csv: str, out_path: str):
    data = load_data(ccle_dir, smiles_csv)

    drug_indices = data["drug_indices"]
    cell_indices = data["cell_indices"]
    y            = data["ic50_z"]

    # Pre-compute feature matrices (two variants)
    X_full,  y = build_features(data, use_ecfp=True)   # ECFP4 + omics
    X_omics, _ = build_features(data, use_ecfp=False)  # omics only

    # Define splits
    splits = {
        "Random":         split_random(drug_indices, cell_indices, y),
        "Leave-Drug-Out": split_leave_drug_out(drug_indices, cell_indices, y),
        "Leave-Cell-Out": split_leave_cell_out(drug_indices, cell_indices, y),
    }

    # Define models
    def make_models():
        return {
            "Ridge (ECFP4+omics)": (Ridge(alpha=1.0), X_full),
            "Ridge (omics only)":  (Ridge(alpha=1.0), X_omics),
            "RF (ECFP4+omics)":    (
                RandomForestRegressor(
                    n_estimators=100, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED
                ),
                X_full,
            ),
        }

    rows = []

    for split_name, (train_idx, val_idx) in splits.items():
        print(f"\n{'='*60}")
        print(f"Split: {split_name}  (train={len(train_idx):,} | val={len(val_idx):,})")
        models = make_models()

        for model_name, (model, X) in models.items():
            print(f"  Fitting {model_name}...", end=" ", flush=True)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            metrics = evaluate(y[val_idx], y_pred)
            print(f"RMSE={metrics['rmse']:.4f}  Pearson={metrics['pearson_r']:.4f}  Spearman={metrics['spearman_r']:.4f}")
            rows.append({
                "Model":      model_name,
                "Split":      split_name,
                "RMSE":       round(metrics["rmse"],       4),
                "Pearson_r":  round(metrics["pearson_r"],  4),
                "Spearman_r": round(metrics["spearman_r"], 4),
            })

    # Summary table
    df_results = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("RÉSULTATS COMPLETS")
    print(df_results.to_string(index=False))

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    df_results.to_csv(out_path, index=False)
    print(f"\nRésultats sauvegardés : {out_path}")

    # Bi-Int comparison (random split, ECFP4+omics Ridge vs reported Bi-Int)
    biint_rmse = 0.472
    random_ridge_rows = df_results[
        (df_results["Split"] == "Random") & (df_results["Model"] == "Ridge (ECFP4+omics)")
    ]
    if not random_ridge_rows.empty:
        ridge_rmse = random_ridge_rows.iloc[0]["RMSE"]
        delta_pct  = (biint_rmse - ridge_rmse) / biint_rmse * 100
        print(f"\n{'─'*60}")
        print(f"Bi-Int  (random split, omics only)        : RMSE = {biint_rmse:.3f}")
        print(f"Baseline Ridge (random split, ECFP4+omics): RMSE = {ridge_rmse:.3f}")
        if delta_pct > 0:
            print(f"→ Bi-Int apporte +{delta_pct:.1f}% vs baseline Ridge")
        else:
            print(f"→ Bi-Int n'apporte pas d'amélioration ({abs(delta_pct):.1f}% inférieur au baseline)")
        print(f"{'─'*60}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCLE baseline models (Ridge + RF + ECFP4)")
    parser.add_argument("--ccle_dir",   default="Dataset/ccle_broad_2019")
    parser.add_argument("--smiles_csv", default="Dataset/ccle_drug_smiles.csv")
    parser.add_argument("--out",        default="Dataset/baseline_results.csv")
    args = parser.parse_args()

    run_baselines(
        ccle_dir   = args.ccle_dir,
        smiles_csv = args.smiles_csv,
        out_path   = args.out,
    )
