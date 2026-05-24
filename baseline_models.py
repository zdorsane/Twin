"""
baseline_models.py
==================
Classical baselines for IC50 prediction on CCLE data.

Models:
  - Ridge regression (ECFP4 + omics)
  - Ridge regression (omics only)
  - Random Forest (ECFP4 + omics)
  - XGBoost (ECFP4 + omics)  [skipped if xgboost not installed]
  - MLP (ECFP4 + omics)

Feature sets:
  - ECFP4 (2048) + GEx (978) + CNA (426) + mutations (735)  = 4187 dims
  - omics only: GEx (978) + CNA (426) + mutations (735)      = 2139 dims

Splits:
  A — Random 80/20
  B — Leave-Drug-Out (20% of drugs → val)
  C — Leave-Cell-Out (20% of cell lines → val)

Outputs:
  Dataset/baseline_results.csv

Usage:
    python baseline_models.py [--ccle_dir Dataset/ccle_broad_2019]
                               [--smiles_csv Dataset/ccle_drug_smiles.csv]
                               [--mut_path  Dataset/ccle_broad_2019/data_mutations_extended.txt]
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
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — XGBoost baseline will be skipped.")

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
GEX_DIM   = 978
CNV_DIM   = 426
MUT_DIM   = 735
ECFP_BITS = 2048   # match fullPipeline Morgan FP nBits
ECFP_RADIUS = 2
VAL_FRAC  = 0.20
RANDOM_SEED = 42

# Bi-Int best result to include in comparison table
_BIINT_RESULTS = [
    {"Model": "Bi-Int (GNN+QuatVAE+4×BiInt)", "Split": "Random",
     "RMSE": 0.4633, "R2": None, "Pearson_r": 0.8840, "Spearman_r": None},
    {"Model": "Bi-Int (GNN+QuatVAE+4×BiInt)", "Split": "Leave-Drug-Out",
     "RMSE": None,   "R2": None, "Pearson_r": -0.129,  "Spearman_r": None},
]


# ─── 1. ECFP4 fingerprint ─────────────────────────────────────────────────────

def smiles_to_ecfp4(smiles: str) -> np.ndarray:
    """Returns ECFP4 bit vector (ECFP_BITS dims) or zeros on failure."""
    fp = np.zeros(ECFP_BITS, dtype=np.float32)
    if not HAS_RDKIT or not isinstance(smiles, str):
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_BITS)
    return np.array(bitvec, dtype=np.float32)


# ─── 2. Data loading ──────────────────────────────────────────────────────────

def load_data(ccle_dir: str, smiles_csv: str, mut_path: str = ""):
    """
    Loads and aligns CCLE IC50, GEx, CNA, mutations, and SMILES data.
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

    # Deterministic common cell line ordering
    common_cells = sorted(set(cell_lines) & set(gex_df.columns) & set(cna_df.columns))
    print(f"  Common cell lines (IC50 ∩ GEx ∩ CNA): {len(common_cells)}")

    # GEx: top GEX_DIM genes by variance
    gex_sub = gex_df[common_cells].T
    top_gex = gex_sub.var(axis=0).sort_values(ascending=False).index[:GEX_DIM].tolist()
    gex_mat = StandardScaler().fit_transform(
        gex_sub[top_gex].values.astype(np.float32)
    ).astype(np.float32)  # (cells, 978)

    # CNA: top CNV_DIM genes by variance
    cna_sub = cna_df[common_cells].T
    top_cna = cna_sub.var(axis=0).sort_values(ascending=False).index[:CNV_DIM].tolist()
    cna_mat = StandardScaler().fit_transform(
        cna_sub[top_cna].values.astype(np.float32)
    ).astype(np.float32)  # (cells, 426)

    # Mutations: binary per-gene matrix (cells, MUT_DIM)
    mut_mat = _load_mutations(mut_path, common_cells)

    omics_mat = np.concatenate([gex_mat, cna_mat, mut_mat], axis=1)  # (cells, GEX+CNA+MUT)
    print(f"  Omics matrix (GEx+CNA+mut): {omics_mat.shape}")

    # Load SMILES
    smiles_map = {}
    if os.path.exists(smiles_csv):
        df_smi = pd.read_csv(smiles_csv)
        for _, row in df_smi.iterrows():
            if pd.notna(row.get("smiles")):
                smiles_map[row["drug_name"]] = row["smiles"]
        print(f"  SMILES loaded: {len(smiles_map)} drugs mapped")
    else:
        print(f"  [WARN] {smiles_csv} not found — ECFP4 columns will be zeros.")

    # Filter drugs that have SMILES
    drugs_with_smiles = [d for d in drug_ids if d in smiles_map]
    print(f"  Drugs excluded (no SMILES): {len(drug_ids)-len(drugs_with_smiles)} / {len(drug_ids)}")

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
            drug_indices.append(drug_id)
            cell_indices.append(cell_to_idx[cell])
            ic50_vals.append(np.log1p(max(float(val), 0.001)))

    drug_indices = np.array(drug_indices)
    cell_indices = np.array(cell_indices, dtype=np.int32)
    ic50_arr     = np.array(ic50_vals, dtype=np.float32)

    # Z-score IC50 — same normalisation as fullPipeline
    ic50_mean = float(ic50_arr.mean())
    ic50_std  = float(ic50_arr.std()) + 1e-6
    ic50_z    = (ic50_arr - ic50_mean) / ic50_std

    print(f"  Total valid triplets: {len(ic50_z):,}")

    return {
        "drug_indices":      drug_indices,
        "cell_indices":      cell_indices,
        "ic50_z":            ic50_z,
        "ic50_mean":         ic50_mean,
        "ic50_std":          ic50_std,
        "omics_mat":         omics_mat,
        "smiles_map":        smiles_map,
        "drugs_with_smiles": drugs_with_smiles,
        "common_cells":      common_cells,
    }


def _load_mutations(mut_path: str, common_cells: list) -> np.ndarray:
    """
    Build a binary mutation matrix (n_cells, MUT_DIM).
    Uses data_mutations_extended.txt if available; otherwise all-zeros.
    Top MUT_DIM genes by mutation frequency are selected.
    """
    n_cells = len(common_cells)
    zeros   = np.zeros((n_cells, MUT_DIM), dtype=np.float32)

    candidates = [
        mut_path,
        "Dataset/ccle_broad_2019/data_mutations_extended.txt",
        "Dataset/ccle_broad_2019/data_mutations.txt",
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)

    if path is None:
        print("  [WARN] Mutations file not found — mutation features will be zeros.")
        return zeros

    print(f"[baseline] Loading mutations from {path}...")
    try:
        mut_df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as e:
        print(f"  [WARN] Could not read mutations file: {e}")
        return zeros

    if "Tumor_Sample_Barcode" not in mut_df.columns or "Hugo_Symbol" not in mut_df.columns:
        print("  [WARN] Missing required columns in mutations file — zeros used.")
        return zeros

    # Binary pivot: cell × gene
    mut_df = mut_df[["Tumor_Sample_Barcode", "Hugo_Symbol"]].drop_duplicates()
    mut_df["value"] = 1
    pivot = mut_df.pivot_table(
        index="Tumor_Sample_Barcode", columns="Hugo_Symbol",
        values="value", aggfunc="max", fill_value=0
    )

    # Align cells — use Tumor_Sample_Barcode directly (full CELLNAME_TISSUE format)
    covered = [c for c in common_cells if c in pivot.index]
    print(f"  Mutations: {len(covered)}/{n_cells} cells covered")

    # Select top MUT_DIM genes by mutation frequency
    gene_freq = pivot.sum(axis=0).sort_values(ascending=False)
    top_genes = gene_freq.index[:MUT_DIM].tolist()
    pivot_sub = pivot.reindex(columns=top_genes, fill_value=0)

    mat = np.zeros((n_cells, MUT_DIM), dtype=np.float32)
    cell_to_row = {c: i for i, c in enumerate(common_cells)}
    for cell in covered:
        mat[cell_to_row[cell]] = pivot_sub.loc[cell].values.astype(np.float32)

    return mat


# ─── 3. Feature construction ──────────────────────────────────────────────────

def build_features(data: dict, use_ecfp: bool = True):
    """
    X = [ECFP4(2048) | GEx(978) | CNA(426) | mut(735)]  if use_ecfp
    X = [GEx(978) | CNA(426) | mut(735)]                  otherwise
    """
    drug_indices = data["drug_indices"]
    cell_indices = data["cell_indices"]
    omics_mat    = data["omics_mat"]
    smiles_map   = data["smiles_map"]
    y            = data["ic50_z"]

    omics_feats = omics_mat[cell_indices]

    if use_ecfp:
        fps = np.stack([smiles_to_ecfp4(smiles_map.get(d, "")) for d in drug_indices])
        X = np.concatenate([fps, omics_feats], axis=1).astype(np.float32)
    else:
        X = omics_feats.astype(np.float32)

    print(f"  Feature matrix (ECFP4={use_ecfp}): {X.shape}")
    return X, y


# ─── 4. Splits ────────────────────────────────────────────────────────────────

def split_random(drug_indices, cell_indices, y, val_frac=VAL_FRAC, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int((1 - val_frac) * len(y))
    return idx[:split], idx[split:]


def split_leave_drug_out(drug_indices, cell_indices, y, val_frac=VAL_FRAC):
    unique_drugs = sorted(set(drug_indices))
    n_val = max(1, int(len(unique_drugs) * val_frac))
    val_drugs = set(unique_drugs[-n_val:])
    train_idx = np.where(~np.isin(drug_indices, list(val_drugs)))[0]
    val_idx   = np.where( np.isin(drug_indices, list(val_drugs)))[0]
    print(f"  Leave-Drug-Out: {len(unique_drugs)-n_val} train | {n_val} val drugs")
    return train_idx, val_idx


def split_leave_cell_out(drug_indices, cell_indices, y, val_frac=VAL_FRAC):
    unique_cells = sorted(set(cell_indices))
    n_val = max(1, int(len(unique_cells) * val_frac))
    val_cells = set(unique_cells[-n_val:])
    train_idx = np.where(~np.isin(cell_indices, list(val_cells)))[0]
    val_idx   = np.where( np.isin(cell_indices, list(val_cells)))[0]
    print(f"  Leave-Cell-Out: {len(unique_cells)-n_val} train | {n_val} val cells")
    return train_idx, val_idx


# ─── 5. Evaluation ────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = float(r2_score(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {
        "rmse":      rmse,
        "r2":        r2,
        "pearson_r": float(pr),
        "spearman_r": float(sr),
    }


# ─── 6. Main training loop ────────────────────────────────────────────────────

def run_baselines(ccle_dir: str, smiles_csv: str, mut_path: str, out_path: str):
    data = load_data(ccle_dir, smiles_csv, mut_path)

    drug_indices = data["drug_indices"]
    cell_indices = data["cell_indices"]
    y            = data["ic50_z"]

    print("\n[baseline] Building feature matrices...")
    X_full,  y = build_features(data, use_ecfp=True)
    X_omics, _ = build_features(data, use_ecfp=False)

    splits = {
        "Random":         split_random(drug_indices, cell_indices, y),
        "Leave-Drug-Out": split_leave_drug_out(drug_indices, cell_indices, y),
        "Leave-Cell-Out": split_leave_cell_out(drug_indices, cell_indices, y),
    }

    def make_models():
        models = {
            "Ridge (ECFP4+omics)": (Ridge(alpha=1.0), X_full),
            "Ridge (omics only)":  (Ridge(alpha=1.0), X_omics),
            "RF (ECFP4+omics)": (
                RandomForestRegressor(
                    n_estimators=50, max_depth=6, max_samples=0.5,
                    n_jobs=-1, random_state=RANDOM_SEED
                ),
                X_full,
            ),
            "MLP (ECFP4+omics)": (
                MLPRegressor(
                    hidden_layer_sizes=(512, 256, 128),
                    activation="relu",
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_SEED,
                    verbose=False,
                ),
                X_full,
            ),
        }
        if HAS_XGB:
            models["XGBoost (ECFP4+omics)"] = (
                xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.5,
                    colsample_bytree=0.5,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                    verbosity=0,
                ),
                X_full,
            )
        return models

    rows = []

    for split_name, (train_idx, val_idx) in splits.items():
        print(f"\n{'='*64}")
        print(f"Split: {split_name}  (train={len(train_idx):,} | val={len(val_idx):,})")
        models = make_models()

        for model_name, (model, X) in models.items():
            print(f"  Fitting {model_name}...", end=" ", flush=True)
            model.fit(X[train_idx], y[train_idx])
            y_pred  = model.predict(X[val_idx])
            metrics = evaluate(y[val_idx], y_pred)
            print(
                f"RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}"
                f"  Pearson={metrics['pearson_r']:.4f}  Spearman={metrics['spearman_r']:.4f}"
            )
            rows.append({
                "Model":      model_name,
                "Split":      split_name,
                "RMSE":       round(metrics["rmse"],       4),
                "R2":         round(metrics["r2"],         4),
                "Pearson_r":  round(metrics["pearson_r"],  4),
                "Spearman_r": round(metrics["spearman_r"], 4),
            })

    df_results = pd.DataFrame(rows)

    # ── Append Bi-Int reference rows ──────────────────────────────────────────
    df_biint = pd.DataFrame(_BIINT_RESULTS)
    df_all   = pd.concat([df_results, df_biint], ignore_index=True)

    print(f"\n{'='*64}")
    print("RÉSULTATS COMPLETS (baselines + Bi-Int)")
    col_order = ["Model", "Split", "RMSE", "R2", "Pearson_r", "Spearman_r"]
    print(df_all[col_order].to_string(index=False))

    # ── Save CSV (baselines only, no None rows) ───────────────────────────────
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nRésultats sauvegardés : {out_path}")

    # ── Head-to-head: Ridge (random) vs Bi-Int (random) ───────────────────────
    ridge_row = df_results[
        (df_results["Split"] == "Random") & (df_results["Model"] == "Ridge (ECFP4+omics)")
    ]
    if not ridge_row.empty:
        ridge_rmse = ridge_row.iloc[0]["RMSE"]
        biint_rmse = 0.4633
        delta_pct  = (ridge_rmse - biint_rmse) / ridge_rmse * 100
        print(f"\n{'─'*64}")
        print(f"Bi-Int   (random, 20 epochs)        : RMSE={biint_rmse:.4f}  Pearson=0.884")
        print(f"Ridge    (random, ECFP4+omics)       : RMSE={ridge_rmse:.4f}")
        print(f"→ Bi-Int {'better' if delta_pct>0 else 'worse'} by {abs(delta_pct):.1f}% RMSE vs best Ridge")
        print(f"{'─'*64}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCLE baseline models")
    parser.add_argument("--ccle_dir",  default="Dataset/ccle_broad_2019")
    parser.add_argument("--smiles_csv", default="Dataset/ccle_drug_smiles.csv")
    parser.add_argument("--mut_path",  default="")
    parser.add_argument("--out",       default="Dataset/baseline_results.csv")
    args = parser.parse_args()

    run_baselines(
        ccle_dir  = args.ccle_dir,
        smiles_csv = args.smiles_csv,
        mut_path  = args.mut_path,
        out_path  = args.out,
    )
