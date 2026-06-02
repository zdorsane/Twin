"""
applicability_domain.py
========================
Tanimoto-based applicability domain for validation drugs.

Outputs:
  Dataset/applicability_domain.csv
  figures/13_applicability_domain.png

Usage:
  python3 scripts/applicability_domain.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit not available")

THRESHOLD_RELIABLE = 0.6
THRESHOLD_CAUTION  = 0.4


def smiles_to_fp(smi, nbits=2048, radius=2):
    if not HAS_RDKIT or not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def max_tanimoto(fp, fp_list):
    if fp is None or not fp_list:
        return 0.0, None
    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_list)
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), best_idx


def main():
    ic50_path = os.path.join(ROOT, "Dataset/ccle_broad_2019/data_drug_treatment_ic50.txt")
    smi_path  = os.path.join(ROOT, "Dataset/ccle_drug_smiles.csv")

    ic50_df = pd.read_csv(ic50_path, sep="\t", index_col=0)
    meta_cols = [c for c in ic50_df.columns if ic50_df[c].dtype == object]
    ic50_df = ic50_df.drop(columns=meta_cols, errors="ignore")

    smiles_df = pd.read_csv(smi_path)
    smiles_map = {}
    for _, r in smiles_df.iterrows():
        if pd.notna(r.get("smiles")) and r["smiles"]:
            smiles_map[r["drug_name"]] = r["smiles"]

    drugs_w_smi = [d for d in ic50_df.index if d in smiles_map]
    print(f"[AD] {len(drugs_w_smi)} drugs with SMILES")

    # LDO split: 85% train, 15% val (same seed as fullPipeline)
    rng = np.random.default_rng(42)
    unique_drugs = np.array(drugs_w_smi)
    val_drugs = set(rng.choice(unique_drugs, size=int(len(unique_drugs) * 0.20), replace=False))
    train_drugs = [d for d in drugs_w_smi if d not in val_drugs]
    val_drugs_list = [d for d in drugs_w_smi if d in val_drugs]
    print(f"[AD] Train: {len(train_drugs)} drugs | Val: {len(val_drugs_list)} drugs")

    # Build train fingerprints
    train_fps, train_names = [], []
    for d in train_drugs:
        fp = smiles_to_fp(smiles_map[d])
        if fp is not None:
            train_fps.append(fp)
            train_names.append(d)
    print(f"[AD] {len(train_fps)} train FPs computed")

    # Compute for all drugs (train + val) for comprehensive picture
    rows = []
    for drug in drugs_w_smi:
        fp = smiles_to_fp(smiles_map[drug])
        if fp is None:
            sim, closest = 0.0, "N/A"
        else:
            sim, best_idx = max_tanimoto(fp, [f for f in train_fps if f is not smiles_to_fp(smiles_map.get(drug, ""))])
            closest = train_names[best_idx] if best_idx is not None and best_idx < len(train_names) else "N/A"

        if sim >= THRESHOLD_RELIABLE:
            alert = "RELIABLE"
        elif sim >= THRESHOLD_CAUTION:
            alert = "CAUTION"
        else:
            alert = "UNRELIABLE"

        split = "val" if drug in val_drugs else "train"
        rows.append(dict(drug=drug, max_tanimoto=round(sim, 4),
                         closest_train_drug=closest, alert=alert, split=split))

    df = pd.DataFrame(rows).sort_values("max_tanimoto", ascending=False)
    out_csv = os.path.join(ROOT, "Dataset/applicability_domain.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[Output] Saved → {out_csv}")

    # Stats on val set only (the set the model actually predicts on)
    val_df = df[df["split"] == "val"]
    n_rel  = (val_df["alert"] == "RELIABLE").sum()
    n_cau  = (val_df["alert"] == "CAUTION").sum()
    n_unr  = (val_df["alert"] == "UNRELIABLE").sum()
    n_tot  = len(val_df)
    print(f"\n  Val drugs — applicability domain:")
    print(f"    🟢 RELIABLE   (Tanimoto ≥ {THRESHOLD_RELIABLE}) : {n_rel:3d} / {n_tot}  ({100*n_rel/n_tot:.1f}%)")
    print(f"    🟡 CAUTION    ({THRESHOLD_CAUTION}–{THRESHOLD_RELIABLE})         : {n_cau:3d} / {n_tot}  ({100*n_cau/n_tot:.1f}%)")
    print(f"    🔴 UNRELIABLE (< {THRESHOLD_CAUTION})             : {n_unr:3d} / {n_tot}  ({100*n_unr/n_tot:.1f}%)")

    # ── Figure 13 ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sims_all = df["max_tanimoto"].values

    ax.axvspan(THRESHOLD_RELIABLE, 1.0,  alpha=0.12, color="#2ca02c", label=f"FIABLE (≥{THRESHOLD_RELIABLE})")
    ax.axvspan(THRESHOLD_CAUTION, THRESHOLD_RELIABLE, alpha=0.12, color="#ff7f0e", label=f"PRUDENCE ({THRESHOLD_CAUTION}–{THRESHOLD_RELIABLE})")
    ax.axvspan(0.0, THRESHOLD_CAUTION, alpha=0.12, color="#d62728", label=f"NON-FIABLE (<{THRESHOLD_CAUTION})")
    ax.axvline(THRESHOLD_RELIABLE, color="#2ca02c", lw=1.5, ls="--")
    ax.axvline(THRESHOLD_CAUTION,  color="#ff7f0e", lw=1.5, ls="--")

    ax.hist(sims_all, bins=30, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Tanimoto max (drogue vs drogues d'entraînement)", fontsize=11)
    ax.set_ylabel("Nombre de drogues", fontsize=11)
    ax.set_title("Domaine d'applicabilité — Tanimoto Morgan FP (r=2, 2048 bits)\n"
                 f"Val: 🔴 {n_unr} non-fiable ({100*n_unr/n_tot:.0f}%) | "
                 f"🟡 {n_cau} prudence ({100*n_cau/n_tot:.0f}%) | "
                 f"🟢 {n_rel} fiable ({100*n_rel/n_tot:.0f}%)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig13 = os.path.join(ROOT, "figures/13_applicability_domain.png")
    fig.savefig(fig13, dpi=150)
    plt.close()
    print(f"[Fig 13] Saved → {fig13}")

    return df, n_rel, n_cau, n_unr, n_tot


if __name__ == "__main__":
    main()
