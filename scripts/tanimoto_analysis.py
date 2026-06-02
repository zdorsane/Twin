"""
tanimoto_analysis.py
====================
Compute Morgan FP Tanimoto similarity between GraphGA candidates and CCLE drugs.

Usage:
    python3 scripts/tanimoto_analysis.py

Outputs:
    Dataset/graphga_tanimoto_vs_ccle.csv
    figures/phase2_validation_ablation/06_tanimoto_distribution.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ── paths ─────────────────────────────────────────────────────────────────────
CANDIDATES_CSV = "graphga_top_candidates.csv"
CCLE_SMILES    = "Dataset/ccle_drug_smiles.csv"
OUT_CSV        = "Dataset/graphga_tanimoto_vs_ccle.csv"
FIG_PATH       = "figures/phase2_validation_ablation/06_tanimoto_distribution.png"

FP_RADIUS = 2
FP_NBITS  = 2048


def mol_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)


def tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    # ── load candidates ───────────────────────────────────────────────────────
    df_cand = pd.read_csv(CANDIDATES_CSV)
    # normalize column names (canonical or smiles)
    smi_col = "canonical" if "canonical" in df_cand.columns else "smiles"
    candidates = df_cand[[smi_col, "qed", "mw", "logp"]].copy()
    candidates = candidates.rename(columns={smi_col: "smiles_candidate"})
    candidates["rank"] = range(1, len(candidates) + 1)

    # ── load CCLE drugs ───────────────────────────────────────────────────────
    df_ccle = pd.read_csv(CCLE_SMILES)
    # detect columns
    name_col  = [c for c in df_ccle.columns if c.lower() in ("drug_name", "name", "drug")][0]
    smiles_col = [c for c in df_ccle.columns if c.lower() in ("smiles", "canonical_smiles", "canonical")][0]

    ccle_fps   = []
    ccle_names = []
    ccle_smis  = []
    for _, row in df_ccle.iterrows():
        fp = mol_to_fp(str(row[smiles_col]))
        if fp is not None:
            ccle_fps.append(fp)
            ccle_names.append(str(row[name_col]))
            ccle_smis.append(str(row[smiles_col]))

    print(f"Loaded {len(ccle_fps)} valid CCLE drug fingerprints out of {len(df_ccle)} entries.")

    # ── compute pairwise Tanimoto ─────────────────────────────────────────────
    results = []
    for _, cand in candidates.iterrows():
        fp_cand = mol_to_fp(str(cand["smiles_candidate"]))
        if fp_cand is None:
            print(f"  [WARN] Could not parse candidate rank {int(cand['rank'])}: {cand['smiles_candidate']}")
            results.append({
                "rank":               int(cand["rank"]),
                "smiles_candidate":   cand["smiles_candidate"],
                "qed":                cand["qed"],
                "mw":                 cand["mw"],
                "logp":               cand["logp"],
                "max_tanimoto":       np.nan,
                "mean_tanimoto":      np.nan,
                "n_similar_07":       0,
                "closest_ccle_drug":  "",
                "closest_ccle_smiles": "",
            })
            continue

        sims = np.array([tanimoto(fp_cand, fp_ref) for fp_ref in ccle_fps])
        max_idx = int(np.argmax(sims))
        results.append({
            "rank":               int(cand["rank"]),
            "smiles_candidate":   cand["smiles_candidate"],
            "qed":                float(cand["qed"]),
            "mw":                 float(cand["mw"]),
            "logp":               float(cand["logp"]),
            "max_tanimoto":       float(sims[max_idx]),
            "mean_tanimoto":      float(sims.mean()),
            "n_similar_07":       int((sims >= 0.7).sum()),
            "closest_ccle_drug":  ccle_names[max_idx],
            "closest_ccle_smiles": ccle_smis[max_idx],
        })
        print(f"  Rank {int(cand['rank'])}: max_tanimoto={sims[max_idx]:.3f}  "
              f"closest={ccle_names[max_idx]}  QED={float(cand['qed']):.3f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # ── print summary ─────────────────────────────────────────────────────────
    valid = df_out.dropna(subset=["max_tanimoto"])
    print(f"\n=== Tanimoto Summary ===")
    print(f"  Candidates analysed : {len(valid)}/10")
    print(f"  max_tanimoto range  : [{valid['max_tanimoto'].min():.3f}, {valid['max_tanimoto'].max():.3f}]")
    print(f"  mean max_tanimoto   : {valid['max_tanimoto'].mean():.3f}")
    print(f"  Candidates <0.3 (novel)   : {(valid['max_tanimoto'] < 0.3).sum()}")
    print(f"  Candidates 0.3–0.7 (ideal): {((valid['max_tanimoto'] >= 0.3) & (valid['max_tanimoto'] < 0.7)).sum()}")
    print(f"  Candidates ≥0.7 (analogue): {(valid['max_tanimoto'] >= 0.7).sum()}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel A — histogram of max Tanimoto
    ax1.hist(valid["max_tanimoto"], bins=15, color="#4C72B0", edgecolor="white",
             linewidth=0.7, alpha=0.85)
    ax1.axvline(0.3, color="#C44E52", linewidth=1.5, linestyle="--", label="Novel (<0.3)")
    ax1.axvline(0.7, color="#DD8452", linewidth=1.5, linestyle="--", label="Analogue (≥0.7)")
    ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1],
                      0.3, 0.7, alpha=0.08, color="#55A868", label="Ideal zone (0.3–0.7)")
    ax1.set_xlabel("Max Tanimoto vs CCLE drugs", fontsize=12)
    ax1.set_ylabel("Number of candidates", fontsize=12)
    ax1.set_title("A — Structural Novelty\n(GraphGA candidates vs 201 CCLE drugs)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)

    # Panel B — scatter QED vs max Tanimoto
    scatter = ax2.scatter(valid["max_tanimoto"], valid["qed"],
                          c=valid["mw"], cmap="viridis", s=80, zorder=3,
                          edgecolors="white", linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Molecular Weight (Da)", fontsize=10)
    for _, row in valid.iterrows():
        ax2.annotate(f"R{int(row['rank'])}", (row["max_tanimoto"], row["qed"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=8, color="#333333")
    ax2.axvline(0.3, color="#C44E52", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.axvline(0.7, color="#DD8452", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.axhline(0.67, color="#9467BD", linewidth=1.2, linestyle=":", alpha=0.7,
                label="Approved drug median QED")
    ax2.set_xlabel("Max Tanimoto vs CCLE drugs", fontsize=12)
    ax2.set_ylabel("QED (drug-likeness)", fontsize=12)
    ax2.set_title("B — Drug-likeness vs Novelty\n(coloured by MW)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle("GraphGA Candidates — Tanimoto Similarity vs CCLE Drug Library",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
