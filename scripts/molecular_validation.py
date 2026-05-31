"""
molecular_validation.py
=======================
Rigorous chemical/biological validation of top generated candidates.

Sources:
  - Dataset/graphga_tanimoto_vs_ccle.csv   (GraphGA top-10)
  - Dataset/brics_dqn_results.csv          (BRICS-DQN, top-50 by reward)

Outputs:
  - Dataset/molecular_validation_report.csv
  - figures/08_internal_diversity.png

Sections:
  1. Tanimoto similarity vs CCLE drugs
  2. Synthetic accessibility (SA score)
  3. MedChem filters: PAINS, Brenk, Lipinski+Veber, NP-likeness
  4. Internal library diversity heatmap
  5. Weighted quality score (QED + SA + diversity + filters, IC50-agnostic)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── RDKit ────────────────────────────────────────────────────────────────────
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed as calc_qed
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# SA score from RDKit Contrib
sys.path.insert(0, "/usr/share/RDKit/Contrib/SA_Score")
import sascorer

# ── NP-likeness (RDKit Contrib) ───────────────────────────────────────────────
NP_SCORER_AVAILABLE = False
try:
    sys.path.insert(0, "/usr/share/RDKit/Contrib/NP_Score")
    import npscorer
    NP_SCORER_AVAILABLE = True
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "Dataset")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

GRAPHGA_CSV = os.path.join(DATASET_DIR, "graphga_tanimoto_vs_ccle.csv")
BRICS_CSV   = os.path.join(DATASET_DIR, "brics_dqn_results.csv")
CCLE_CSV    = os.path.join(DATASET_DIR, "ccle_drug_smiles.csv")
OUT_CSV     = os.path.join(DATASET_DIR, "molecular_validation_report.csv")
OUT_FIG     = os.path.join(FIGURES_DIR, "08_internal_diversity.png")

TOP_N_BRICS = 50  # take top-N BRICS-DQN by reward

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load candidates
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates() -> pd.DataFrame:
    frames = []

    # GraphGA top candidates
    gga = pd.read_csv(GRAPHGA_CSV)
    gga = gga.rename(columns={"smiles_candidate": "smiles"})
    gga["source"] = "GraphGA"
    gga["reward"] = float("nan")
    frames.append(gga[["smiles", "qed", "source", "reward"]])

    # BRICS-DQN: top-N valid molecules by reward
    brics_raw = pd.read_csv(BRICS_CSV)
    brics_valid = brics_raw[brics_raw["valid"] == True].copy()
    brics_valid = brics_valid[brics_valid["reward"] > 0]
    brics_top = brics_valid.nlargest(TOP_N_BRICS, "reward")[["smiles", "reward"]].copy()
    brics_top["source"] = "BRICS-DQN"
    brics_top["qed"] = float("nan")
    frames.append(brics_top[["smiles", "qed", "source", "reward"]])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="smiles")
    return df


def load_ccle_mols():
    ccle = pd.read_csv(CCLE_CSV)
    ccle = ccle.dropna(subset=["smiles"])
    ccle = ccle[ccle["smiles"].str.strip() != ""]
    ccle = ccle.drop_duplicates(subset=["smiles"])
    mols, names = [], []
    for _, row in ccle.iterrows():
        m = Chem.MolFromSmiles(row["smiles"])
        if m is not None:
            mols.append(m)
            names.append(str(row.get("query_name", row.get("drug_name", ""))))
    return mols, names


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tanimoto vs CCLE
# ─────────────────────────────────────────────────────────────────────────────

def morgan_fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_similarity_interpretation(sim: float) -> str:
    if sim > 0.7:
        return "analogue proche (peu novateur, réaliste)"
    elif sim >= 0.4:
        return "zone idéale (proche du connu, brevetable)"
    else:
        return "structurellement nouveau (risqué)"


def compute_tanimoto_vs_ccle(cand_mol, ccle_fps, ccle_names):
    sims = DataStructs.BulkTanimotoSimilarity(morgan_fp(cand_mol), ccle_fps)
    idx_max = int(np.argmax(sims))
    return float(sims[idx_max]), ccle_names[idx_max]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SA score
# ─────────────────────────────────────────────────────────────────────────────

def compute_sa(mol) -> float:
    return sascorer.calculateScore(mol)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MedChem filters
# ─────────────────────────────────────────────────────────────────────────────

def build_filter_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


def lipinski_pass(mol) -> bool:
    mw   = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1


def veber_pass(mol) -> bool:
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa     = Descriptors.TPSA(mol)
    return rotbonds <= 10 and tpsa <= 140


def compute_medchem(mol, catalog) -> dict:
    pains_entries = catalog.GetMatches(mol)
    pains_names = [e.GetDescription() for e in pains_entries
                   if "PAINS" in e.GetDescription().upper() or
                      e.GetDescription().upper().startswith("P")]
    brenk_names = [e.GetDescription() for e in pains_entries
                   if e.GetDescription() not in pains_names]
    has_pains = len(pains_names) > 0
    has_brenk = len(brenk_names) > 0
    lip = lipinski_pass(mol)
    veb = veber_pass(mol)
    return {
        "pains_flag":   has_pains,
        "pains_alerts": "; ".join(pains_names) if has_pains else "",
        "brenk_flag":   has_brenk,
        "brenk_alerts": "; ".join(brenk_names) if has_brenk else "",
        "lipinski_pass": lip,
        "veber_pass":    veb,
        "mw":   round(Descriptors.ExactMolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "hbd":  rdMolDescriptors.CalcNumHBD(mol),
        "hba":  rdMolDescriptors.CalcNumHBA(mol),
        "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "tpsa": round(Descriptors.TPSA(mol), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. NP-likeness
# ─────────────────────────────────────────────────────────────────────────────

_NP_MODEL = None

def compute_np_likeness(mol) -> float:
    global _NP_MODEL
    if not NP_SCORER_AVAILABLE:
        return float("nan")
    if _NP_MODEL is None:
        _NP_MODEL = npscorer.readNPModel()
    return round(npscorer.scoreMol(mol, _NP_MODEL), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Internal diversity
# ─────────────────────────────────────────────────────────────────────────────

def compute_internal_diversity(fps):
    n = len(fps)
    sims = np.zeros((n, n))
    for i in range(n):
        row = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sims[i] = row
    off_diag = sims[np.triu_indices(n, k=1)]
    mean_sim  = float(np.mean(off_diag))
    diversity = 1.0 - mean_sim
    return sims, mean_sim, diversity


def plot_diversity_heatmap(sims, labels, out_path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.28), max(6, n * 0.25)))
    im = ax.imshow(sims, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Tanimoto similarity")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Internal similarity heatmap — candidate library", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Weighted quality score
#    QED (0-1): weight 0.30
#    SA  (1-10, inverted to 0-1): weight 0.25
#    Diversity contribution (0-1): weight 0.20
#    Medchem clean (0/1): weight 0.25
#    IC50 NOT included — explicitly per Marouane's request
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_score(row, mean_sim_to_others):
    qed_score = row["qed_computed"]
    sa_norm   = 1.0 - (row["sa_score"] - 1.0) / 9.0   # 1→1.0, 10→0.0
    div_score = 1.0 - mean_sim_to_others                # less similar = more diverse
    medchem_ok = (
        not row["pains_flag"]
        and not row["brenk_flag"]
        and row["lipinski_pass"]
        and row["veber_pass"]
    )
    medchem_score = 1.0 if medchem_ok else 0.0

    score = (0.30 * qed_score
             + 0.25 * sa_norm
             + 0.20 * div_score
             + 0.25 * medchem_score)
    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  Molecular Validation — Twin project")
    print("=" * 66)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading candidates...")
    cands = load_candidates()
    print(f"      {len(cands)} unique candidates (GraphGA + BRICS-DQN top-{TOP_N_BRICS})")

    print("[2/6] Loading CCLE reference drugs...")
    ccle_mols, ccle_names = load_ccle_mols()
    ccle_fps = [morgan_fp(m) for m in ccle_mols]
    print(f"      {len(ccle_mols)} CCLE drugs with valid SMILES")

    # ── Parse candidates ───────────────────────────────────────────────────
    valid_rows = []
    for _, row in cands.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is not None:
            valid_rows.append((row, mol))
    print(f"      {len(valid_rows)}/{len(cands)} candidates parsed by RDKit")

    # ── Build filter catalog ───────────────────────────────────────────────
    catalog = build_filter_catalog()

    # ── Per-molecule metrics ───────────────────────────────────────────────
    print("[3/6] Computing per-molecule metrics (Tanimoto, SA, MedChem, NP)...")
    records = []
    fps_list = []
    short_labels = []

    for i, (row, mol) in enumerate(valid_rows):
        fp = morgan_fp(mol)
        fps_list.append(fp)

        max_sim, closest_drug = compute_tanimoto_vs_ccle(mol, ccle_fps, ccle_names)
        sa_score = compute_sa(mol)
        mc = compute_medchem(mol, catalog)
        qed_val = calc_qed(mol)
        np_val  = compute_np_likeness(mol)

        smiles = str(row["smiles"])
        label  = f"{row['source'][:3]}-{i+1}"
        short_labels.append(label)

        records.append({
            "id":              label,
            "source":          row["source"],
            "smiles":          smiles,
            # ── Tanimoto vs CCLE ──────────────────────────────────────────
            "max_tanimoto_ccle":    round(max_sim, 4),
            "closest_ccle_drug":    closest_drug,
            "tanimoto_interpretation": tanimoto_similarity_interpretation(max_sim),
            # ── Drug-likeness ─────────────────────────────────────────────
            "qed_computed":    round(qed_val, 4),
            "sa_score":        round(sa_score, 3),
            "sa_flag_hard":    sa_score > 6,
            "np_likeness":     np_val,
            # ── Physicochemical ───────────────────────────────────────────
            **mc,
            # ── MedChem summary ───────────────────────────────────────────
            "medchem_clean":   (not mc["pains_flag"] and not mc["brenk_flag"]
                                and mc["lipinski_pass"] and mc["veber_pass"]),
        })

    df = pd.DataFrame(records)

    # ── Internal diversity ─────────────────────────────────────────────────
    print("[4/6] Computing internal diversity matrix...")
    sims_matrix, mean_sim, diversity = compute_internal_diversity(fps_list)
    print(f"      Mean intra-library Tanimoto: {mean_sim:.3f}")
    print(f"      Internal diversity (1 - mean_sim): {diversity:.3f}")

    # Per-molecule mean similarity to all others (for quality score)
    n = len(fps_list)
    mean_sim_to_others = []
    for i in range(n):
        row_sims = list(sims_matrix[i])
        row_sims.pop(i)
        mean_sim_to_others.append(float(np.mean(row_sims)) if row_sims else 0.0)
    df["mean_sim_to_library"] = [round(v, 4) for v in mean_sim_to_others]

    # ── Quality score ──────────────────────────────────────────────────────
    print("[5/6] Computing weighted quality scores...")
    df["quality_score"] = df.apply(
        lambda r: compute_quality_score(r, mean_sim_to_others[int(r.name)]), axis=1
    )
    df = df.sort_values("quality_score", ascending=False).reset_index(drop=True)

    # ── Save CSV ───────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    print(f"[saved] {OUT_CSV}")

    # ── Heatmap ────────────────────────────────────────────────────────────
    plot_diversity_heatmap(sims_matrix, short_labels, OUT_FIG)

    # ─────────────────────────────────────────────────────────────────────
    # Console summary
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)

    n_total       = len(df)
    n_medchem_ok  = df["medchem_clean"].sum()
    n_pains       = df["pains_flag"].sum()
    n_brenk       = df["brenk_flag"].sum()
    n_lip_fail    = (~df["lipinski_pass"]).sum()
    n_veber_fail  = (~df["veber_pass"]).sum()
    n_sa_hard     = df["sa_flag_hard"].sum()
    n_novel       = (df["max_tanimoto_ccle"] < 0.3).sum()
    n_ideal       = ((df["max_tanimoto_ccle"] >= 0.4) & (df["max_tanimoto_ccle"] <= 0.6)).sum()
    n_analogue    = (df["max_tanimoto_ccle"] > 0.7).sum()

    print(f"\n  Candidates total         : {n_total}")
    print(f"  Medchem CLEAN (all pass) : {n_medchem_ok} / {n_total}  "
          f"({100*n_medchem_ok/n_total:.0f}%)")
    print(f"\n  MedChem filter breakdown :")
    print(f"    PAINS alerts           : {n_pains}")
    print(f"    Brenk alerts           : {n_brenk}")
    print(f"    Lipinski fail          : {n_lip_fail}")
    print(f"    Veber fail             : {n_veber_fail}")
    print(f"    SA > 6 (hard to synth) : {n_sa_hard}")
    print(f"\n  Tanimoto vs CCLE :")
    print(f"    > 0.7  analogue proche : {n_analogue}")
    print(f"    0.4-0.6 zone idéale    : {n_ideal}")
    print(f"    < 0.3  structuralement : {n_novel}")
    print(f"\n  Internal library diversity : {diversity:.3f}  "
          f"(mean Tanimoto {mean_sim:.3f})")

    print("\n  Top-10 candidates by quality score (IC50-agnostic):")
    cols_show = ["id", "source", "quality_score", "qed_computed",
                 "sa_score", "medchem_clean", "max_tanimoto_ccle", "closest_ccle_drug"]
    print(df[cols_show].head(10).to_string(index=False))

    # ── IC50 DISCLAIMER (Marouane explicit request) ────────────────────────
    print("\n" + "┌" + "─" * 64 + "┐")
    print("│  ⚠  DISCLAIMER IC50                                           │")
    print("├" + "─" * 64 + "┤")
    print("│  Les IC50 prédits pour ces molécules sont extrapolés par un   │")
    print("│  modèle dont la performance en Leave-Drug-Out est limitée     │")
    print("│  (r = 0.316). Ces valeurs ne doivent PAS être interprétées   │")
    print("│  comme des prédictions fiables de potency.                    │")
    print("│  Validation in vitro requise avant toute conclusion.          │")
    print("└" + "─" * 64 + "┘")

    print(f"\n[6/6] Done. Report saved to:\n      {OUT_CSV}\n      {OUT_FIG}\n")


if __name__ == "__main__":
    main()
