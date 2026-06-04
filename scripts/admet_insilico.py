"""
admet_insilico.py
==================
Calcul ADMET in silico sur les top-5 candidats (GraphGA + BRICS-DQN)
via RDKit + propriétés physicochimiques étendues.

Propriétés calculées :
  - Lipinski (MW, LogP, HBD, HBA)
  - Solubilité aqueux estimée (ESOL — Delaney 2004)
  - Perméabilité membrane (TPSA)
  - Absorption orale prédite (règle de Veber)
  - Règle des 3 de Pfizer (toxicité hépato)
  - Score de drug-likeness étendu (QED + SA + ADMET composite)
  - Alertes de toxicité structurales (PAINS, Brenk, Ames mutagenicity motifs)

Outputs:
  Dataset/admet_insilico_top5.csv
  figures/phase3_interpretability_reliability/15_admet_radar.png
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
    from rdkit.Chem import Descriptors, AllChem, FilterCatalog, QED
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[ERROR] RDKit requis pour l'analyse ADMET.")
    sys.exit(1)

# ── Top candidats (GraphGA top-3 + BRICS-DQN top-2) ─────────────────────────
CANDIDATES = [
    {
        "id": "Gra-1", "source": "GraphGA",
        "smiles": "CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1",
        "note": "QED=0.872, MW=303, score_composite=1.667"
    },
    {
        "id": "Gra-2", "source": "GraphGA",
        "smiles": "COC(=O)OCC(=O)OCC(=O)Nc1ccccc1N(C)C",
        "note": "QED=0.784, MW=310, score_composite=1.656"
    },
    {
        "id": "Gra-9", "source": "GraphGA",
        "smiles": "CC(C)CN1CCCN(C)CC(c2ccccc2CO)C1C",
        "note": "QED=0.926, MW=304, plus haute drug-likeness"
    },
    {
        "id": "BRI-46", "source": "BRICS-DQN",
        "smiles": "O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1",
        "note": "quality_score=0.925, SA=1.89, sulfonamide"
    },
    {
        "id": "BRI-12", "source": "BRICS-DQN",
        "smiles": "NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1",
        "note": "quality_score=0.916, SA=1.68 (plus facile à synthétiser)"
    },
]


def esol(mol):
    """ESOL (Delaney 2004) — solubilité aqueuse estimée log(mol/L)."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rb   = Descriptors.NumRotatableBonds(mol)
    ap   = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def pfizer_3_rule(mw, logp):
    """Règle des 3 de Pfizer : MW > 500 ET LogP > 3 → risque hépatotoxique."""
    return mw > 500 and logp > 3


def ames_alert(mol):
    """Alertes structurales de mutagénicité de type Ames (motifs SMARTS simplifiés)."""
    AMES_SMARTS = [
        "[N;X3;v3]~[N;X3;v3]",            # hydrazine/azide
        "[nH]1cccc1",                       # imidazole NH
        "O=N(=O)[#6]",                      # nitro aromatique
        "[#6][N+](=O)[O-]",                 # nitro aliphatique
        "c1ccc2[nH]ccc2c1",                # indole (pro-mutagène dans certains contextes)
        "[CX3](=O)[F,Cl,Br,I]",           # halogénure acyle
    ]
    for sma in AMES_SMARTS:
        patt = Chem.MolFromSmarts(sma)
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False


def compute_admet(smiles, mol_id):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Propriétés physicochimiques de base ──────────────────────────────────
    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    hbd   = Descriptors.NumHDonors(mol)
    hba   = Descriptors.NumHAcceptors(mol)
    tpsa  = Descriptors.TPSA(mol)
    rb    = Descriptors.NumRotatableBonds(mol)
    arom  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    mw_exact = Descriptors.ExactMolWt(mol)

    # ── Règles de filtre ─────────────────────────────────────────────────────
    lipinski_ok = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    veber_ok    = (rb <= 10 and tpsa <= 140)
    pfizer_ok   = not pfizer_3_rule(mw, logp)
    ames_ok     = not ames_alert(mol)

    # ── Solubilité ESOL ──────────────────────────────────────────────────────
    esol_val = esol(mol)
    solubility_class = (
        "Très soluble"  if esol_val > -1 else
        "Soluble"       if esol_val > -2 else
        "Modérément"    if esol_val > -4 else
        "Peu soluble"
    )

    # ── Absorption orale prédite ─────────────────────────────────────────────
    # Règle empirique : TPSA < 90 Å² → bonne absorption
    absorption_pred = (
        "Bonne"     if tpsa < 60 else
        "Modérée"   if tpsa < 90 else
        "Limitée"
    )

    # ── Perméabilité BBB (Blood-Brain Barrier) ───────────────────────────────
    # PSA < 60 + MW < 400 + LogP 1-4 → probable pénétration SNC
    bbb_pen = (tpsa < 60 and mw < 400 and 1 <= logp <= 4)

    # ── QED et SA ────────────────────────────────────────────────────────────
    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = None

    try:
        from rdkit.Contrib.SA_Score import sascorer
        sa_val = sascorer.calculateScore(mol)
    except Exception:
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(ROOT, "venv_tf/lib/python3.12/site-packages/rdkit/Contrib"))
            from SA_Score import sascorer
            sa_val = sascorer.calculateScore(mol)
        except Exception:
            sa_val = None

    # ── Filtres PAINS ────────────────────────────────────────────────────────
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_cat  = FilterCatalog.FilterCatalog(pains_params)
    pains_flag = pains_cat.HasMatch(mol)

    # ── Score ADMET composite (0–1, 1=meilleur) ──────────────────────────────
    score_parts = [
        1.0 if lipinski_ok else 0.0,
        1.0 if veber_ok    else 0.0,
        1.0 if pfizer_ok   else 0.0,
        1.0 if ames_ok     else 0.0,
        1.0 if not pains_flag else 0.0,
        min(1.0, max(0.0, (esol_val + 6) / 6)),   # solubilité normalisée
    ]
    admet_score = round(np.mean(score_parts), 3)

    return {
        "id":               mol_id,
        "MW (Da)":          round(mw, 1),
        "LogP":             round(logp, 3),
        "HBD":              hbd,
        "HBA":              hba,
        "TPSA (Å²)":        round(tpsa, 1),
        "RotBonds":         rb,
        "AromaticAtoms":    arom,
        "ESOL log(mol/L)":  round(esol_val, 2),
        "Solubilité":       solubility_class,
        "Absorption orale": absorption_pred,
        "BBB pénétration":  "Probable" if bbb_pen else "Improbable",
        "QED":              round(qed_val, 3) if qed_val else "N/A",
        "SA Score":         round(sa_val, 2)  if sa_val  else "N/A",
        "Lipinski OK":      lipinski_ok,
        "Veber OK":         veber_ok,
        "Pfizer OK (non-hépato)": pfizer_ok,
        "Ames alert":       not ames_ok,
        "PAINS":            pains_flag,
        "ADMET score":      admet_score,
    }


def main():
    print("=== ADMET in silico — Top-5 candidats ===\n")
    rows = []
    for cand in CANDIDATES:
        print(f"[{cand['id']}] {cand['smiles'][:50]}...")
        result = compute_admet(cand["smiles"], cand["id"])
        if result is None:
            print(f"  [WARN] SMILES invalide pour {cand['id']}")
            continue
        result["source"]  = cand["source"]
        result["note"]    = cand["note"]
        result["smiles"]  = cand["smiles"]
        rows.append(result)

        print(f"  MW={result['MW (Da)']} | LogP={result['LogP']} | TPSA={result['TPSA (Å²)']} | "
              f"ESOL={result['ESOL log(mol/L)']} ({result['Solubilité']})")
        print(f"  Lipinski={result['Lipinski OK']} | Veber={result['Veber OK']} | "
              f"Pfizer={result['Pfizer OK (non-hépato)']} | Ames={not result['Ames alert']} | PAINS={result['PAINS']}")
        print(f"  BBB={result['BBB pénétration']} | Absorption={result['Absorption orale']} | "
              f"ADMET_score={result['ADMET score']}")
        print()

    if not rows:
        print("[ERROR] Aucun candidat valide.")
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, "Dataset/admet_insilico_top5.csv")
    df.to_csv(out_csv, index=False)
    print(f"[CSV] → {out_csv}")

    # ── Figure 15 — Radar ADMET ───────────────────────────────────────────────
    RADAR_PROPS = ["MW (Da)", "LogP", "TPSA (Å²)", "HBD", "HBA", "ESOL log(mol/L)"]
    RADAR_LIMITS = {
        "MW (Da)":           (0, 500),
        "LogP":              (-2, 5),
        "TPSA (Å²)":         (0, 140),
        "HBD":               (0, 5),
        "HBA":               (0, 10),
        "ESOL log(mol/L)":   (-6, 0),
    }
    RADAR_OPTIMAL = {
        "MW (Da)":           250,
        "LogP":              2.0,
        "TPSA (Å²)":         75,
        "HBD":               2,
        "HBA":               5,
        "ESOL log(mol/L)":   -2,
    }

    def normalize(val, prop):
        lo, hi = RADAR_LIMITS[prop]
        return max(0, min(1, (float(val) - lo) / (hi - lo)))

    N = len(RADAR_PROPS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Radar chart
    ax = axes[0]
    ax = plt.subplot(121, projection="polar")
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    for i, row in df.iterrows():
        vals = [normalize(row[p], p) for p in RADAR_PROPS]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i % len(colors)], linewidth=2, label=row["id"])
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.08)

    # Zone optimale
    opt_vals = [normalize(RADAR_OPTIMAL[p], p) for p in RADAR_PROPS]
    opt_vals += opt_vals[:1]
    ax.plot(angles, opt_vals, "k--", linewidth=1, alpha=0.4, label="Optimal")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_PROPS, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Profil ADMET normalisé\n(extérieur = limite Lipinski)", size=11, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    # Tableau récap
    ax2 = axes[1]
    ax2.axis("off")
    table_data = []
    col_labels = ["ID", "ADMET\nscore", "Solubilité", "Absorption", "BBB", "Ames\nalert", "PAINS"]
    for _, row in df.iterrows():
        table_data.append([
            row["id"],
            f"{row['ADMET score']:.3f}",
            row["Solubilité"],
            row["Absorption orale"],
            row["BBB pénétration"],
            "⚠️" if row["Ames alert"] else "✓",
            "⚠️" if row["PAINS"]      else "✓",
        ])
    tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # Colorier les cellules ADMET score
    for i, row in enumerate(table_data):
        score = float(row[1])
        color = "#c8e6c9" if score >= 0.8 else ("#fff9c4" if score >= 0.6 else "#ffcdd2")
        tbl[i+1, 1].set_facecolor(color)
    ax2.set_title("Récapitulatif ADMET in silico\nTop-5 candidats", size=11, pad=10)

    plt.tight_layout()
    out_fig = os.path.join(ROOT, "figures/phase3_interpretability_reliability/15_admet_radar.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 15] → {out_fig}")

    # ── Résumé final ─────────────────────────────────────────────────────────
    print("\n── Résumé ADMET top-5 ──")
    print(df[["id", "source", "ADMET score", "Solubilité", "Absorption orale",
              "Lipinski OK", "Veber OK", "Pfizer OK (non-hépato)",
              "Ames alert", "PAINS"]].to_string(index=False))


if __name__ == "__main__":
    main()
