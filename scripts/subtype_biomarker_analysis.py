"""
subtype_biomarker_analysis.py
==============================
Gradient×Input attribution par sous-type tumoral (breast, lung, haematological).
Réutilise le cache GxI de ncrna_biomarker_analysis.py.

Outputs:
  Dataset/subtype_biomarker_breast.csv
  Dataset/subtype_biomarker_lung.csv
  Dataset/subtype_biomarker_haem.csv
  figures/phase3_interpretability_reliability/14_subtype_biomarkers.png
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ── Sous-types et leurs mots-clés CCLE ───────────────────────────────────────
SUBTYPES = {
    "breast":         ["BREAST"],
    "lung":           ["LUNG"],
    "haematological": ["HAEMATOPOIETIC", "LYMPHOID"],
}

# Gènes cibles prioritaires (ncRNA + codants connus)
TARGET_GENES = {
    # ncRNA
    "H19":    "lncRNA oncogène — résistance chimio",
    "GAS5":   "lncRNA suppresseur — sensibilité chimio",
    "MALAT1": "lncRNA oncogène — métastase",
    "HOTAIR": "lncRNA oncogène — invasion",
    "RMRP":   "ncRNA — ribonucléase mitochondriale",
    # Codants oncologiques classiques
    "EGFR":   "RTK — cible erlotinib/gefitinib",
    "ERBB2":  "RTK — cible trastuzumab",
    "KRAS":   "GTPase — driver pancréas/colon/poumon",
    "BRAF":   "kinase — cible vemurafenib",
    "TP53":   "suppresseur — muté ~50% cancers",
    "MYC":    "TF oncogène — amplification fréquente",
    "PIK3CA": "PI3K — driver sein/colon",
    "CCND3":  "cycline D3 — rang 1 gènes codants",
    "CTGF":   "TGF-β — microenvironnement",
    "AREG":   "ligand EGFR — résistance inhibiteurs EGFR",
    "SFN":    "14-3-3σ — checkpoint G2/M",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attrs-cache", default=os.path.join(ROOT, "Dataset/gex_attrs_cache.npy"))
    parser.add_argument("--hp",          default=os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json"))
    parser.add_argument("--n-pairs",     type=int, default=300)
    args = parser.parse_args()

    # ── Charger le cache GxI (ou calculer si absent) ─────────────────────────
    if not os.path.exists(args.attrs_cache):
        print(f"[ERROR] Cache GxI non trouvé : {args.attrs_cache}")
        print("  Lancer d'abord : python3 scripts/ncrna_biomarker_analysis.py")
        sys.exit(1)

    data      = np.load(args.attrs_cache, allow_pickle=True).item()
    all_attrs = data["attrs"]      # (n_pairs, 978)
    drug_names = data["drug_names"]  # list of n_pairs drug names
    top_genes  = data.get("top_genes", [])
    n_pairs    = len(all_attrs)
    print(f"[Cache] {n_pairs} paires chargées, {len(top_genes)} gènes")

    # ── Charger noms des cellules pour ce run ────────────────────────────────
    from _ccle_loader import load_ccle_cached, sample_pairs
    import json
    with open(args.hp) as f:
        hp = json.load(f)

    gex_mat, cna_mat, mut_mat, common_cells, top_genes_live, _, smiles_map, ic50_df, drugs_w_smi = \
        load_ccle_cached(args.hp)

    if not top_genes:
        top_genes = top_genes_live

    # Re-tirer les mêmes paires (seed identique) pour récupérer les noms de cellules
    pairs, labels = sample_pairs(gex_mat, cna_mat, mut_mat, common_cells,
                                 smiles_map, ic50_df, drugs_w_smi,
                                 n_pairs=min(args.n_pairs, n_pairs), seed=42)
    cell_names = [l[1] for l in labels]
    n_use = min(len(cell_names), n_pairs)
    attrs_use  = all_attrs[:n_use]
    cell_names = cell_names[:n_use]
    drug_names = drug_names[:n_use] if len(drug_names) >= n_use else drug_names

    print(f"[Info] {n_use} paires utilisées pour l'analyse sous-types")

    # ── Créer les masques de sous-types ──────────────────────────────────────
    subtype_masks = {}
    for st_name, keywords in SUBTYPES.items():
        mask = [any(kw in c.upper() for kw in keywords) for c in cell_names]
        n_st = sum(mask)
        print(f"  {st_name:20s}: {n_st} paires")
        if n_st >= 5:
            subtype_masks[st_name] = mask
        else:
            print(f"  [SKIP] {st_name} — moins de 5 paires, ignoré")

    if not subtype_masks:
        print("[WARN] Aucun sous-type avec assez de paires. Vérifier les noms de cellules.")
        print("  Exemples :", cell_names[:10])
        sys.exit(0)

    # ── Générer le dictionnaire index de gène ────────────────────────────────
    gene_to_idx = {g: i for i, g in enumerate(top_genes)}

    # ── Calculer importance par sous-type ────────────────────────────────────
    results = {}  # st_name -> dict(gene -> importance)
    for st_name, mask in subtype_masks.items():
        idx_list = [i for i, m in enumerate(mask) if m]
        sub_attrs = attrs_use[idx_list]  # (n_st, 978)
        mean_abs  = np.mean(np.abs(sub_attrs), axis=0)
        results[st_name] = mean_abs

    # ── Construire tableau des gènes cibles ──────────────────────────────────
    rows = []
    for gene, desc in TARGET_GENES.items():
        if gene not in gene_to_idx:
            continue
        idx = gene_to_idx[gene]
        row = {"gene": gene, "description": desc}
        for st_name in subtype_masks:
            row[f"importance_{st_name}"] = float(results[st_name][idx])
        rows.append(row)

    df_tgt = pd.DataFrame(rows)
    print(f"\n[Targets] {len(df_tgt)} gènes cibles trouvés dans les 978 features")

    # ── Sauvegarder CSV par sous-type ─────────────────────────────────────────
    out_dir = os.path.join(ROOT, "Dataset")
    for st_name, mean_abs in results.items():
        ranked = sorted(enumerate(mean_abs), key=lambda x: x[1], reverse=True)
        rows_st = []
        for rank_i, (gene_idx, imp) in enumerate(ranked[:50]):
            gene = top_genes[gene_idx] if gene_idx < len(top_genes) else f"gene_{gene_idx}"
            rows_st.append({"rank": rank_i+1, "gene": gene,
                             "importance": round(imp, 8), "subtype": st_name})
        df_st = pd.DataFrame(rows_st)
        out_path = os.path.join(out_dir, f"subtype_biomarker_{st_name}.csv")
        df_st.to_csv(out_path, index=False)
        print(f"[CSV] {out_path}")
        print(f"  Top-5 {st_name}: {[r['gene'] for r in rows_st[:5]]}")

    # ── Figure 14 — Heatmap gènes cibles × sous-types ────────────────────────
    imp_cols = [c for c in df_tgt.columns if c.startswith("importance_")]
    if len(imp_cols) == 0 or df_tgt.empty:
        print("[WARN] Pas de données pour la figure.")
        return

    # Normaliser par colonne (sous-type)
    mat = df_tgt[imp_cols].values.astype(float)
    for j in range(mat.shape[1]):
        col_max = mat[:, j].max()
        if col_max > 0:
            mat[:, j] /= col_max

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Heatmap
    ax = axes[0]
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(imp_cols)))
    ax.set_xticklabels([c.replace("importance_", "").capitalize() for c in imp_cols],
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(df_tgt)))
    ax.set_yticklabels(df_tgt["gene"], fontsize=9)
    ax.set_title("Importance normalisée par sous-type tumoral\n(Gradient×Input, gènes cibles)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Importance normalisée")

    # Barres empilées
    ax2 = axes[1]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    bar_w = 0.6
    x = np.arange(len(df_tgt))
    bottoms = np.zeros(len(df_tgt))
    for j, (col, color) in enumerate(zip(imp_cols, colors)):
        st_label = col.replace("importance_", "").capitalize()
        ax2.barh(x, mat[:, j], left=bottoms, height=bar_w,
                 color=color, alpha=0.8, label=st_label)
        bottoms += mat[:, j]
    ax2.set_yticks(x)
    ax2.set_yticklabels(df_tgt["gene"], fontsize=9)
    ax2.set_xlabel("Importance cumulée normalisée", fontsize=10)
    ax2.set_title("Score cumulé\npar gène cible", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.invert_yaxis()

    plt.tight_layout()
    out_fig = os.path.join(ROOT, "figures/phase3_interpretability_reliability/14_subtype_biomarkers.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 14] → {out_fig}")

    # ── Afficher un résumé des gènes H19 / GAS5 par sous-type ────────────────
    print("\n── Résultats H19 / GAS5 par sous-type ──")
    for gene in ["H19", "GAS5"]:
        row = df_tgt[df_tgt["gene"] == gene]
        if row.empty:
            print(f"  {gene}: non trouvé dans les features")
            continue
        vals = {col.replace("importance_", ""): float(row[col].values[0])
                for col in imp_cols}
        # rang dans chaque sous-type
        for st_name, mean_abs in results.items():
            gene_imp = vals.get(st_name, 0)
            rank_st = sum(1 for v in mean_abs if v > gene_imp) + 1
            n_feats = len(mean_abs)
            print(f"  {gene} [{st_name}]: importance={gene_imp:.2e}, rang {rank_st}/{n_feats}")


if __name__ == "__main__":
    main()
