"""
coding_biomarker_analysis.py
=============================
Attribution on coding genes (978 − ncRNA), reuses GxI cache from ncrna script.

Outputs:
  Dataset/coding_biomarker_importance.csv
  figures/phase3_interpretability_reliability/11_coding_biomarkers.png
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

KNOWN_MARKERS = {
    "EGFR","ERBB2","ERBB3","KRAS","NRAS","HRAS","BRAF","RAF1",
    "MYC","MYCN","CCND1","CDK4","CDK6","MDM2","MDM4",
    "PIK3CA","PIK3CB","AKT1","AKT2","MTOR","RPS6KB1",
    "MET","FGFR1","FGFR2","FGFR3","ALK","RET","ROS1",
    "BCR","ABL1","JAK2","STAT3","STAT5A","STAT5B",
    "FLT3","KIT","PDGFRA","PDGFRB",
    "NOTCH1","NOTCH2","HIF1A","VEGFA",
    "TP53","RB1","PTEN","APC","BRCA1","BRCA2",
    "CDKN2A","CDKN2B","NF1","NF2","VHL","SMAD4",
    "ATM","CHEK2","STK11","FBXW7","ARID1A",
    "TYMS","DHFR","TOP1","TOP2A","TOP2B",
    "CYP1A1","CYP2D6","GSTP1","ABCB1","ABCG2",
    "BCL2","BCL2L1","MCL1","BAX","CASP3",
    "MSH2","MLH1","PMS2","MSH6",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp",         default=os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json"))
    parser.add_argument("--attrs-cache", default=os.path.join(ROOT, "Dataset/gex_attrs_cache.npy"))
    args = parser.parse_args()

    # Load attrs from cache (built by ncrna script)
    if not os.path.exists(args.attrs_cache):
        print(f"[ERROR] Attrs cache not found: {args.attrs_cache}")
        print("  Run ncrna_biomarker_analysis.py first.")
        sys.exit(1)

    print(f"[Cache] Loading attributions from {args.attrs_cache}")
    data = np.load(args.attrs_cache, allow_pickle=True).item()
    all_attrs  = data["attrs"]    # (n_pairs, 978)
    drug_names = data["drug_names"]
    top_genes  = data["top_genes"]

    mean_abs = np.mean(np.abs(all_attrs), axis=0)   # (978,)

    # ncRNA indices to exclude
    ncrna_df  = pd.read_csv(os.path.join(ROOT, "Dataset/ncrna_in_top978.csv"))
    ncrna_set = set(ncrna_df["index_in_978"].values)

    rows = []
    for i, gene in enumerate(top_genes):
        if i in ncrna_set:
            continue
        rows.append(dict(name=gene, index_in_978=i,
                         importance=float(mean_abs[i]),
                         is_known_marker=(gene in KNOWN_MARKERS)))

    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_csv = os.path.join(ROOT, "Dataset/coding_biomarker_importance.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Output] Saved → {out_csv}")

    top20 = df.head(20)
    known_in_top20 = int(top20["is_known_marker"].sum())

    # ── Figure 11 ─────────────────────────────────────────────────────────────
    colors = ["#d62728" if km else "#aec7e8" for km in top20["is_known_marker"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(top20)), top20["importance"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["name"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution moyenne |gradient×input| sur GEx", fontsize=11)
    ax.set_title(f"Top-20 gènes codants — Importance Bi-Int\n"
                 f"({known_in_top20}/20 sont des biomarqueurs oncologiques connus)",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#d62728",  label="Biomarqueur oncologique connu"),
        mpatches.Patch(color="#aec7e8",  label="Gène codant (non annoté)"),
    ], loc="lower right", fontsize=10)
    plt.tight_layout()
    p11 = os.path.join(ROOT, "figures/phase3_interpretability_reliability/11_coding_biomarkers.png")
    fig.savefig(p11, dpi=150); plt.close()
    print(f"[Fig 11] → {p11}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TOP-20 GÈNES CODANTS")
    print("="*60)
    print(f"  {'Gène':<20} {'Rang':>6} {'Importance':>12}  Connu")
    print("  " + "-"*48)
    for _, row in top20.iterrows():
        mark = "★ " if row["is_known_marker"] else "  "
        print(f"  {mark}{row['name']:<18} {int(row['rank']):>6} "
              f"{row['importance']:>12.6f}  {'OUI' if row['is_known_marker'] else ''}")
    print(f"\n  Biomarqueurs connus dans top-20 : {known_in_top20}/20")

    return df


if __name__ == "__main__":
    main()
