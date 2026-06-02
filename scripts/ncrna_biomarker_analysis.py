"""
ncrna_biomarker_analysis.py
============================
Gradient×Input attribution on GEx to identify ncRNA biomarkers.
Uses exact same gex_mat as training (from NPZ cache).

Outputs:
  Dataset/ncrna_biomarker_importance.csv
  Dataset/gex_attrs_cache.npy          (reused by coding_biomarker_analysis.py)
  figures/phase3_interpretability_reliability/09_ncrna_importance.png
  figures/phase3_interpretability_reliability/10_ncrna_vs_drugs.png
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

NCRNA_ROLES = {
    "H19":    ("oncogene",   "oncogène — résistance chimiothérapie, prolifération"),
    "GAS5":   ("suppressor", "suppresseur de tumeur — apoptose, sensibilité chimio"),
    "MALAT1": ("oncogene",   "oncogène — métastase, migration"),
    "NEAT1":  ("oncogene",   "oncogène — stress, résistance"),
    "HOTAIR": ("oncogene",   "oncogène — invasion, métastase"),
    "MEG3":   ("suppressor", "suppresseur — apoptose"),
    "XIST":   ("suppressor", "suppresseur — inactivation X"),
    "TUG1":   ("oncogene",   "oncogène — prolifération, résistance"),
    "PVT1":   ("oncogene",   "oncogène — amplifié avec MYC"),
    "SNHG5":  ("oncogene",   "oncogène — prolifération"),
    "RMRP":   ("unknown",    "ARN ribonucléase mitochondriale"),
}
ROLE_COLORS = {"oncogene": "#d62728", "suppressor": "#1f77b4", "unknown": "#7f7f7f"}


def gxi_gex(model, atoms, adj, gex_v, mut_v, cnv_v, HP):
    """Gradient × Input on the GEx vector."""
    import tensorflow as tf
    drug_t = tf.constant(atoms[np.newaxis], dtype=tf.float32)
    adj_t  = tf.constant(adj[np.newaxis],   dtype=tf.float32)
    gex_var = tf.Variable(gex_v[np.newaxis].astype(np.float32))
    mut_t  = tf.constant(mut_v[np.newaxis],  dtype=tf.float32)
    cnv_t  = tf.constant(cnv_v[np.newaxis],  dtype=tf.float32)
    with tf.GradientTape() as tape:
        pred, _ = model((drug_t, adj_t, gex_var, mut_t, cnv_t), training=False)
    g = tape.gradient(pred, gex_var)
    if g is None:
        return np.zeros(HP["gex_dim"], dtype=np.float32)
    return (gex_var.numpy() * g.numpy()).squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(ROOT, "logs/ldo_checkpoint/biint_ic50_model.weights.h5"))
    parser.add_argument("--hp",      default=os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json"))
    parser.add_argument("--n-pairs", type=int, default=150)
    args = parser.parse_args()

    import tensorflow as tf
    from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch
    from _ccle_loader import load_ccle_cached, sample_pairs
    import json

    with open(args.hp) as f:
        HP.update(json.load(f))

    model = BiIntDigitalTwin(HP)
    dummy = generate_synthetic_ccle_batch(batch_size=2)
    model(dummy[:-1], training=False)
    model.load_weights(args.weights)
    print(f"[Model] {model.count_params():,} params loaded")

    gex_mat, cna_mat, mut_mat, common_cells, top_genes, _, smiles_map, ic50_df, drugs_w_smi = \
        load_ccle_cached(args.hp)
    pairs, labels = sample_pairs(gex_mat, cna_mat, mut_mat, common_cells,
                                  smiles_map, ic50_df, drugs_w_smi, n_pairs=args.n_pairs)
    drug_names = [l[0] for l in labels]

    # ── Compute attributions ──────────────────────────────────────────────────
    cache_path = os.path.join(ROOT, "Dataset/gex_attrs_cache.npy")
    if os.path.exists(cache_path):
        print(f"[Cache] Loading attrs from {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()
        all_attrs = data["attrs"]
        drug_names = data["drug_names"]
    else:
        print(f"[GxI] Computing on {len(pairs)} pairs...")
        all_attrs = []
        for i, (atoms, adj, gex_v, mut_v, cnv_v) in enumerate(pairs):
            if i % 25 == 0: print(f"  {i+1}/{len(pairs)}")
            attr = gxi_gex(model, atoms, adj, gex_v, mut_v, cnv_v, HP)
            all_attrs.append(attr)
        all_attrs = np.array(all_attrs)
        np.save(cache_path, {"attrs": all_attrs, "drug_names": drug_names, "top_genes": top_genes})
        print(f"[Cache] Saved → {cache_path}")

    mean_abs = np.mean(np.abs(all_attrs), axis=0)   # (978,)

    # ── ncRNA identification ──────────────────────────────────────────────────
    ncrna_df = pd.read_csv(os.path.join(ROOT, "Dataset/ncrna_in_top978.csv"))

    rows = []
    for _, row in ncrna_df.iterrows():
        idx  = int(row["index_in_978"])
        name = row["gene"]
        imp  = float(mean_abs[idx])
        role_key, role_desc = NCRNA_ROLES.get(name, ("unknown", "rôle non documenté"))
        rows.append(dict(name=name, index_in_978=idx, importance=imp,
                         role=role_key, role_description=role_desc))

    df_nc = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    df_nc["rank_ncrna"]  = df_nc.index + 1
    df_nc["rank_all978"] = df_nc["importance"].rank(ascending=False,
                               method="first").astype(int)   # rank among all 978

    # Recalculate rank_all978 correctly
    all_ranks = pd.Series(mean_abs).rank(ascending=False, method="first").astype(int)
    df_nc["rank_all978"] = df_nc["index_in_978"].apply(lambda i: int(all_ranks[i]))

    out_csv = os.path.join(ROOT, "Dataset/ncrna_biomarker_importance.csv")
    df_nc.to_csv(out_csv, index=False)
    print(f"[Output] Saved → {out_csv}")

    # ── Figure 09 ─────────────────────────────────────────────────────────────
    top20 = df_nc.head(20)
    colors = [ROLE_COLORS[r] for r in top20["role"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(top20)), top20["importance"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["name"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution moyenne |gradient×input| sur GEx", fontsize=11)
    ax.set_title("Top-20 transcrits non-codants — Importance Bi-Int\n"
                 "(Gradient×Input, 150 paires LDO validation)", fontsize=12, fontweight="bold")
    legend_handles = [
        mpatches.Patch(color=ROLE_COLORS["oncogene"],   label="Oncogène"),
        mpatches.Patch(color=ROLE_COLORS["suppressor"], label="Suppresseur de tumeur"),
        mpatches.Patch(color=ROLE_COLORS["unknown"],    label="Rôle non documenté"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    for i, (_, row) in enumerate(top20.iterrows()):
        if row["role"] != "unknown" and row["name"] in NCRNA_ROLES:
            ax.annotate(f"  rang {row['rank_all978']}/978",
                        xy=(row["importance"], i), xytext=(row["importance"] * 0.05, i),
                        fontsize=7, va="center", color="#555555")
    plt.tight_layout()
    p09 = os.path.join(ROOT, "figures/phase3_interpretability_reliability/09_ncrna_importance.png")
    fig.savefig(p09, dpi=150); plt.close()
    print(f"[Fig 09] → {p09}")

    # ── Figure 10 — heatmap ncRNA × drogues ──────────────────────────────────
    top10_idx   = df_nc.head(10)["index_in_978"].values
    top10_names = df_nc.head(10)["name"].values

    drug_series = {}
    for i, dname in enumerate(drug_names):
        drug_series.setdefault(dname, []).append(all_attrs[i])

    drug_means = {d: np.mean(np.abs(np.array(v)), axis=0)
                  for d, v in drug_series.items() if len(v) >= 2}
    top_drugs = sorted(drug_means, key=lambda d: np.mean(drug_means[d][top10_idx]),
                       reverse=True)[:15]

    if top_drugs:
        heat = np.array([[drug_means[d][idx] for idx in top10_idx] for d in top_drugs])
        denom = heat.max(axis=0, keepdims=True) - heat.min(axis=0, keepdims=True) + 1e-9
        heat_norm = (heat - heat.min(axis=0, keepdims=True)) / denom

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heat_norm, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(top10_names)))
        ax.set_xticklabels(top10_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(top_drugs)))
        ax.set_yticklabels([d[:28] for d in top_drugs], fontsize=8)
        ax.set_title("Importance ncRNA × Drogues (normalisée)\nTop-15 drogues × Top-10 ncRNA",
                     fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Importance normalisée")
        plt.tight_layout()
        p10 = os.path.join(ROOT, "figures/phase3_interpretability_reliability/10_ncrna_vs_drugs.png")
        fig.savefig(p10, dpi=150); plt.close()
        print(f"[Fig 10] → {p10}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TOP-20 ncRNA BIOMARQUEURS")
    print("="*65)
    print(f"  {'Gène':<22} {'Rang/76':>8} {'Rang/978':>9} {'Importance':>12}  Rôle")
    print("  " + "-"*62)
    for _, row in df_nc.head(20).iterrows():
        print(f"  {row['name']:<22} {row['rank_ncrna']:>8} {row['rank_all978']:>9} "
              f"{row['importance']:>12.6f}  {row['role']}")

    for t in ["H19", "GAS5"]:
        sub = df_nc[df_nc["name"] == t]
        if not sub.empty:
            r = sub.iloc[0]
            print(f"\n  ★ {t}: rang {int(r['rank_ncrna'])}/76 ncRNA | "
                  f"rang {int(r['rank_all978'])}/978 total | imp={r['importance']:.6f}")

    return df_nc


if __name__ == "__main__":
    main()
