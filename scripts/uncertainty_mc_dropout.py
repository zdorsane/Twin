"""
uncertainty_mc_dropout.py
==========================
MC Dropout uncertainty (N=30 passes). Uses exact same gex_mat as training.

Outputs:
  Dataset/uncertainty_mc_dropout.csv
  figures/phase3_interpretability_reliability/12_uncertainty_distribution.png
"""

import argparse, os, sys, warnings, json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default=os.path.join(ROOT, "logs/ldo_checkpoint/biint_ic50_model.weights.h5"))
    parser.add_argument("--hp",        default=os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json"))
    parser.add_argument("--n-pairs",   type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=30)
    args = parser.parse_args()

    import tensorflow as tf
    from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch
    from _ccle_loader import load_ccle_cached, sample_pairs

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
                                  smiles_map, ic50_df, drugs_w_smi,
                                  n_pairs=args.n_pairs, seed=123)

    print(f"[MC] {len(pairs)} pairs × {args.n_samples} passes")
    rows = []
    for i, (atoms, adj, gex_v, mut_v, cnv_v) in enumerate(pairs):
        if i % 40 == 0:
            print(f"  pair {i+1}/{len(pairs)}")
        drug, cell, ic50_true = labels[i]
        drug_t = tf.constant(atoms[np.newaxis], dtype=tf.float32)
        adj_t  = tf.constant(adj[np.newaxis],   dtype=tf.float32)
        gex_t  = tf.constant(gex_v[np.newaxis],  dtype=tf.float32)
        mut_t  = tf.constant(mut_v[np.newaxis],  dtype=tf.float32)
        cnv_t  = tf.constant(cnv_v[np.newaxis],  dtype=tf.float32)
        inp = (drug_t, adj_t, gex_t, mut_t, cnv_t)

        preds = [float(model(inp, training=True)[0].numpy().squeeze())
                 for _ in range(args.n_samples)]

        mean_p = float(np.mean(preds))
        std_p  = float(np.std(preds))
        rows.append(dict(
            drug=drug, cell=cell,
            ic50_true=round(ic50_true, 4),
            ic50_mean=round(mean_p, 4),
            ic50_std=round(std_p, 6),
            ci_low=round(float(np.percentile(preds, 2.5)), 4),
            ci_high=round(float(np.percentile(preds, 97.5)), 4),
        ))

    df = pd.DataFrame(rows)
    std_vals  = df["ic50_std"].values
    threshold = float(np.median(std_vals) + np.std(std_vals))
    df["alert"] = df["ic50_std"].apply(lambda s: "HIGH_UNCERTAINTY" if s > threshold else "OK")
    n_alert = int((df["alert"] == "HIGH_UNCERTAINTY").sum())

    out_csv = os.path.join(ROOT, "Dataset/uncertainty_mc_dropout.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[Output] Saved → {out_csv}")
    print(f"[MC] Threshold: {threshold:.4f}  |  HIGH_UNCERTAINTY: {n_alert}/{len(df)} ({100*n_alert/len(df):.1f}%)")

    # ── Figure 12 ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(std_vals, bins=30, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="#d62728", lw=2, ls="--",
               label=f"Seuil = {threshold:.4f}")
    ax.set_xlabel("Écart-type MC Dropout (N=30)", fontsize=11)
    ax.set_ylabel("Nombre de paires", fontsize=11)
    ax.set_title(f"Distribution de l'incertitude MC Dropout\n"
                 f"{n_alert}/{len(df)} paires au-dessus du seuil ({100*n_alert/len(df):.0f}%)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    sdf = df.sort_values("ic50_std").reset_index(drop=True)
    colors_sc = ["#d62728" if a == "HIGH_UNCERTAINTY" else "#1f77b4" for a in sdf["alert"]]
    ax2.scatter(range(len(sdf)), sdf["ic50_mean"], c=colors_sc, s=10, alpha=0.6, zorder=3)
    ax2.fill_between(range(len(sdf)), sdf["ci_low"], sdf["ci_high"],
                     alpha=0.15, color="#1f77b4", label="IC 95%")
    ax2.set_xlabel("Paires (triées par incertitude croissante)", fontsize=10)
    ax2.set_ylabel("IC50 prédit (log µM, z-score)", fontsize=10)
    ax2.set_title("Prédictions ± IC 95%\n(rouge = haute incertitude)", fontsize=11)
    ax2.legend(handles=[
        mpatches.Patch(color="#d62728", label=f"Haute incertitude ({n_alert})"),
        mpatches.Patch(color="#1f77b4", label=f"OK ({len(df)-n_alert})"),
    ], fontsize=9)

    plt.tight_layout()
    p12 = os.path.join(ROOT, "figures/phase3_interpretability_reliability/12_uncertainty_distribution.png")
    fig.savefig(p12, dpi=150); plt.close()
    print(f"[Fig 12] → {p12}")

    return df, threshold


if __name__ == "__main__":
    main()
