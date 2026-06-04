"""
uncertainty_mc_dropout_v2.py
=============================
MC Dropout recalibré avec dropout_rate=0.2 (correction sous-estimation v1).
N=50 passes, 300 paires pour un seuil statistiquement plus robuste.

Outputs:
  Dataset/uncertainty_mc_dropout_v2.csv
  figures/phase3_interpretability_reliability/12b_uncertainty_v2.png
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

MC_DROPOUT_RATE = 0.2   # taux recalibré (v1 = 0.1)
N_SAMPLES       = 50    # passages forward (v1 = 30)
N_PAIRS         = 300   # paires (v1 = 200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    default=os.path.join(ROOT, "logs/ldo_checkpoint/biint_ic50_model.weights.h5"))
    parser.add_argument("--hp",         default=os.path.join(ROOT, "logs/ldo_checkpoint/hp_snapshot.json"))
    parser.add_argument("--n-pairs",    type=int, default=N_PAIRS)
    parser.add_argument("--n-samples",  type=int, default=N_SAMPLES)
    parser.add_argument("--dropout-rate", type=float, default=MC_DROPOUT_RATE)
    args = parser.parse_args()

    import tensorflow as tf
    from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch
    from _ccle_loader import load_ccle_cached, sample_pairs

    with open(args.hp) as f:
        hp_snap = json.load(f)

    # ── Appliquer le dropout rate recalibré ──────────────────────────────────
    hp_snap["dropout_rate"] = args.dropout_rate
    HP.update(hp_snap)
    print(f"[Config] dropout_rate={args.dropout_rate} (recalibré, v1=0.1)")
    print(f"[Config] n_samples={args.n_samples} (v1=30), n_pairs={args.n_pairs} (v1=200)")

    # ── Charger le modèle avec le nouveau dropout rate ───────────────────────
    model = BiIntDigitalTwin(HP)
    dummy = generate_synthetic_ccle_batch(batch_size=2)
    model(dummy[:-1], training=False)
    model.load_weights(args.weights)
    print(f"[Model] {model.count_params():,} params  |  dropout={args.dropout_rate}")

    # ── Charger données CCLE ─────────────────────────────────────────────────
    gex_mat, cna_mat, mut_mat, common_cells, top_genes, _, smiles_map, ic50_df, drugs_w_smi = \
        load_ccle_cached(args.hp)
    pairs, labels = sample_pairs(gex_mat, cna_mat, mut_mat, common_cells,
                                  smiles_map, ic50_df, drugs_w_smi,
                                  n_pairs=args.n_pairs, seed=456)
    print(f"[MC] {len(pairs)} paires × {args.n_samples} passes forward")

    # ── Calcul MC Dropout ────────────────────────────────────────────────────
    rows = []
    for i, (atoms, adj, gex_v, mut_v, cnv_v) in enumerate(pairs):
        if i % 60 == 0:
            print(f"  paire {i+1}/{len(pairs)}")
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

    # ── Seuil : médiane + 1.5×IQR (plus robuste que médiane + std) ──────────
    q75, q25  = np.percentile(std_vals, [75, 25])
    threshold = float(np.median(std_vals) + 1.5 * (q75 - q25))
    df["alert"] = df["ic50_std"].apply(lambda s: "HIGH_UNCERTAINTY" if s > threshold else "OK")
    n_alert = int((df["alert"] == "HIGH_UNCERTAINTY").sum())

    print(f"\n[MC v2] Dropout={args.dropout_rate} | N={args.n_samples}")
    print(f"[MC v2] σ médian={np.median(std_vals):.4f} | σ moyen={np.mean(std_vals):.4f}")
    print(f"[MC v2] Seuil (médiane + 1.5×IQR)={threshold:.4f}")
    print(f"[MC v2] Alertes: {n_alert}/{len(df)} ({100*n_alert/len(df):.1f}%)")

    # Comparaison avec v1
    v1_path = os.path.join(ROOT, "Dataset/uncertainty_mc_dropout.csv")
    if os.path.exists(v1_path):
        df_v1 = pd.read_csv(v1_path)
        v1_med = df_v1["ic50_std"].median()
        v1_thr = df_v1["ic50_std"].median() + df_v1["ic50_std"].std()
        n_v1   = int((df_v1["alert"] == "HIGH_UNCERTAINTY").sum())
        print(f"\n── Comparaison v1 vs v2 ──")
        print(f"  v1 (dropout=0.10, N=30): σ médian={v1_med:.4f}, seuil={v1_thr:.4f}, alertes={n_v1}/{len(df_v1)}")
        print(f"  v2 (dropout={args.dropout_rate}, N={args.n_samples}): σ médian={np.median(std_vals):.4f}, seuil={threshold:.4f}, alertes={n_alert}/{len(df)}")
        ratio = np.median(std_vals) / v1_med if v1_med > 0 else 1
        print(f"  Ratio σ v2/v1 = {ratio:.2f}x (attendu > 1 si meilleure calibration)")

    # ── Sauvegarder CSV ───────────────────────────────────────────────────────
    out_csv = os.path.join(ROOT, "Dataset/uncertainty_mc_dropout_v2.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[CSV] → {out_csv}")

    # ── Figure 12b ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(std_vals, bins=35, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="#d62728", lw=2, ls="--",
               label=f"Seuil (médiane+1.5×IQR) = {threshold:.4f}")
    # Overlay v1 si disponible
    if os.path.exists(v1_path):
        df_v1 = pd.read_csv(v1_path)
        ax.hist(df_v1["ic50_std"].values, bins=35, color="#ff7f0e",
                alpha=0.35, edgecolor="white", label=f"v1 (dropout=0.10)")
        ax.legend(fontsize=8)
    ax.set_xlabel(f"σ MC Dropout (N={args.n_samples}, dropout={args.dropout_rate})", fontsize=11)
    ax.set_ylabel("Nombre de paires", fontsize=11)
    ax.set_title(f"Incertitude recalibrée (v2)\n{n_alert}/{len(df)} alertes ({100*n_alert/len(df):.0f}%)",
                 fontsize=11, fontweight="bold")

    ax2 = axes[1]
    sdf = df.sort_values("ic50_std").reset_index(drop=True)
    colors_sc = ["#d62728" if a == "HIGH_UNCERTAINTY" else "#1f77b4" for a in sdf["alert"]]
    ax2.scatter(range(len(sdf)), sdf["ic50_mean"], c=colors_sc, s=8, alpha=0.6, zorder=3)
    ax2.fill_between(range(len(sdf)), sdf["ci_low"], sdf["ci_high"],
                     alpha=0.12, color="#1f77b4")
    ax2.set_xlabel("Paires (triées par incertitude croissante)", fontsize=10)
    ax2.set_ylabel("IC50 prédit (log µM, z-score)", fontsize=10)
    ax2.set_title(f"Prédictions ± IC 95% (N={args.n_samples} passes)\n(rouge = haute incertitude)", fontsize=11)
    ax2.legend(handles=[
        mpatches.Patch(color="#d62728", label=f"Haute incertitude ({n_alert})"),
        mpatches.Patch(color="#1f77b4", label=f"OK ({len(df)-n_alert})"),
    ], fontsize=9)

    plt.tight_layout()
    p12b = os.path.join(ROOT, "figures/phase3_interpretability_reliability/12b_uncertainty_v2.png")
    os.makedirs(os.path.dirname(p12b), exist_ok=True)
    fig.savefig(p12b, dpi=150)
    plt.close()
    print(f"[Fig 12b] → {p12b}")

    return df, threshold


if __name__ == "__main__":
    main()
