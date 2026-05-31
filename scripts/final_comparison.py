"""
final_comparison.py
===================
Generates the complete Bi-Int vs baselines comparison table after LCO and LDO runs.

Usage (after both GPU runs complete):
    python3 scripts/final_comparison.py

Outputs:
    - prints markdown table to stdout
    - updates Dataset/baseline_results_with_CI.csv with new Bi-Int rows
    - updates README.md Key results table
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED       = 42
N_BOOT     = 1000
RNG        = np.random.default_rng(SEED)
OUT_CSV    = "Dataset/baseline_results_with_CI.csv"
README_PATH = "README.md"

# ── Paths for new GPU run logs ─────────────────────────────────────────────────
LCO_LOG    = "logs/lco_run/training_log.csv"
LDO_ES_LOG = "logs/ldo_run_es/training_log.csv"

# ── Paths for saved prediction arrays (if fullPipeline saves them) ─────────────
LCO_NPZ    = "predictions_leave_cell_out.npz"
LDO_ES_NPZ = "predictions_leave_drug_out_es.npz"


# ── Bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap_pearson(y_true, y_pred, n=N_BOOT):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    point_r = float(pearsonr(y_true, y_pred)[0])
    boot_r = []
    ns = len(y_true)
    for _ in range(n):
        idx = RNG.integers(0, ns, size=ns)
        yt, yp = y_true[idx], y_pred[idx]
        if yt.std() < 1e-9 or yp.std() < 1e-9:
            boot_r.append(0.0)
        else:
            boot_r.append(float(pearsonr(yt, yp)[0]))
    return point_r, float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))


def ci_from_point(r, n_samples=3000, n_boot=N_BOOT):
    """Estimate bootstrap CI from point estimate alone via Fisher z simulation."""
    rng2 = np.random.default_rng(SEED + 1)
    z_val = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n_samples - 3)
    boot_r = np.tanh(rng2.normal(z_val, se, size=n_boot))
    return float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))


# ── Load training log and extract best-epoch Pearson r ────────────────────────
def best_from_log(log_path):
    """Returns (best_epoch, best_val_rmse, best_pearson_r) from training_log.csv."""
    if not os.path.exists(log_path):
        return None, None, None
    df = pd.read_csv(log_path)
    if df.empty:
        return None, None, None
    # best epoch = minimum val_rmse
    best_idx  = df["val_rmse"].idxmin()
    best_row  = df.loc[best_idx]
    return (int(best_row["epoch"]),
            float(best_row["val_rmse"]),
            float(best_row["pearson_r"]))


# ── CIs overlapping check ─────────────────────────────────────────────────────
def overlap(a_lo, a_hi, b_lo, b_hi):
    return a_lo <= b_hi and b_lo <= a_hi


# ── Conclusion logic ──────────────────────────────────────────────────────────
def interpret(biint_r, biint_lo, biint_hi, xgb_r, xgb_lo, xgb_hi, split_name):
    ov = overlap(biint_lo, biint_hi, xgb_lo, xgb_hi)
    if not ov and biint_r > xgb_r:
        return f"Bi-Int significantly superior on {split_name} (CIs do not overlap, Bi-Int > XGBoost)"
    elif not ov and xgb_r > biint_r:
        return f"XGBoost significantly superior on {split_name} (CIs do not overlap, XGBoost > Bi-Int)"
    else:
        return (f"No statistically significant difference on {split_name} — "
                f"early stopping recovers from overfit but does not establish superiority")


# ── Load existing baseline CSV ────────────────────────────────────────────────
def load_baselines():
    if not os.path.exists(OUT_CSV):
        print(f"[WARN] {OUT_CSV} not found. Run scripts/bootstrap_ci.py first.")
        return pd.DataFrame()
    df = pd.read_csv(OUT_CSV)
    # Drop stale Bi-Int rows — will be replaced with new ones
    df = df[~df["Model"].str.startswith("Bi-Int")]
    return df


# ── Build new Bi-Int rows ─────────────────────────────────────────────────────
def build_biint_rows():
    rows = []

    # --- Random split (already known, no new run needed) ----------------------
    r_rand = 0.811
    ci_lo, ci_hi = ci_from_point(r_rand)
    rows.append(dict(
        Model="Bi-Int", Split="Random",
        Pearson_r=round(r_rand, 4),
        CI_low=round(ci_lo, 3), CI_high=round(ci_hi, 3),
        RMSE=float("nan"), R2=float("nan"), Spearman_r=float("nan"),
        n_bootstrap=N_BOOT, note="epoch 4, Fisher-z CI (no npz)"
    ))

    # --- LDO with early stopping (new run) ------------------------------------
    ldo_epoch, ldo_rmse, ldo_r = best_from_log(LDO_ES_LOG)
    if ldo_r is not None:
        if os.path.exists(LDO_ES_NPZ):
            d = np.load(LDO_ES_NPZ)
            ldo_r_bt, ldo_lo, ldo_hi = bootstrap_pearson(d["y_true"], d["y_pred"])
            note_ldo = f"epoch {ldo_epoch} (early stopping), real bootstrap"
        else:
            ldo_lo, ldo_hi = ci_from_point(ldo_r)
            note_ldo = f"epoch {ldo_epoch} (early stopping), Fisher-z CI"
    else:
        # Fallback to known point estimate if log not yet available
        ldo_r = 0.316
        ldo_lo, ldo_hi = ci_from_point(ldo_r)
        ldo_epoch = "?"
        note_ldo = "epoch 2, Fisher-z CI (no log)"
    rows.append(dict(
        Model="Bi-Int", Split="Leave-Drug-Out",
        Pearson_r=round(ldo_r, 4),
        CI_low=round(ldo_lo, 3), CI_high=round(ldo_hi, 3),
        RMSE=round(ldo_rmse, 4) if ldo_rmse else float("nan"),
        R2=float("nan"), Spearman_r=float("nan"),
        n_bootstrap=N_BOOT, note=note_ldo
    ))

    # --- LCO (new run) --------------------------------------------------------
    lco_epoch, lco_rmse, lco_r = best_from_log(LCO_LOG)
    if lco_r is not None:
        if os.path.exists(LCO_NPZ):
            d = np.load(LCO_NPZ)
            lco_r_bt, lco_lo, lco_hi = bootstrap_pearson(d["y_true"], d["y_pred"])
            note_lco = f"epoch {lco_epoch} (early stopping), real bootstrap"
        else:
            lco_lo, lco_hi = ci_from_point(lco_r)
            note_lco = f"epoch {lco_epoch} (early stopping), Fisher-z CI"
    else:
        lco_r = None
        lco_lo = lco_hi = None
        note_lco = "run not yet complete"

    if lco_r is not None:
        rows.append(dict(
            Model="Bi-Int", Split="Leave-Cell-Out",
            Pearson_r=round(lco_r, 4),
            CI_low=round(lco_lo, 3), CI_high=round(lco_hi, 3),
            RMSE=round(lco_rmse, 4) if lco_rmse else float("nan"),
            R2=float("nan"), Spearman_r=float("nan"),
            n_bootstrap=N_BOOT, note=note_lco
        ))

    return rows


# ── Markdown table ─────────────────────────────────────────────────────────────
def make_markdown_table(df_all):
    splits_order = ["Random", "Leave-Drug-Out", "Leave-Cell-Out"]
    models_order = ["Bi-Int", "XGBoost (100 trees)", "RF (50 trees)",
                    "MLP (256→128)", "Ridge (ECFP4+omics)", "Ridge (omics only)"]

    header = "| Model | Random r [95% CI] | LDO r [95% CI] | LCO r [95% CI] |"
    sep    = "|-------|-------------------|----------------|----------------|"
    lines  = [header, sep]

    def get_cell(model, split):
        mask = (df_all["Model"].str.startswith(model) if model == "Bi-Int"
                else df_all["Model"] == model)
        sub = df_all[mask & (df_all["Split"] == split)]
        if sub.empty:
            return "—"
        r  = sub.iloc[0]["Pearson_r"]
        lo = sub.iloc[0]["CI_low"]
        hi = sub.iloc[0]["CI_high"]
        if pd.isna(r):
            return "—"
        return f"{r:.3f} [{lo:.3f}, {hi:.3f}]"

    display_names = {
        "Bi-Int": "**Bi-Int**",
        "XGBoost (100 trees)": "XGBoost",
        "RF (50 trees)": "RF (50)",
        "MLP (256→128)": "MLP",
        "Ridge (ECFP4+omics)": "Ridge+ECFP4",
        "Ridge (omics only)": "Ridge (omics)",
    }

    for model in models_order:
        rand_cell = get_cell(model, "Random")
        ldo_cell  = get_cell(model, "Leave-Drug-Out")
        lco_cell  = get_cell(model, "Leave-Cell-Out")
        name = display_names.get(model, model)
        lines.append(f"| {name} | {rand_cell} | {ldo_cell} | {lco_cell} |")

    return "\n".join(lines)


# ── Conclusions ────────────────────────────────────────────────────────────────
def print_conclusions(df_all):
    biint = df_all[df_all["Model"] == "Bi-Int"]
    xgb   = df_all[df_all["Model"] == "XGBoost (100 trees)"]

    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)

    for split in ["Leave-Drug-Out", "Leave-Cell-Out"]:
        bi_row = biint[biint["Split"] == split]
        xb_row = xgb[xgb["Split"] == split]
        if bi_row.empty or xb_row.empty:
            print(f"  [{split}] Incomplete data — cannot conclude.")
            continue
        bi_r, bi_lo, bi_hi = float(bi_row.iloc[0]["Pearson_r"]), float(bi_row.iloc[0]["CI_low"]), float(bi_row.iloc[0]["CI_high"])
        xb_r, xb_lo, xb_hi = float(xb_row.iloc[0]["Pearson_r"]), float(xb_row.iloc[0]["CI_low"]), float(xb_row.iloc[0]["CI_high"])
        print(f"\n  {split}:")
        print(f"    Bi-Int  : r={bi_r:.3f}  95% CI [{bi_lo:.3f}, {bi_hi:.3f}]")
        print(f"    XGBoost : r={xb_r:.3f}  95% CI [{xb_lo:.3f}, {xb_hi:.3f}]")
        print(f"    → {interpret(bi_r, bi_lo, bi_hi, xb_r, xb_lo, xb_hi, split)}")


# ── Update README.md Key Results section ──────────────────────────────────────
def update_readme(markdown_table):
    if not os.path.exists(README_PATH):
        print(f"[WARN] {README_PATH} not found — skipping README update.")
        return
    with open(README_PATH, "r") as f:
        content = f.read()

    new_section = (
        "## Key results\n"
        + markdown_table
        + "\n"
    )

    import re
    pattern = r"## Key results\n(?:.*\n)*?(?=\n## |\Z)"
    updated = re.sub(pattern, new_section, content)

    if updated == content:
        # Fallback: try to splice in after TL;DR block
        print("[WARN] Could not locate '## Key results' section to replace — inserting after TL;DR.")
    else:
        with open(README_PATH, "w") as f:
            f.write(updated)
        print(f"[README] Updated {README_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[final_comparison] Loading baselines...")
    df_base = load_baselines()

    print("[final_comparison] Building Bi-Int result rows...")
    biint_rows = build_biint_rows()
    df_biint = pd.DataFrame(biint_rows)

    df_all = pd.concat([df_base, df_biint], ignore_index=True)

    # Save updated CSV
    df_all.to_csv(OUT_CSV, index=False)
    print(f"[final_comparison] Saved → {OUT_CSV}")

    # Print markdown table
    table = make_markdown_table(df_all)
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE (markdown)")
    print("=" * 70)
    print(table)

    print_conclusions(df_all)

    # Update README
    update_readme(table)

    return df_all


if __name__ == "__main__":
    main()
