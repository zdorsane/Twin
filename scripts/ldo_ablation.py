"""
LDO Ablation Study — compare improvement levers for Leave-Drug-Out generalisation.

Runs each configuration via subprocess, reads the resulting val_curves.json,
then produces:
  • Dataset/ldo_improvement_ablation.csv
  • figures/07_ldo_ablation.png

Usage:
    python3 scripts/ldo_ablation.py [--dry-run] [--skip-runs]

  --dry-run   Print the commands that would be run, but do not execute them.
  --skip-runs Skip training runs and just (re)generate the table/figure from
              any existing log directories.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

PYTHON = sys.executable  # use same interpreter / venv as this script

# Each entry: (label, log_dir, extra_args)
# extra_args is a list of CLI tokens appended to the base command.
BASE_CMD = [
    PYTHON, "fullPipeline.py",
    "--mode", "pretrained",
    "--loss-mode", "cross_entropy",
    "--split-mode", "leave_drug_out",
    "--no-ppo",
]

CONFIGS = [
    {
        "label":    "Baseline (actuel)",
        "log_dir":  "logs/ldo_baseline",
        "args":     ["--epochs", "5"],
        "note":     "Baseline sans early stopping, 5 epochs max",
    },
    {
        "label":    "+ Early stopping (patience=3, 15ep)",
        "log_dir":  "logs/ldo_es",
        "args":     ["--epochs", "15", "--early-stopping", "3"],
        "note":     "Levier 1: early stopping",
    },
    {
        "label":    "+ Dropout 0.3 + L2 1e-4",
        "log_dir":  "logs/ldo_reg",
        "args":     ["--epochs", "15", "--early-stopping", "3",
                     "--dropout-rate", "0.3", "--weight-decay", "1e-4"],
        "note":     "Levier 2: regularisation renforcée",
    },
    {
        "label":    "+ GNN freeze 3 ep",
        "log_dir":  "logs/ldo_freeze",
        "args":     ["--epochs", "15", "--early-stopping", "3",
                     "--dropout-rate", "0.3", "--weight-decay", "1e-4",
                     "--gnn-freeze-epochs", "3"],
        "note":     "Levier 2b: progressive unfreezing GNN",
    },
    {
        "label":    "+ 40k triplets",
        "log_dir":  "logs/ldo_40k",
        "args":     ["--epochs", "15", "--early-stopping", "3",
                     "--dropout-rate", "0.3", "--weight-decay", "1e-4",
                     "--data-size", "40000"],
        "note":     "Levier 3: plus de données (40k)",
    },
]

# XGBoost reference (from earlier runs)
XGBOOST_REF = {
    "label":         "XGBoost (cible à battre)",
    "pearson_r":     0.367,
    "val_rmse":      0.938,
    "epochs_to_best": "—",
    "note":          "Référence benchmark",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_config(cfg: dict, dry_run: bool = False) -> int:
    """Launch a training run; return subprocess return code."""
    cmd = BASE_CMD + cfg["args"] + ["--log-dir", cfg["log_dir"]]
    print(f"\n{'='*70}")
    print(f"[RUN] {cfg['label']}")
    print(f"  CMD: {' '.join(cmd)}")
    if dry_run:
        print("  [DRY-RUN] Skipped.")
        return 0
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__))).returncode
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed/60:.1f} min  (return code {ret})")
    return ret


def read_results(log_dir: str) -> dict:
    """
    Parse val_curves.json from a finished run.

    Returns dict with keys: pearson_r, val_rmse, epochs_to_best, epochs_run.
    Returns None if the file is missing.
    """
    json_path = os.path.join(log_dir, "val_curves.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        d = json.load(f)

    pearson_vals = d.get("pearson_r", [])
    val_rmse_vals = d.get("val_rmse", [])
    epochs_run    = d.get("epochs_run", len(pearson_vals))

    if not pearson_vals:
        return None

    # Best epoch = epoch with highest Pearson r (1-indexed)
    best_idx = int(np.argmax(pearson_vals))
    return {
        "pearson_r":      round(float(pearson_vals[best_idx]), 3),
        "val_rmse":       round(float(val_rmse_vals[best_idx]), 3) if val_rmse_vals else None,
        "epochs_to_best": best_idx + 1,
        "epochs_run":     epochs_run,
    }


def build_table(configs: list) -> pd.DataFrame:
    """Assemble results from all config log dirs into a DataFrame."""
    rows = []
    for cfg in configs:
        res = read_results(cfg["log_dir"])
        if res is None:
            print(f"  [WARN] No results found for '{cfg['label']}' in {cfg['log_dir']}")
            row = {
                "Config":         cfg["label"],
                "LDO r":          None,
                "LDO RMSE":       None,
                "epochs_to_best": None,
            }
        else:
            row = {
                "Config":         cfg["label"],
                "LDO r":          res["pearson_r"],
                "LDO RMSE":       res["val_rmse"],
                "epochs_to_best": res["epochs_to_best"],
            }
        rows.append(row)

    # Add XGBoost reference row
    rows.append({
        "Config":         XGBOOST_REF["label"],
        "LDO r":          XGBOOST_REF["pearson_r"],
        "LDO RMSE":       XGBOOST_REF["val_rmse"],
        "epochs_to_best": XGBOOST_REF["epochs_to_best"],
    })

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[CSV] Saved → {path}")


def save_figure(df: pd.DataFrame, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping figure generation.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    labels = df["Config"].tolist()
    r_vals = df["LDO r"].tolist()

    # Colours: blue for Bi-Int variants, orange for XGBoost reference
    colors = ["#e74c3c" if "XGBoost" in lbl else "#2980b9" for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, r_vals, color=colors, edgecolor="white", height=0.6)

    # Annotate bars with value
    for bar, val in zip(bars, r_vals):
        if val is None:
            continue
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=10,
        )

    # Reference line at XGBoost r
    ax.axvline(XGBOOST_REF["pearson_r"], color="#e74c3c",
               linestyle="--", linewidth=1.5, label=f"XGBoost r={XGBOOST_REF['pearson_r']}")
    ax.set_xlabel("Pearson r (LDO val)", fontsize=12)
    ax.set_title("LDO Improvement Ablation — Bi-Int vs XGBoost", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(v for v in r_vals if v is not None) + 0.08)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved → {path}")


def print_table(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  LDO Ablation Results")
    print("=" * 70)
    header = f"{'Config':<42} | {'LDO r':>6} | {'LDO RMSE':>8} | {'epochs':>6}"
    print(header)
    print("-" * 70)
    for _, row in df.iterrows():
        r   = f"{row['LDO r']:.3f}"   if row["LDO r"]   is not None else "  —  "
        rm  = f"{row['LDO RMSE']:.3f}" if row["LDO RMSE"] is not None else "   —  "
        ep  = str(row["epochs_to_best"]) if row["epochs_to_best"] is not None else "  —"
        print(f"{row['Config']:<42} | {r:>6} | {rm:>8} | {ep:>6}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LDO ablation runner + table/figure")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-runs", action="store_true",
                        help="Skip training; only rebuild table and figure from existing logs")
    args = parser.parse_args()

    # Change to project root so fullPipeline.py paths resolve correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"[CWD] {os.getcwd()}")

    if not args.skip_runs:
        failed = []
        for cfg in CONFIGS:
            rc = run_config(cfg, dry_run=args.dry_run)
            if rc != 0:
                print(f"  [WARN] Run failed (rc={rc}) — results will be missing for this config.")
                failed.append(cfg["label"])
        if failed:
            print(f"\n[WARN] {len(failed)} run(s) failed: {failed}")

    # Build results table
    df = build_table(CONFIGS)
    print_table(df)

    # Save outputs
    save_csv(df, "Dataset/ldo_improvement_ablation.csv")
    save_figure(df, "figures/07_ldo_ablation.png")

    print("\n[Done] Ablation complete.")


if __name__ == "__main__":
    main()
