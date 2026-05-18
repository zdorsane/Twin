"""
compare_vae_losses.py
=====================
Compares three VAE loss modes on the CCLE QSAR IC50 prediction task:
  - kl            : original KL divergence (baseline)
  - cross_entropy : binary CE reconstruction, no KL regularization
  - both          : KL + binary CE reconstruction

Runs 10 epochs each on the same data split (random seed fixed for reproducibility).
Saves results to Dataset/vae_loss_comparison.csv.

Usage:
    python3 compare_vae_losses.py
    python3 compare_vae_losses.py --epochs 20 --no-pretrain
"""

import os, sys, argparse, warnings, logging
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, "/home/crbt/Twin")
from fullPipeline import (
    BiIntDigitalTwin, BiIntTrainer, BRICSMolecularFeaturizer,
    SMILESVocabulary, HP, load_ccle_real_data, make_tf_dataset,
    load_pretrained_drug_encoder,
)

CCLE_DIR = "Dataset/ccle_broad_2019"
OUTPUT_CSV = "Dataset/vae_loss_comparison.csv"
LOSS_MODES = ["kl", "cross_entropy", "both"]

NOTES = {
    "kl":           "Baseline — KL regularized",
    "cross_entropy": "No KL — pure reconstruction",
    "both":          "KL + CE combined",
}


def load_data(batch_size: int, max_samples: int = 0):
    """Return (train_ds, val_ds, data_source) trying CCLE first.
    max_samples > 0 : truncate the dataset for fast runs (--fast flag).
    """
    if os.path.isdir(CCLE_DIR):
        try:
            train_ds, val_ds, n_real = load_ccle_real_data(
                ccle_dir=CCLE_DIR, batch_size=batch_size
            )
            if max_samples > 0:
                n_batches = max(1, max_samples // batch_size)
                train_ds = train_ds.take(n_batches)
                val_ds   = val_ds.take(max(1, n_batches // 4))
                print(f"[Data] --fast: using {n_batches * batch_size} train / "
                      f"{max(1, n_batches // 4) * batch_size} val samples.")
            else:
                print(f"[Data] Loaded real CCLE data ({n_real} samples).")
            return train_ds, val_ds, "real"
        except Exception as exc:
            print(f"[Data] CCLE load failed ({exc}) — falling back to synthetic data.")
    else:
        print(f"[Data] CCLE directory '{CCLE_DIR}' not found — using synthetic data.")

    print(
        "[WARNING] Results below are on SYNTHETIC data and are NOT scientifically "
        "meaningful. They exist only to verify the training loop runs correctly."
    )
    train_ds = make_tf_dataset(n_samples=256, batch_size=batch_size)
    val_ds   = make_tf_dataset(n_samples=64,  batch_size=batch_size)
    return train_ds, val_ds, "synthetic"


def run_comparison(epochs: int = 10, use_pretrained: bool = True,
                   batch_size: int = None, max_samples: int = 0):
    """Train each loss mode and collect final metrics."""

    tf.random.set_seed(42)
    np.random.seed(42)

    bs = batch_size or HP["batch_size"]
    train_ds, val_ds, data_source = load_data(bs, max_samples=max_samples)

    rows = []

    for mode in LOSS_MODES:
        print(f"\n{'='*55}")
        print(f"=== Loss mode: {mode} ===")
        print(f"{'='*55}")

        # Fresh model — independent weights for each run
        model = BiIntDigitalTwin(HP)
        # Build the model with a dummy forward pass so weights exist before loading
        for batch in train_ds.take(1):
            drug_atoms, adj_mask, gex, mut, cnv, _ = batch
            model((drug_atoms, adj_mask, gex, mut, cnv), training=False, loss_mode=mode)
            break

        if use_pretrained:
            load_pretrained_drug_encoder(model)

        trainer = BiIntTrainer(model, HP, loss_mode=mode)
        history = trainer.fit(train_ds, val_ds, epochs=epochs)

        final_val_rmse  = float(history["val"][-1])
        final_pearson_r = float(history["pearson_r"][-1])
        # 'kl_loss' key tracks the vae_loss returned by the model (regardless of mode)
        final_loss      = float(history["val"][-1])  # val loss (rmse proxy used for comparison)

        rows.append({
            "loss_mode":   mode,
            "val_rmse":    round(final_val_rmse,  4),
            "pearson_r":   round(final_pearson_r, 4),
            "final_loss":  round(final_loss,      4),
            "epochs":      epochs,
            "data_source": data_source,
        })

        print(
            f"  -> Final val RMSE: {final_val_rmse:.4f} | "
            f"Pearson r: {final_pearson_r:.4f}"
        )

    return rows


def print_table(rows):
    """Print a formatted comparison table to stdout."""
    header = f"{'Loss mode':<16} | {'Val RMSE':>8} | {'Pearson r':>9} | {'Final Loss':>10} | Notes"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)

    label_map = {
        "kl":           "kl (original)",
        "cross_entropy": "cross_entropy",
        "both":          "both",
    }

    for row in rows:
        mode  = row["loss_mode"]
        label = label_map.get(mode, mode)
        note  = NOTES.get(mode, "")
        print(
            f"{label:<16} | {row['val_rmse']:>8.3f} | {row['pearson_r']:>9.3f} | "
            f"{row['final_loss']:>10.3f} | {note}"
        )
    print(separator)


def declare_winners(rows):
    """Print which mode won on val_rmse and pearson_r."""
    best_rmse = min(rows, key=lambda r: r["val_rmse"])
    best_pr   = max(rows, key=lambda r: r["pearson_r"])

    print(f"\n[Results] Best val RMSE  : {best_rmse['loss_mode']} "
          f"({best_rmse['val_rmse']:.4f})")
    print(f"[Results] Best Pearson r : {best_pr['loss_mode']} "
          f"({best_pr['pearson_r']:.4f})")


def save_csv(rows, path: str):
    """Write results DataFrame to CSV, creating parent dir if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "loss_mode", "val_rmse", "pearson_r", "final_loss", "epochs", "data_source"
    ])
    df.to_csv(path, index=False)
    print(f"\n[Saved] Results written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare three VAE loss modes on the CCLE QSAR IC50 task."
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs per loss mode (default: 10)."
    )
    parser.add_argument(
        "--no-pretrain", action="store_true",
        help="Skip loading pre-trained drug encoder weights."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: HP['batch_size']=32). Use 256 for faster runs."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use only 20k train samples per mode (~3-4 min total instead of 45-60 min)."
    )
    args = parser.parse_args()

    use_pretrained = not args.no_pretrain
    max_samples    = 20_000 if args.fast else 0
    bs             = args.batch_size

    print("=" * 55)
    print("  VAE Loss Mode Comparison — CCLE QSAR IC50 Task")
    print(f"  Epochs: {args.epochs}  |  Pretrained: {use_pretrained}")
    if args.fast:
        print("  Mode: --fast (20k samples, indicative results only)")
    if bs:
        print(f"  Batch size: {bs}")
    print("=" * 55)

    rows = run_comparison(epochs=args.epochs, use_pretrained=use_pretrained,
                          batch_size=bs, max_samples=max_samples)

    print_table(rows)
    declare_winners(rows)
    save_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
