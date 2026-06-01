"""
Test that BiIntDigitalTwin weights can be saved and reloaded correctly.

Usage:
    # After a training run that produced weights:
    python3 scripts/test_model_loading.py --weights logs/ldo_checkpoint/biint_ic50_model.weights.h5

    # Quick round-trip test (save + reload on random weights, no training needed):
    python3 scripts/test_model_loading.py --self-test

Exit code 0 = all checks passed.
Exit code 1 = loading failed (inspect stderr before launching a 30-min run).
"""

import argparse
import os
import sys
import tempfile

import numpy as np


def build_and_build_model():
    """Import fullPipeline, construct BiIntDigitalTwin, run one dummy forward pass."""
    # Add src/ to path so we can import fullPipeline
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch

    model = BiIntDigitalTwin(HP)

    # Build by running one dummy forward pass (required for subclassed models)
    dummy = generate_synthetic_ccle_batch(batch_size=2)
    ic50_out, kl_out = model(dummy[:-1], training=False)
    print(f"  [Build] OK — output shape {ic50_out.shape}, KL={float(kl_out):.4f}")
    print(f"  [Build] Trainable params: {model.count_params():,}")
    return model, HP, generate_synthetic_ccle_batch


def run_self_test():
    """Round-trip: build → save weights → reload → predict → compare."""
    print("\n=== SELF-TEST: save_weights / load_weights round-trip ===")
    model, HP, gen_batch = build_and_build_model()

    dummy = gen_batch(batch_size=4)
    pred_before, _ = model(dummy[:-1], training=False)
    pred_before = pred_before.numpy()
    print(f"  [Before] predictions: {pred_before[:4]}")

    with tempfile.TemporaryDirectory() as tmpdir:
        w_path = os.path.join(tmpdir, "test_weights.weights.h5")
        model.save_weights(w_path)
        print(f"  [Save] Weights written to {w_path} ({os.path.getsize(w_path)/1e6:.1f} MB)")

        # Reconstruct fresh model and load weights
        from fullPipeline import BiIntDigitalTwin
        model2 = BiIntDigitalTwin(HP)
        model2(dummy[:-1], training=False)  # build
        model2.load_weights(w_path)
        print(f"  [Load] Weights loaded into fresh model instance")

        pred_after, _ = model2(dummy[:-1], training=False)
        pred_after = pred_after.numpy()
        print(f"  [After] predictions: {pred_after[:4]}")

        max_diff = float(np.max(np.abs(pred_before - pred_after)))
        print(f"  [Check] Max prediction diff (should be ~0): {max_diff:.2e}")

        if max_diff < 1e-4:
            print("\n✅ SELF-TEST PASSED — save/load is numerically identical")
            return True
        else:
            print(f"\n❌ SELF-TEST FAILED — max diff {max_diff:.2e} exceeds threshold 1e-4")
            return False


def run_load_test(weights_path: str):
    """Load existing weights from a finished training run and run a forward pass."""
    print(f"\n=== LOAD TEST: {weights_path} ===")

    if not os.path.exists(weights_path):
        print(f"❌ File not found: {weights_path}")
        return False

    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"  [File] {size_mb:.1f} MB")

    # Try to load HP snapshot from same directory
    hp_path = os.path.join(os.path.dirname(weights_path), "hp_snapshot.json")
    hp_override = {}
    if os.path.exists(hp_path):
        import json
        with open(hp_path) as f:
            hp_override = json.load(f)
        print(f"  [HP] Loaded snapshot from {hp_path}")

    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from fullPipeline import BiIntDigitalTwin, HP, generate_synthetic_ccle_batch

    # Apply any saved HP overrides
    HP.update(hp_override)

    model = BiIntDigitalTwin(HP)
    dummy = generate_synthetic_ccle_batch(batch_size=4)
    model(dummy[:-1], training=False)  # build
    print(f"  [Build] Fresh model built ({model.count_params():,} params)")

    try:
        model.load_weights(weights_path)
        print(f"  [Load] Weights loaded successfully")
    except Exception as e:
        print(f"❌ load_weights failed: {e}")
        return False

    # Forward pass
    pred, kl = model(dummy[:-1], training=False)
    pred = pred.numpy()
    print(f"  [Inference] predictions: {pred}")
    print(f"  [Inference] KL loss: {float(kl):.4f}")

    if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
        print(f"\n✅ LOAD TEST PASSED — model predicts normally from saved weights")
        return True
    else:
        print(f"\n❌ LOAD TEST FAILED — NaN/Inf in predictions")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test BiIntDigitalTwin weight save/load")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to .weights.h5 file to test loading")
    parser.add_argument("--self-test", action="store_true",
                        help="Run round-trip save/load test without a pre-existing file")
    args = parser.parse_args()

    if not args.weights and not args.self_test:
        print("Running self-test (no --weights provided)...")
        args.self_test = True

    ok = True
    if args.self_test:
        ok = run_self_test() and ok
    if args.weights:
        ok = run_load_test(args.weights) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
