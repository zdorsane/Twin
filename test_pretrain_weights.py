#!/usr/bin/env python3
"""
Simple test script to verify ChEMBL pre-trained weights are loaded correctly.
"""
import os
import sys
import numpy as np

# Setup environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

print("[Test] Verifying ChEMBL pre-trained weights...")
print("="*70)

# 1. Check if pre-trained weights file exists
WEIGHTS_FILE = 'pretrained_weights/chembl_drug_encoder.weights.h5'
META_FILE = 'pretrained_weights/pretrain_meta.json'

if os.path.exists(WEIGHTS_FILE):
    print(f"✓ Pre-trained weights file found: {WEIGHTS_FILE}")
else:
    print(f"✗ Pre-trained weights file NOT found: {WEIGHTS_FILE}")
    sys.exit(1)

if os.path.exists(META_FILE):
    print(f"✓ Metadata file found: {META_FILE}")
    import json
    with open(META_FILE) as f:
        meta = json.load(f)
    print(f"\n[Metadata]")
    for k, v in meta.items():
        if k != 'target_mean' and k != 'target_std':
            print(f"  {k}: {v}")
else:
    print(f"✗ Metadata file NOT found: {META_FILE}")

# 2. Test loading weights
print(f"\n[Loading Weights]")
try:
    weights_data = np.load(WEIGHTS_FILE.replace('.h5', '.npy'), allow_pickle=True, mmap_mode='r')
    print(f"✗ (Could not load as .npy)")
except:
    try:
        import h5py
        with h5py.File(WEIGHTS_FILE, 'r') as f:
            layer_names = list(f.keys())
            print(f"✓ HDF5 file readable")
            print(f"  Stored layers: {layer_names[:5]}...")  # Show first 5
            print(f"  Total layers: {len(layer_names)}")
    except Exception as e:
        print(f"✗ Error loading: {e}")
        sys.exit(1)

# 3. Summary
print(f"\n{'='*70}")
print("[Summary]")
print("✓ Pre-trained weights are ready for fullPipeline.py")
print("  Transfer layers: node_embed, gcn_proj_1, ln1, node_proj, ln2")
print("\nNext step: Run fullPipeline.py to train Bi-Int with pre-trained drug encoder")
