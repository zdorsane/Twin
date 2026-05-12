#!/bin/bash
# ============================================================================
#  Run fullPipeline.py with pre-trained ChEMBL weights
# ============================================================================
set -e

cd /home/crbt/Twin

# Suppress TensorFlow/CUDA warnings before Python starts
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONWARNINGS=ignore

# Activate virtual environment
source venv_tf/bin/activate

echo "[Pipeline] Starting fullPipeline.py with pre-trained drug encoder weights..."
echo "=========================================================================="

# Run the pipeline — redirect stderr to suppress remaining C++ log messages
python3 -W ignore fullPipeline.py 2>/dev/null | tee fullPipeline_run.log

echo "=========================================================================="
echo "[Pipeline] Completed!"
