#!/usr/bin/env bash
# run_sequential.sh
# Runs LCO then LDO (with early stopping) sequentially on a single GPU.
# Usage: bash run_sequential.sh   (or: nohup bash run_sequential.sh > run_sequential.log 2>&1 &)

set -euo pipefail
cd "$(dirname "$0")"

source venv_tf/bin/activate

echo "============================================================"
echo " RUN 1 : Bi-Int Leave-Cell-Out (--early-stopping 3)"
echo " Started: $(date)"
echo "============================================================"
python3 fullPipeline.py \
  --mode pretrained \
  --loss-mode cross_entropy \
  --split-mode leave_cell_out \
  --epochs 10 \
  --early-stopping 3 \
  --log-dir logs/lco_run \
  --no-ppo \
  > logs_lco.txt 2>&1
LCO_EXIT=$?

echo ""
echo "============================================================"
if [ $LCO_EXIT -eq 0 ]; then
  echo " RUN 1 FINISHED OK at $(date)"
else
  echo " RUN 1 FAILED (exit $LCO_EXIT) at $(date)"
  echo " Check logs_lco.txt for details."
  echo " Attempting RUN 2 anyway..."
fi
echo "============================================================"
echo ""

# Brief pause to let GPU VRAM settle
sleep 10

echo "============================================================"
echo " RUN 2 : Bi-Int Leave-Drug-Out ES (--early-stopping 3)"
echo " Started: $(date)"
echo "============================================================"
python3 fullPipeline.py \
  --mode pretrained \
  --loss-mode cross_entropy \
  --split-mode leave_drug_out \
  --epochs 10 \
  --early-stopping 3 \
  --log-dir logs/ldo_run_es \
  --no-ppo \
  > logs_ldo_es.txt 2>&1
LDO_EXIT=$?

echo ""
echo "============================================================"
if [ $LDO_EXIT -eq 0 ]; then
  echo " RUN 2 FINISHED OK at $(date)"
else
  echo " RUN 2 FAILED (exit $LDO_EXIT) at $(date)"
  echo " Check logs_ldo_es.txt for details."
fi
echo "============================================================"
echo ""

echo "============================================================"
echo " BOTH RUNS DONE — running final comparison..."
echo "============================================================"
python3 scripts/final_comparison.py

echo ""
echo "All done at $(date)"
