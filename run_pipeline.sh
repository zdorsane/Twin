#!/bin/bash
cd /home/crbt/Twin
export PATH=/home/crbt/anaconda3/envs/TwinCell/bin:$PATH
LOGDIR="logs/run_gpu_main"
mkdir -p "$LOGDIR"
echo "[$(date)] Starting fullPipeline.py -> $LOGDIR" | tee -a run_log.txt
python fullPipeline.py --loss-mode cross_entropy --epochs 5 --no-ppo --log-dir "$LOGDIR" 2>&1 | tee -a run_log.txt
echo "[$(date)] Exit: ${PIPESTATUS[0]}" | tee -a run_log.txt
