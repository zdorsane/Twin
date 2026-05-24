#!/bin/bash
export PATH=/home/crbt/anaconda3/envs/TwinCell/bin:$PATH
LOGFILE=/home/crbt/Twin/logs/run_ldo/ldo_run.txt
mkdir -p /home/crbt/Twin/logs/run_ldo
echo "[$(date)] Starting LDO run" > "$LOGFILE"
cd /home/crbt/Twin
python -u fullPipeline.py \
    --loss-mode cross_entropy \
    --epochs 5 \
    --no-ppo \
    --split-mode leave_drug_out \
    --log-dir /home/crbt/Twin/logs/run_ldo \
    >> "$LOGFILE" 2>&1
echo "[$(date)] EXIT: $?" >> "$LOGFILE"
