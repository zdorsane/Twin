#!/bin/bash
# ============================================================================
#  Entraînement uniquement — sans PPO/RL
#  Utilise le GPU si disponible
# ============================================================================
set -e

cd /home/crbt/Twin

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONWARNINGS=ignore

source venv_tf/bin/activate

echo "[Train] Démarrage de l'entraînement Bi-Int Digital Twin..."
echo "=========================================================================="

python3 -W ignore fullPipeline.py --mode pretrained --epochs 20 --no-ppo 2>/dev/null | tee train_only.log

echo "=========================================================================="
echo "[Train] Terminé ! Résultats dans train_only.log"
