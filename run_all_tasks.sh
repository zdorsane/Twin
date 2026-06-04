#!/bin/bash
# ============================================================
# run_all_tasks.sh — Toutes les tâches critiques + importantes
# Lancer depuis le terminal WSL : bash run_all_tasks.sh
# Durée totale estimée : 6-8h GPU
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"
source venv_tf/bin/activate

LOG_DIR="logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "=== Logs dans : $LOG_DIR ==="

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"; }

# ────────────────────────────────────────────────────────────
# TÂCHE 1 — Ablation LDO (configs 1-4, baseline déjà fait)
# Durée estimée : ~3h GPU
# ────────────────────────────────────────────────────────────
log "TÂCHE 1 : Ablation LDO (early stopping + dropout + GNN freeze + 40k)"
python3 scripts/ldo_ablation.py --start-from 1 2>&1 | tee "$LOG_DIR/ldo_ablation.log"
log "TÂCHE 1 terminée."

# ────────────────────────────────────────────────────────────
# TÂCHE 2 — Run LCO final (15 epochs max, early stopping p=3)
# Durée estimée : ~2h GPU
# ────────────────────────────────────────────────────────────
log "TÂCHE 2 : Run LCO final (15 epochs, early stopping patience=3)"
python3 src/fullPipeline.py \
    --mode pretrained \
    --loss-mode cross_entropy \
    --split-mode leave_cell_out \
    --no-ppo \
    --epochs 15 \
    --early-stopping 3 \
    --log-dir logs/lco_final \
    --save-model \
    2>&1 | tee "$LOG_DIR/lco_final.log"
log "TÂCHE 2 terminée."

# ────────────────────────────────────────────────────────────
# TÂCHE 3 — MC Dropout recalibré (dropout=0.2, N=50, 300 paires)
# Durée estimée : ~45 min GPU
# ────────────────────────────────────────────────────────────
log "TÂCHE 3 : MC Dropout v2 (dropout=0.2, N=50, 300 paires)"
python3 scripts/uncertainty_mc_dropout_v2.py \
    --dropout-rate 0.2 \
    --n-samples 50 \
    --n-pairs 300 \
    2>&1 | tee "$LOG_DIR/mc_dropout_v2.log"
log "TÂCHE 3 terminée."

# ────────────────────────────────────────────────────────────
# TÂCHE 4 — Biomarqueurs par sous-type tumoral
# Durée estimée : ~15 min (réutilise cache GxI)
# ────────────────────────────────────────────────────────────
log "TÂCHE 4 : Biomarqueurs sous-types (breast / lung / haematological)"
python3 scripts/subtype_biomarker_analysis.py \
    --n-pairs 300 \
    2>&1 | tee "$LOG_DIR/subtype_biomarkers.log"
log "TÂCHE 4 terminée."

# ────────────────────────────────────────────────────────────
# TÂCHE 5 — ADMET in silico top-5 candidats
# Durée estimée : ~2 min (CPU, pas de GPU)
# ────────────────────────────────────────────────────────────
log "TÂCHE 5 : ADMET in silico top-5 candidats"
python3 scripts/admet_insilico.py \
    2>&1 | tee "$LOG_DIR/admet_insilico.log"
log "TÂCHE 5 terminée."

# ────────────────────────────────────────────────────────────
# Résumé final
# ────────────────────────────────────────────────────────────
log "=== TOUTES LES TÂCHES TERMINÉES ==="
log ""
log "Résultats produits :"
log "  Ablation LDO     : Dataset/ldo_improvement_ablation.csv"
log "                     figures/phase2_validation_ablation/07_ldo_ablation.png"
log "  LCO final        : logs/lco_final/val_curves.json"
log "                     logs/lco_final/training_log.csv"
log "  MC Dropout v2    : Dataset/uncertainty_mc_dropout_v2.csv"
log "                     figures/phase3_interpretability_reliability/12b_uncertainty_v2.png"
log "  Sous-types       : Dataset/subtype_biomarker_{breast,lung,haem}.csv"
log "                     figures/phase3_interpretability_reliability/14_subtype_biomarkers.png"
log "  ADMET in silico  : Dataset/admet_insilico_top5.csv"
log "                     figures/phase3_interpretability_reliability/15_admet_radar.png"
log ""
log "Log complet : $LOG_DIR/master.log"

echo ""
echo "Vérifier les résultats LDO ablation :"
python3 scripts/ldo_ablation.py --skip-runs 2>/dev/null || true
