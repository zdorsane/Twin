#!/bin/bash
# Fix all figure paths after reorganization into phase subfolders
cd /home/crbt/Twin

FILES=(
  "README.md"
  "docs/FIGURE_INTERPRETATIONS.md"
  "docs/FIGURES_GUIDE.md"
  "docs/rapport_31mai2026.md"
  "scripts/ncrna_biomarker_analysis.py"
  "scripts/coding_biomarker_analysis.py"
  "scripts/uncertainty_mc_dropout.py"
  "scripts/applicability_domain.py"
  "scripts/ldo_ablation.py"
  "scripts/tanimoto_analysis.py"
  "scripts/molecular_validation.py"
)

# Phase 1
P1=(01_molecular_structures 02_training_curves 03_dqn_reward 04_qed_lipinski 05_dashboard \
    nb_01_ccle_summary nb_02_chembl_pretrain nb_03_qsar_random nb_04_qsar_ldo \
    nb_05_baselines nb_07_dqn_reward nb_08_dashboard)

# Phase 2
P2=(06_tanimoto_distribution 07_ldo_ablation 08_internal_diversity)

# Phase 3
P3=(09_ncrna_importance 10_ncrna_vs_drugs 11_coding_biomarkers \
    12_uncertainty_distribution 13_applicability_domain)

for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  for name in "${P1[@]}"; do
    sed -i "s|figures/${name}\.png|figures/phase1_training_generation/${name}.png|g" "$f"
  done
  for name in "${P2[@]}"; do
    sed -i "s|figures/${name}\.png|figures/phase2_validation_ablation/${name}.png|g" "$f"
  done
  for name in "${P3[@]}"; do
    sed -i "s|figures/${name}\.png|figures/phase3_interpretability_reliability/${name}.png|g" "$f"
  done
  echo "Updated: $f"
done
echo "All done."
