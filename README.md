![Python](https://img.shields.io/badge/Python-3.12-blue.svg) ![License](https://img.shields.io/badge/license-Unspecified-lightgrey.svg) ![Last update](https://img.shields.io/badge/last%20update-May%2031%202026-orange.svg)

# Twin
Multimodal Drug Response Predictor on CCLE — IC50 prediction + de novo molecular generation

## TL;DR
- Predicts CCLE IC50 from drug SMILES and multimodal cell-line omics (GEx, CNA, mutations).
- Best performance: Pearson r = 0.811 (random split, optimistic) and **r = 0.316** (leave-drug-out, the honest generalisation metric).
- XGBoost outperforms the deep model on LDO: r = 0.367.
- 60 de novo molecules generated (GraphGA + BRICS-DQN): **38/60 pass all MedChem filters**, internal diversity = **0.90**.
- IC50 predictions for generated molecules are **extrapolated out-of-distribution** — not reliable; in vitro validation required.

---

## Key results

### IC50 prediction

| Split | Model | Pearson r | 95% CI |
|-------|-------|-----------|--------|
| Random | Bi-Int (epoch 4) | **0.811** | [0.736, 0.886] |
| Leave-Drug-Out | Bi-Int (epoch 2) | **0.316** | [0.287, 0.344] |
| Leave-Drug-Out | XGBoost | **0.367** | [0.338, 0.393] |
| Leave-Drug-Out | RF (50 trees) | 0.231 | [0.202, 0.259] |
| Leave-Drug-Out | Ridge (ECFP4+omics) | 0.228 | [0.196, 0.256] |
| Leave-Drug-Out | MLP (256→128) | 0.225 | [0.194, 0.255] |

> The random-split r = 0.811 is inflated (same drugs in train and test). The leave-drug-out r = 0.316 is the honest metric: the model must predict responses to structurally unseen drugs. XGBoost (r = 0.367) is currently the strongest generaliser.

### Molecular generation & validation (31 May 2026)

| Metric | Value |
|--------|-------|
| Candidates validated | 60 (10 GraphGA + 50 BRICS-DQN top-reward) |
| MedChem clean (all filters passed) | **38 / 60 (63%)** |
| PAINS alerts | 1 |
| Brenk alerts | 18 |
| Lipinski failures | 5 |
| Veber failures (rotbonds / TPSA) | 6 |
| SA > 6 (hard to synthesise) | **0** |
| Tanimoto > 0.7 vs CCLE drugs | 0 — no close analogues |
| Tanimoto < 0.3 (structurally novel) | **58 / 60** |
| Internal library diversity | **0.90** (mean Tanimoto 0.10) |

**Top 3 candidates (IC50-agnostic quality score = 0.30×QED + 0.25×SA_norm + 0.20×diversity + 0.25×medchem):**

| Rank | ID | SMILES | Score | QED | SA | Notes |
|------|----|--------|-------|-----|----|-------|
| 1 | BRI-46 | `O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1` | 0.925 | 0.903 | 1.89 | Naphthyl-sulfonamide piperazine |
| 2 | BRI-12 | `NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1` | 0.916 | 0.850 | 1.68 | Biaryl aminosulfonamide, easiest to synthesise |
| 3 | BRI-58 | `O=C1Nc2cccnc2N(CCO)c2ccccc21` | 0.900 | 0.858 | 2.27 | Tricyclic lactam, close to TrkA inhibitor GW441756 |

> ⚠️ **IC50 disclaimer:** Predicted IC50 values for generated molecules are extrapolated by a model with limited leave-drug-out performance (r = 0.316). These values must NOT be interpreted as reliable potency predictions. In vitro validation is required before any conclusion.

---

## Quick Start

```bash
git clone https://github.com/zdorsane/Twin && cd Twin
conda activate TwinCell
python fullPipeline.py --loss-mode cross_entropy --split-mode random --epochs 5
```

Run molecular validation on generated candidates:
```bash
python scripts/molecular_validation.py
```

---

## Architecture

```
SMILES ──► GNN encoder ─────────────────────────────┐
                                                     ├─► Bi-Interaction fusion ──► MLP ──► IC50
GEx + CNA + Mutations ──► Quaternion VAE encoder ───┘
```

**Molecular generation pipeline:**
```
BRICS fragments ──► DQN agent ──────────────────────┐
                                                     ├─► Top candidates ──► MedChem validation ──► Ranking
Graph molecules ──► Genetic Algorithm (GraphGA) ────┘
```

---

## Data

| Source | Size | Notes |
|--------|------|-------|
| CCLE Broad 2019 | 647 cell lines, 201/266 drugs, 103,477 valid IC50 triplets | log1p + z-score IC50 |
| PubChem SMILES | 184/201 drugs with valid SMILES | 17 drugs with empty/invalid SMILES |
| BRICS-DQN runs | 5,000 episodes, 3,008 valid molecules (reward > 0) | Validity ~60% |
| GraphGA | 10 top candidates | QED-optimised |

---

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/bootstrap_ci.py` | Bootstrap 95% CI on Pearson r | `Dataset/baseline_results_with_CI.csv` |
| `scripts/tanimoto_analysis.py` | Tanimoto similarity vs CCLE drugs | `Dataset/graphga_tanimoto_vs_ccle.csv`, `figures/06_tanimoto_distribution.png` |
| `scripts/smiles_augmentation.py` | Random SMILES enumeration for data augmentation | Module (call `augment_train_smiles()`) |
| `scripts/ldo_ablation.py` | Ablation study across 5 LDO improvement levers | `Dataset/ldo_improvement_ablation.csv`, `figures/07_ldo_ablation.png` |
| `scripts/molecular_validation.py` | Full MedChem validation of generated candidates | `Dataset/molecular_validation_report.csv`, `figures/08_internal_diversity.png` |

---

## Improvement roadmap (LDO r = 0.316 → target ≥ 0.40)

| Lever | Status | Expected gain |
|-------|--------|---------------|
| Early stopping (patience=3) | Implemented | Prevents overfitting past epoch 2 |
| Dropout 0.3 + L2 1e-4 | Implemented | Better out-of-distribution generalisation |
| SMILES augmentation (4× random SMILES per drug, train only) | Implemented | More robust GNN encoder |
| Full dataset (50k–100k triplets, vs current 20k) | Planned | Reduced selection bias |
| LDO ablation study (5 configs) | Script ready, GPU runs pending | Quantifies marginal gain per lever |
| SMILES mapping for 65 missing drugs | Planned | +30% dataset size |

---

## Limitations

- Leave-drug-out r = 0.316: weak but statistically significant. Predicted IC50 for novel drugs are unreliable.
- XGBoost outperforms Bi-Int on LDO: deep learning benefit is not yet justified at this data scale.
- BRICS-DQN validity ~60%: valence penalties in the reward function are a next step.
- SA score is a heuristic — actual synthetic difficulty should be verified with retrosynthesis tools (AiZynthFinder, ASKCOS).
- All 60 generated candidates have Tanimoto < 0.35 vs CCLE drugs: high novelty but high ADMET uncertainty.

---

## Reports

- [docs/rapport_31mai2026.md](docs/rapport_31mai2026.md) — Full session report with all results, interpretations, and next steps (31 May 2026)

---

## Citation + License + Contact

- Research prototype code for the Twin project.
- License: unspecified.
- Contact: open an issue on https://github.com/zdorsane/Twin
