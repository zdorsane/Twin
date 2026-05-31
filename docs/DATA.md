# Dataset Documentation

## Source

**CCLE (Cancer Cell Line Encyclopedia) — Broad Institute 2019**

- Publication: Barretina et al., *Nature* 2012 (original); Ghandi et al., *Nature* 2019 (update).
- Download: https://depmap.org/portal/download/ → Public 23Q4 or 2019 release.
- Licence: CC BY 4.0 (academic use, attribution required).

---

## Raw files (gitignored — download separately)

| File | Size | Description |
|------|------|-------------|
| `Dataset/ccle_broad_2019/CCLE_NP24.2009_Drug_data_2015.02.24.csv` | ~50 MB | IC50 matrix (drug × cell line) |
| `Dataset/ccle_broad_2019/CCLE_GEX.csv` | ~200 MB | Gene expression profiles (978 landmark genes, z-scored) |
| `Dataset/ccle_broad_2019/CCLE_CNA.csv` | ~80 MB | Copy number alterations (426 genes) |
| `Dataset/ccle_broad_2019/CCLE_Mutations.csv` | ~40 MB | Binary mutation matrix (735 genes) |
| `Dataset/chembl_36.sdf` | ~2.3 GB | ChEMBL 36 full compound library (pre-training only) |

---

## Dimensions after preprocessing

| Dimension | Value | Notes |
|-----------|-------|-------|
| Drugs (IC50 matrix) | 266 | Raw |
| Drugs with valid SMILES | 201 | 65 still missing (PubChem lookup incomplete) |
| Cell lines (omics+IC50) | 647 | Intersection GEx ∩ CNA ∩ IC50 |
| Valid (drug, cell, IC50) triplets | 103,477 | After removing NaN and IC50 ≤ 0 |
| Subsampled for training | 20,000 | Random seed 42, RAM/GPU constraint |
| GEx feature dims | 978 | Landmark genes, z-score normalised |
| CNA feature dims | 426 | log2(ratio+1) normalised |
| Mutation feature dims | 735 | Binary (0/1), sparsity 0.844 |
| IC50 transform | log1p(µM) | Mean 2.67, std 1.85 |

---

## Preprocessing pipeline

```
Raw IC50 (µM)
  → remove NaN, IC50 ≤ 0
  → log1p transform (log1p(IC50_µM))
  → z-score normalisation per cell line (mean=0, std=1)

GEx (RPKM)
  → subset to 978 landmark genes
  → z-score normalisation per gene across cell lines

CNA
  → log2(ratio + 1) per gene
  → z-score normalisation

Mutations
  → binary presence/absence matrix (1 = non-silent coding mutation)

Drug SMILES
  → PubChem REST API lookup (3-level: exact name, synonym, CID)
  → fallback: manual curation
  → BRICS featurization: atom features (12-dim) + bond features (5-dim)
  → GNN graph: max 50 atoms, self-loops
```

---

## Data splits

Three evaluation protocols are implemented in `src/fullPipeline.py`:

### Random split (80/20)
- All drugs and cell lines are randomly split between train and validation.
- **Optimistic**: the same drug often appears in both train and val.
- Use for: sanity check, convergence monitoring.

### Leave-Drug-Out (LDO)
- 171 train drugs, 30 validation drugs (randomly sampled, seed 42).
- No drug from the validation set is seen during training.
- **Honest**: evaluates generalisation to structurally novel drugs.
- Typical train size: ~16,832 triplets / val: ~3,168 triplets.

### Leave-Cell-Out (LCO)
- 518 train cell lines, 129 validation cell lines.
- No cell line from validation is seen during training.
- Use for: evaluating generalisation to new tumour profiles.

---

## Tracked CSV files (small, committed to repo)

| File | Rows | Description |
|------|------|-------------|
| `Dataset/ccle_drug_smiles.csv` | 402 | Drug name → SMILES mapping (201 unique drugs × 2 replicates) |
| `Dataset/baseline_results_with_CI.csv` | 16 | Classical baselines + Bi-Int results with 95% bootstrap CI |
| `Dataset/baseline_results_with_mutations.csv` | 12 | Ridge/RF/MLP results with mutation features included |
| `Dataset/graphga_tanimoto_vs_ccle.csv` | 10 | GraphGA top-10 candidates with Tanimoto vs CCLE drugs |
| `Dataset/brics_dqn_results.csv` | 5,000 | BRICS-DQN episode log (reward, validity, SMILES) |
| `Dataset/molecular_validation_report.csv` | 60 | Full MedChem validation (24 metrics per candidate) |

---

## SMILES mapping completeness

```
266 total CCLE drugs
  ├─ 201 mapped via PubChem (75.6%)
  │    ├─ isomeric SMILES: 198
  │    └─ canonical SMILES:   3
  └─  65 unmapped (24.4%)
       └─ Includes: AKTinhibitorVIII, GSK2126458, JNJ-26854165, ...
```

Unmapped drugs are excluded from QSAR training. Completing the mapping would
increase the training set by ~32% and improve LDO coverage.

---

## Chemical space coverage

The 201 CCLE drugs span a wide range of oncology targets:
- Kinase inhibitors (EGFR, PI3K, mTOR, ALK, CDK, etc.)
- HDAC inhibitors (Vorinostat, Belinostat, Entinostat, ...)
- DNA-damaging agents (Topotecan, Camptothecin, Gemcitabine)
- Targeted therapies (Erlotinib, Lapatinib, Imatinib)
- Classical chemotherapy (Paclitaxel, Vincristine, Bortezomib)

This diversity creates a challenging LDO task: new drugs from these classes
often have scaffold-hopped structures with limited Tanimoto similarity to
training compounds.
