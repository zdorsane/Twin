# Technical Notes — Twin Project

This file collects the technical details that have been removed from the public-facing `README.md`.

## What this project does
- Predicts CCLE IC50 from multimodal input: drug SMILES + cell line omics.
- Uses a GNN encoder for molecule representation and a quaternion VAE for omics fusion.
- Adds an optimization path for de novo molecule generation via BRICS-based DQN.

## Data sources
- CCLE Broad 2019 drug response dataset.
- PubChem lookup for SMILES mapping (cache + CSV + API).
- Omics modalities: gene expression (GEx), copy number alterations (CNA), mutations.

## Preprocessing
- IC50 values cleaned with NaN/inf filtering and values ≤ 0 removed.
- IC50 transformed by `log1p` and z-score normalized.
- Omics features selected by variance:
  - GEx top 978 genes
  - CNA top 426 features
  - Mut top 735 genes
- Drug mapping achieved for 201/266 CCLE compounds; remaining 65 drugs currently use placeholder vectors.

## Model architecture
- **Drug encoder**: GNN pretrained on ChEMBL.
- **Omics encoder**: UnifiedOmicsVAE with quaternion fusion (GEx+CNA+Mut).
- **Interaction block**: 4× bipartite interaction units with cross-attention.
- **Prediction head**: dense MLP (`256→128→64→1`) with dropout.

## Training and evaluation
- Training uses three split modes:
  - `random` (drug identity shared between train and val)
  - `leave_drug_out` (unseen drugs in validation)
  - `leave_cell_out` (unseen cell lines in validation)
- Current baseline comparison uses 20k subsampled triplets because of WSL RAM/GPU limits.
- EarlyStopping is enabled in `BiIntTrainer.fit()` with TensorBoard logging and CSV output.

## Observability
- TensorBoard events are written during training.
- A per-epoch CSV logger is emitted at `logs/run_gpu_main/training_log.csv`.
- Gradient norm (`L2`) and KL statistics are tracked.

## Known limitations
- 65 CCLE drugs still lack SMILES mapping.
- Random split results are optimistic; Leave-Drug-Out is the true generalization metric.
- Model currently evaluates on a 20k subsample of 103k available triplets.
- BRICS-DQN generation is preliminary: validity ~60.5% and aromaticity remains an open improvement.
- No license file is present in this repository.
