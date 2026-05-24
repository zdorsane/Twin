# Bi-Int — Multimodal Drug Response Predictor & Molecular Generator

> **An end-to-end pipeline for cancer drug response prediction (IC50) and de novo molecular generation,
> combining GNN pre-training on ChEMBL, multimodal omics VAE (GEx + CNA + mutations), and BRICS-DQN reinforcement learning — trained on real CCLE data.**
>
> **Note on terminology:** This project is sometimes described as a "digital twin" in the literature sense of a patient-specific computational model. More precisely, it is a **multimodal QSAR model** that predicts drug response from omics profiles, with a generative component for molecule design. The term "digital twin" is aspirational — full personalisation would require patient-specific sequencing data beyond CCLE.

---

## Session Log — What Was Done, What Works, What Remains

### Status at a glance (24 May 2026)

| Component | Status | Key result |
|-----------|--------|-----------|
| ChEMBL GNN pre-training | ✅ Complete | Val RMSE = 0.2187 (epoch 9/10), val loss = 0.0491 |
| CCLE data loading (P1–P3 fixed) | ✅ Complete | 647 cells, 201/266 drugs with SMILES, 103,477 triplets |
| Omics NPZ cache | ✅ Complete | `omics_cache_gex978_cna426.npz` — instant reload |
| BiInt training (corrected data) | ⏳ In progress | **Epoch 1: val RMSE = 0.846, Pearson r = 0.492** |
| Baseline models (Random split) | ✅ **NEW — first real results** | Ridge r=0.864, MLP r=0.881, XGB r=0.849 |
| Baseline models (Leave-Drug-Out) | ⏳ In progress | Results pending |
| Baseline models (Leave-Cell-Out) | ⏳ In progress | Results pending |
| BRICS-DQN generation | ✅ Complete | Best reward R=6.124, validity=60.5%, 5000 episodes |
| Repo cleanup | ✅ **NEW** | `dqn_weights_*/`, `logs/`, weight files excluded from git |
| "Digital twin" terminology | ✅ **NEW** | Reformulated as multimodal QSAR + honest note added |

---

## Session Updates — 21–24 May 2026

### Fixes applied 21 May 2026 (P1–P7)

Six prioritised fixes (P1–P6) implemented, plus P7 (OOM fixes):

**P1 — Real drug SMILES:** `BRICSMolecularFeaturizer` rewritten with real BRICS topology adjacency. SMILES lookup: 3-level cascade (pkl cache → CSV → live PubChem REST). **201/266 CCLE drugs now have real SMILES** (76%); 65 still missing.

**P2 — Mutation alignment:** Full `Tumor_Sample_Barcode` string match (not prefix). Sorted cell lines before index assignment. Shape assertions added. Resulting mutation matrix: **(647, 735), sparsity=0.844, mean 115 mutations/cell line**.

**P3 — IC50 validation + 3 split modes:** Diagnostic logging of IC50 distribution (range 0.0001–400,374 µM; log1p mean=2.67, std=1.85). Three rigorous split modes: `random`, `leave_drug_out` (unseen drug scaffolds), `leave_cell_out` (unseen cell lines). 103,477 valid triplets after SMILES filter.

**P4 — DQN reward:** SA score (Ertl & Schuffenhauer synthetic accessibility) integrated. Hard Lipinski penalty −2.0 (not soft deduction). Tanimoto CCLE diversity bonus.

**P5 — Baselines:** `baseline_models.py` implements Ridge / RF / MLP / XGBoost on ECFP4(2048) + GEx(978) + CNA(426) + Mut(735) = 4,187 features. R² + Pearson r + Spearman r reported per split.

**P6 — Observability:** `BiIntTrainer.fit()` logs TensorBoard events, CSVLogger (`training_log.csv`), gradient L2 norm per epoch, EarlyStopping (patience=5).

**P7 — OOM fixes:** GPU OOM (SelectV2 at `tf.maximum(kl_per_dim, free_bits)` in backward pass) fixed by switching to `--loss-mode cross_entropy`. CPU RAM OOM (22 GB peak from `np.stack()` on 6 modalities × 103k) fixed with: (a) sequential stack + immediate `del`, (b) 20k subsample via `np.random.default_rng(42).choice(n, 20000)`. NPZ omics cache added (`omics_cache_gex978_cna426.npz`) for instant reload.

---

### Fixes applied 24 May 2026 (P8–P9)

**P8 — `leave_drug_out` / `leave_cell_out` NameError bug:** Both split modes referenced `ic50_df.loc[]` in an O(n²) loop (201 drugs × 647 cells = 130k `.loc[]` calls) *after* `del ic50_df`. Replaced with vectorised `ic50_np[drug_row[drug_id]]` lookup — O(n) instead of O(n²). This caused the training process to appear "frozen" for 3+ days.

**P9 — Epoch-2 crash fix (`drop_remainder=True`):** After epoch 1 completed, the second epoch crashed inside `tf.reshape()` with `tf.reduce_prod(tensor_shape[axis[0]:])` — caused by the last (incomplete) batch having a dynamic `None` batch dimension. Fixed with `batch(batch_size, drop_remainder=True)` in `make_real_ds()`.

**Repo cleanup:** `.gitignore` updated to exclude `dqn_weights_*/`, `pretrained_weights/`, `logs/`, `run_log.txt`, `*.keras`. These are large binary/runtime files that do not belong in version control.

**Terminology fix:** README header reformulated from "Digital Twin" to "Multimodal Drug Response Predictor". Added honest note: the term "digital twin" is aspirational — the model is a multimodal QSAR predictor; full personalisation would require patient-specific sequencing.

---

## First Real Baseline Results (24 May 2026, Random Split)

These are the **first valid quantitative comparisons** between classical ML and the Bi-Int model on real CCLE data with corrected drug features and omics alignment.

| Model | Features | RMSE | R² | Pearson r | Spearman r |
|-------|----------|------|-----|-----------|------------|
| Ridge regression | ECFP4 + GEx + CNA | 0.508 | 0.746 | **0.864** | 0.859 |
| Ridge regression | GEx + CNA only | 0.971 | 0.070 | 0.265 | 0.254 |
| Random Forest (50 trees) | ECFP4 + GEx + CNA | 0.824 | 0.331 | 0.584 | 0.616 |
| MLP (512→256→128) | ECFP4 + GEx + CNA | 0.477 | 0.776 | **0.881** | 0.878 |
| XGBoost (100 trees) | ECFP4 + GEx + CNA | 0.548 | 0.704 | 0.849 | 0.846 |
| **Bi-Int (epoch 1)** | GNN + QuatVAE (GEx+CNA+Mut) | **0.854** | — | **0.506** | — |

**Split: Leave-Drug-Out and Leave-Cell-Out results in progress.**

### Scientific interpretation of Random Split results

**Why Ridge r=0.864 on a "simple" model?** This is a known property of CCLE: when drug identity is encoded as a fixed ECFP4 fingerprint, a linear model can memorize drug-specific mean IC50 values (some drugs are universally potent, others universally weak). The high r on random split reflects this memorization, not genuine structure–activity learning.

**The critical test is Leave-Drug-Out:** Here, the model must predict IC50 for drug scaffolds never seen during training. Ridge and RF are expected to drop sharply (r < 0.3) because they cannot generalize molecular structure knowledge. Bi-Int, with its GNN encoder pre-trained on 100k ChEMBL molecules, should generalize better.

**Bi-Int epoch 1 r=0.492 on random split:** This is below Ridge (0.864) after only 1 epoch — expected, since the model hasn't converged yet and the GNN encoder is still adapting from the ChEMBL pre-training domain (property prediction) to the CCLE domain (IC50 prediction). Pearson r is expected to improve substantially over epochs 2–5.

**Ridge (omics only) r=0.265:** When ECFP4 fingerprints are removed, a linear model on omics alone performs near-random. This confirms that the molecular structure signal is the dominant predictor in the random split — which is consistent with the memorization hypothesis above.

---

## Known Limitations (Honest Summary)

| Limitation | Status | Scientific impact |
|-----------|--------|------------------|
| 65/266 drugs missing SMILES | ❌ Pending | 24% of CCLE drugs excluded from training; PubChem found no match after 3-level lookup (pkl + CSV + REST) |
| Mutations file parsing error in baselines | ❌ Bug | `data_mutations.txt` has variable columns → `pd.read_csv` fails without `on_bad_lines='skip'`. Fixed in code, baseline re-run needed |
| BiInt training not converged | ⏳ In progress | Only epoch 1 completed (r=0.492); 5-epoch run in progress on GPU |
| Leave-Drug-Out / Leave-Cell-Out baseline results | ⏳ In progress | These are the scientifically meaningful splits — random split r inflated by drug memorization |
| Random split r inflated | ⚠️ Known issue | All models (incl. Ridge r=0.864) benefit from drug identity leakage in random split; not a fair measure of generalization |
| "Digital twin" claim | ⚠️ Terminology | Model is a QSAR predictor, not a patient-specific digital twin. Full personalisation requires patient sequencing data |
| No statistical significance testing | ❌ Missing | No confidence intervals or permutation tests on r values |
| CCLE vs. PDX generalization | ❌ Missing | All results are in-distribution CCLE; no external validation dataset used |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Steps & Results](#3-pipeline-steps--results)
   - [Step 1 — ChEMBL GNN Pre-training](#step-1--chembl-gnn-pre-training)
   - [Step 2 — QSAR Training on Real CCLE Data](#step-2--qsar-training-on-real-ccle-data)
   - [Step 3 — DQN Reinforcement Learning (full version history)](#step-3--dqn-reinforcement-learning-drug-generation)
4. [Detailed Metric Interpretation](#4-detailed-metric-interpretation)
5. [Dataset Description](#5-dataset-description)
6. [GPU Environment](#6-gpu-environment)
7. [Project Structure](#7-project-structure)
8. [Reproduction Commands](#8-reproduction-commands)
9. [RL Methods Comparison](#9-rl-methods-comparison)
10. [Known Limitations & Next Steps](#10-known-limitations--next-steps)
11. [References](#11-references)

---

## 1. Project Overview

This project implements a **Bipartite Intersite Interaction (Bi-Int) Digital Twin** for cancer precision medicine. The system predicts drug sensitivity (IC50) for cancer cell lines by jointly encoding:

- **Molecular structure** of drugs via a Graph Neural Network (GNN) pre-trained on 100,000 ChEMBL molecules using self-supervised multi-task regression
- **Multi-omics profiles** of cell lines (gene expression + copy-number alterations) via a Quaternion Variational Autoencoder (QuatVAE)
- **Drug–cell interaction** via 4 stacked Bi-Int blocks combining row-wise cross-attention, column-wise cross-attention, and triangular update layers, inspired by AlphaFold2

The trained model serves as a **reward oracle** for three reinforcement learning drug generators: PPO (Proximal Policy Optimization), GraphGA (evolutionary algorithm), and a Double DQN with SELFIES representation.

**Scientific domain:** Computational drug discovery · QSAR · Pharmacogenomics · Precision oncology · Deep RL · Molecular generation

---

## 2. System Architecture

```
─────────────────────────────────────────────────────────────────────────
  MOLECULAR ENCODING BRANCH
─────────────────────────────────────────────────────────────────────────

  Drug SMILES
       │
       ▼
  BRICS Fragmentation + Atom Feature Extraction
  (16-dim per atom: atomic num, degree, valence, aromaticity,
   hybridization, formal charge, H count, ring membership)
       │
       ▼
  Graph Neural Network (GNN) — 3-layer message passing
  ┌──────────────────────────────────────────────────────────┐
  │  node_embed   : Linear(16 → 64)                          │
  │  gcn_proj_1   : Linear(64 → 64) + ReLU  (message pass)  │
  │  ln1          : LayerNorm(64)                            │
  │  node_proj    : Linear(64 → 64) + ReLU  (message pass)  │
  │  ln2          : LayerNorm(64)                            │
  │  GlobalAvgPool ‖ GlobalMaxPool → concat → 128-dim        │
  │  proj_head    : Linear(128 → 8)   [pre-training only]    │
  └──────────────────────────────────────────────────────────┘
  Pre-trained on 100k ChEMBL molecules (self-supervised,
  targets: LogP, TPSA, MW, QED, HBD, HBA, NumRings, NumAromRings)
       │
       ▼
  Drug Embedding  D ∈ ℝ^(N_atoms × 64)   [transferred to QSAR]

─────────────────────────────────────────────────────────────────────────
  OMICS ENCODING BRANCH  (QuaternionVAE)
─────────────────────────────────────────────────────────────────────────

  GEx (978 genes, RPKM)   CNA (426 genes, copy number)
        │                         │
        ▼                         ▼
  Dense(978→256→128)       Dense(426→256→128)
        │                         │
        └──────────┬──────────────┘
                   ▼
  Quaternion Fusion Layer  (Hamilton product)
  → treats each 128-dim embedding as a quaternion (R, i, j, k components)
  → captures multiplicative cross-modal interactions impossible with concat
                   │
                   ▼
  VAE Bottleneck: μ ∈ ℝ^128, log σ² ∈ ℝ^128
  → z ~ N(μ, σ²I)   via reparameterization trick
  → KL loss = β · D_KL[q(z|x) ‖ p(z)]   β=2.0, free_bits=0.5
                   │
                   ▼
  Cell Embedding  z ∈ ℝ^128

─────────────────────────────────────────────────────────────────────────
  BI-INT INTERACTION BLOCKS  (× 4)
─────────────────────────────────────────────────────────────────────────

  Drug D ∈ ℝ^(N×64)   +   Cell z ∈ ℝ^128
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Row-wise Cross-Attention   (drug tokens → cell)    │
  │  Col-wise Cross-Attention   (cell → drug tokens)    │
  │  Triangular Multiplicative Update  (joint refine)   │
  │  LayerNorm + Residual connections                   │
  └─────────────────────────────────────────────────────┘  × 4
       │
       ▼
  Pooled joint representation  ∈ ℝ^256
       │
       ▼
  MLP Head: Dense(256→128→64→1)  + Dropout(0.1)
       │
       ▼
  IC50 prediction  ŷ ∈ ℝ  (normalized log µM)

─────────────────────────────────────────────────────────────────────────
  RL DRUG GENERATION (post-training, reward oracle = IC50 predictor)
─────────────────────────────────────────────────────────────────────────

       ├── PPO  (Proximal Policy Optimization — LSTM policy)
       ├── GraphGA  (Genetic Algorithm on molecular graphs)
       └── DQN  (Double Deep Q-Network — SELFIES v3.4)  ← this work
```

---

## 3. Pipeline Steps & Results

### Step 1 — ChEMBL GNN Pre-training

**Objective:** Initialize the drug encoder with chemically meaningful representations via self-supervised learning, before QSAR fine-tuning on IC50 data. This transfers molecular chemistry knowledge learned from 100k diverse structures, reducing the IC50 supervision signal required.

**Approach:** Multi-task regression — given the atom feature matrix and adjacency matrix of a molecule, predict 8 RDKit molecular descriptors simultaneously. No labels needed beyond the structure itself.

**Dataset:** ChEMBL 36 SDF (2,854,815 compounds, 7.4 GB). Filtered to 100,000 valid molecules satisfying:
- `mol.GetNumHeavyAtoms() ≤ 60`
- RDKit sanitization successful
- All 8 descriptors finite and computable

Accepted rate: 100,000 / 101,792 scanned = **98.2%**

**Target descriptors (normalized to zero mean, unit variance):**

| Descriptor | Physical meaning | Mean (raw) | Std (raw) |
|------------|-----------------|-----------|---------|
| MolLogP    | Lipophilicity (membrane permeability) | 3.34 | 2.14 |
| TPSA       | Topological polar surface area (absorption) | 81.73 Å² | 45.80 |
| MolWt      | Molecular weight | 392.85 Da | 125.96 |
| NumHDonors | H-bond donors (Lipinski criterion) | 1.76 | 1.60 |
| NumHAcceptors | H-bond acceptors (Lipinski criterion) | 4.69 | 2.23 |
| QED        | Quantitative drug-likeness [0,1] | 0.52 | 0.22 |
| NumRings   | Ring count (structural complexity) | 3.19 | 1.41 |
| NumAromaticRings | Aromatic ring count | 2.28 | 1.24 |

**Training results (GPU RTX 4000 Ada, 20,475 MiB VRAM, CUDA 13.0, TF 2.15.0, batch=64, lr=1e-3):**

| Epoch | Train RMSE | Val RMSE | Val MAE | Val Loss | LR |
|-------|-----------|---------|--------|---------|-----|
| 1     | 0.4875    | 0.3519  | 0.2451 | —       | 1e-3 |
| 2     | 0.3501    | 0.3177  | 0.2246 | —       | 1e-3 |
| 3     | 0.3107    | 0.2755  | 0.1904 | —       | 1e-3 |
| 4     | 0.2869    | 0.2627  | 0.1843 | —       | 1e-3 |
| 5     | 0.2687    | 0.2503  | 0.1747 | —       | 1e-3 |
| 6     | 0.2544    | 0.2306  | 0.1614 | —       | 1e-3 |
| 7     | 0.2436    | 0.2434  | 0.1794 | —       | 1e-3 |
| 8     | 0.2338    | 0.2322  | 0.1690 | —       | 5e-4 |
| 9     | 0.2140    | **0.2187** | **0.1552** | **0.0491** | 5e-4 |
| 10    | 0.2100    | 0.2244  | 0.1598 | 0.0503  | 5e-4 |

**Interpretation:**
- RMSE is on normalized targets → RMSE = 0.2187 means average error of ~22% of σ across all 8 descriptors simultaneously. For reference, RMSE = 1.0 is equivalent to predicting the mean (no learning).
- ReduceLROnPlateau triggered at epoch 8 (patience=2): LR halved 1e-3 → 5e-4, giving best val at epoch 9 (val_loss=0.0491, RMSE=0.2187, MAE=0.1552).
- Best checkpoint saved at epoch 9. Epoch 10 shows marginal overfit (+0.0012 val_loss vs epoch 9).
- **Transferred layers:** `node_embed`, `gcn_proj_1`, `ln1`, `node_proj`, `ln2` — the 5 GNN layers that encode molecular topology and atomic chemistry, directly reused in QSAR training.

**Weights saved:** `pretrained_weights/chembl_drug_encoder.weights.h5`

---

### Step 2 — QSAR Training on Real CCLE Data

**Objective:** Fine-tune the full Bi-Int model to predict drug IC50 on cancer cell lines using real pharmacogenomics measurements. This is the core prediction task: given a drug's molecular structure and a cell line's omics profile, predict sensitivity.

**Why real data matters:** Previous versions used synthetic (random) IC50 values. Using CCLE ground-truth data means the model learns genuine structure–activity relationships across 266 clinical/investigational drugs and 647 human cancer cell lines.

**Data loading pipeline (`load_ccle_real_data()` in `fullPipeline.py`):**

1. Load IC50 matrix: `data_drug_treatment_ic50.txt` (266 drugs × 1,068 cell lines, µM); float32 to reduce RAM
2. Load GEx: `data_mrna_seq_rpkm.txt` (56,319 genes × cell lines) → select **top 978 genes by variance** (L-1000 landmark gene space)
3. Load CNA: `data_cna.txt` (23,312 genes × cell lines) → select **top 426 genes by variance**
4. Load mutations: `data_mutations.txt` (MAF format, `comment='#', on_bad_lines='skip'`) → binary matrix **(647, 735)**, sparsity=0.844, mean 115 mutations/cell line
5. Align cell line IDs across IC50, GEx, CNA, mutations → **647 common cell lines** (sorted)
6. Build (drug_idx, cell_idx, IC50) triplets, drop NaN; SMILES filter keeps only drugs with real PubChem SMILES → **103,477 valid triplets** (201 drugs)
7. IC50 range: **0.0001–400,374 µM**; post-log1p: **mean=2.67, std=1.85**
8. IC50 transform: `log1p(max(IC50, 0.001))` → z-score (zero mean, unit variance)
9. Split: 85/15 stratified → **87,955 train | 15,522 val** (approx.)
10. Omics features cached to `Dataset/ccle_broad_2019/omics_cache_gex978_cna426.npz`

**Key implementation fixes (P1-P3):**
- Gene selection uses `sort_values(ascending=False).index[:n]` — `nlargest()` returned extra genes due to tied variance, causing shape mismatch in the GEx projector.
- Mutation alignment uses full `Tumor_Sample_Barcode` string match and sorted common cells; shape assertions added.
- SMILES lookup is a 3-level cascade (pkl cache → CSV → PubChem REST); 201/266 drugs resolved.

**Pre-trained weights loaded:** ChEMBL GNN encoder weights (epoch 9, val_loss=0.0491) transferred to `model.drug_gnn` before training.

**NOTE — previous results invalidated:** The Pearson r=0.884 (random split) and r=−0.35 (leave-drug-out) reported in earlier sessions were obtained with **random drug vectors and a zero mutation matrix**. These numbers do not reflect the corrected pipeline and should not be used for comparison. A corrected 20-epoch run at batch_size=16 (reduced from 32 due to GPU OOM) is currently in progress on the RTX 4000 Ada (20,475 MiB VRAM); results will be reported when complete.

**Previous (invalid) training results — random drug features, zero mutations, batch=32:**

| Epoch | Train RMSE | Val RMSE | KL Loss | Note |
|-------|-----------|---------|---------|------|
| 1     | 0.7754    | 0.5749  | 64.54   | Random drug features, zero mutations — INVALID |
| 5     | 0.5125    | 0.4943  | 64.00   | — |
| 10    | 0.4818    | 0.4847  | 64.00   | — |
| 15    | 0.4712    | 0.4720  | 64.00   | — |
| 20    | 0.4635    | 0.4723  | 64.00   | These numbers reflect omics memorization only |

**Corrected training results:** Pending — rerun in progress at batch_size=16 with 103,477 real-SMILES triplets + mutation features.

**Interpretation (previous run, for reference only):**

- **40% RMSE reduction** (0.775 → 0.464) over 20 epochs in the invalid run demonstrates the model can memorize omics signals, but without real drug features it cannot generalize to unseen drugs.
- **KL = 64.0** with latent_dim=128 → mean per-dimension KL = 0.5 nats = the `free_bits` threshold — the intended VAE operating point, confirmed in both runs.

**Biological significance:**
```
(Drug molecular graph) ⊗ (Gene Expression RPKM) ⊗ (Copy Number Alteration)
                                    ↓  QuatVAE + Bi-Int attention
                         Predicted IC50  (log µM)
```
The model learns which tumor transcriptomic/genomic subtypes respond to which drug chemical scaffolds — the core problem of **computational precision oncology**.

**Model saved:** 9,255,070 trainable parameters

---

### Step 3 — DQN Reinforcement Learning Drug Generation

**Objective:** Use the trained Bi-Int IC50 predictor as a reward oracle to generate novel drug candidates de novo, optimizing simultaneously for predicted anti-tumor potency, physicochemical drug-likeness, and chemical diversity.

---

#### 3.1 — Why char-level SMILES failed (v1 & v2)

The initial DQN formulations used character-level SMILES tokenization (33-token vocabulary: `C`, `N`, `O`, `(`, `)`, `1`–`9`, `=`, `#`, etc.). This approach has a fundamental flaw: **SMILES is a context-sensitive grammar**. A sequence of valid tokens does not necessarily constitute a valid SMILES string — unbalanced parentheses, unclosed rings, or illegal atom sequences produce strings that RDKit cannot parse.

| Version | Representation | Vocab | Valid % | Failure mode |
|---------|---------------|-------|---------|-------------|
| **v1** | char-level SMILES | 33 tok | ~1–3% | `<EOS>` token named `<END>` in vocab → decode always returned empty string |
| **v2** | char-level SMILES + action masking | 33 tok | 50% | Grammar learning impossible; best molecule = `P=P` (2 heavy atoms) |

**v2 analysis:** Despite 2,000 episodes and action masking (blocking `)` without open `(`, blocking early EOS), the agent converged to trivially valid molecules (`P=P`, `SSS`, `CCCCCCC`) because short repetitive sequences satisfy basic RDKit parsing while yielding non-zero reward from Lipinski MW compliance. Character n-gram patterns cannot capture the long-range dependencies inherent in SMILES ring closures and branch structures.

---

#### 3.2 — SELFIES representation (v3 and beyond)

**v3 replaces SMILES with SELFIES** (Self-Referencing Embedded Strings, Krenn et al. 2020):

> SELFIES is a molecular string representation where every token sequence maps to a valid molecule by construction. The grammar is context-free and semantically closed: each token specifies an atom or bond in a way that is always syntactically consistent, regardless of what precedes or follows it.

**Formal guarantee:** For any sequence of SELFIES tokens t₁t₂...tₙ, `selfies.decoder(t₁t₂...tₙ)` returns a valid, RDKit-parseable SMILES. This property holds without action masking or grammar enforcement.

**Consequence for RL:** The agent can explore the full token space freely. Every trajectory produces a valid molecule. Reward now signals chemical quality, not syntactic correctness — a fundamentally better learning signal.

---

#### 3.3 — MDP formulation (v3+)

| Component | v1/v2 (SMILES) | v3+ (SELFIES) |
|-----------|---------------|--------------|
| **State** s_t | z_omics ‖ one-hot(last_SMILES_char) ∈ ℝ^161 | z_omics ‖ one-hot(last_SELFIES_tok) ∈ ℝ^182–223 |
| **Action** a_t | Next SMILES char (33 choices) | Next SELFIES token (54→95 choices) |
| **Transition** | Append token to sequence | Append token to sequence |
| **Episode end** | `[EOS]` token or max_len=40 chars | `[EOS]` token or max_len=30 tokens |
| **Terminal reward** | Multi-objective (see below) | Multi-objective (see below) |
| **Intermediate reward** | 0.0 (terminal only) | 0.0 (terminal only) |
| **Validity guarantee** | None | **100% by construction** |

**Algorithm — Double DQN (van Hasselt et al. 2016):**
```
Two networks: Q_online (updated every step) and Q_target (hard-copied every 200 steps)

Double DQN target:
  y = r + γ · Q_target(s', argmax_a Q_online(s', a))
  (decouples action selection from value estimation → reduces overestimation bias)

Loss: Huber(y, Q_online(s, a))   [less sensitive to outlier rewards than MSE]
Optimizer: Adam(lr=3e-4)
Replay buffer: 20,000 transitions (uniform sampling)
Exploration: ε-greedy, ε: 1.0 → 0.05 linearly over 8,000 steps
```

**Q-Network architecture:**
```
Input: state ∈ ℝ^(128 + vocab_size)
  Dense(256, ReLU) → LayerNorm → Dense(256, ReLU) → Dropout(0.1)
  → Dense(128, ReLU) → Dense(vocab_size)   [Q-values for all actions]
```

---

#### 3.4 — Reward function evolution

**v3.0 reward (initial SELFIES):**
```
R = -0.5                              if mol invalid or empty
  = -0.2                              if n_heavy_atoms < 5
  + 2.5 × QED(mol)                   drug-likeness [0,1]
  + 0.5 × exp(-(logP-2.0)²/4)        lipophilicity gaussian (peak at logP=2)
  + 0.8                               if Lipinski Rule of 5 satisfied
  + 0.8 × exp(-(IC50-(-1.5))²/2)     IC50 potency gaussian (target: -1.5 log µM)
  + 0.4 × (1 - max_Tanimoto_sim)     diversity vs. previously accepted molecules
```

**v3.0 result — reward hacking observed (2000 episodes):**

| Episode | ε | Valid % | Mean R (last 50) | Best molecule | R_best |
|---------|---|---------|-----------------|---------------|--------|
| 1       | 0.996 | 100% | +1.971 | (random) | 1.971 |
| 250     | 0.050 | 100% | +1.222 | `[S+1]=[S+1]=[S+1]...` | 3.383 |
| 700     | 0.050 | 99.7% | +1.539 | `[S+1]#[S+1]=[S+1]...` | 3.580 |
| 2000    | 0.050 | 99.2% | +1.216 | polycyclic cumulene | **3.668** |

**Root cause of hacking:** RDKit's QED implementation assigns non-trivial scores to chemically pathological structures:
1. **Polysulfide chains** (`[S+1]=[S+1]=[S+1]...`): charged sulfur chains have balanced atom counts that score non-zero on QED's internal Gaussian desirability functions, despite having no pharmacological relevance.
2. **Cumulene skeletons** (`C=C=C=C=C=`): cumulated double bond chains satisfy Lipinski MW and have nonzero ring count when cyclized, but are chemically unstable (reactive intermediates, not isolable drugs).

The agent discovered these pathways during ε-greedy exploration and reinforced them because they reliably outscored the baseline.

---

**v3.1 fix — chemical filters added:**
```
+ carbon_penalty = -1.5  if C_count/n_heavy < 0.30  (eliminates inorganic hacks)
+ cumul_penalty  = -1.0  if count(=C=) ≥ 3          (eliminates cumulenes)
+ arom_bonus     = +0.8 × min(n_aromatic_rings, 3)/3 (steers toward benzene/indole scaffolds)
  qed_weight: 2.5 → 2.0
  lipinski_bonus: 0.8 → 1.0
```

**v3.1 result (2000 episodes, 54-token vocab from 25 seed SMILES):**
- Validity dropped to **~85%** — expected: harder constraints mean fewer hacks pass, but genuine drug-like molecules must now be discovered
- Dominant pattern shifted from polysulfides to **`[C@H1][C@H1][C@H1]...`** (repetitive stereocarbon chains)
- Root cause: 54-token vocab from 25 molecules is too small; agent memorizes token patterns instead of learning real scaffold diversity

---

**v3.2 — corpus upgrade + structural penalties:**

**Corpus upgrade:** 10,000 SMILES extracted directly from `chembl_36.sdf` using drug-likeness filters:
```python
filters: QED ≥ 0.3 | 8 ≤ heavy_atoms ≤ 40 | contains carbon | no metals/metalloids
         forbidden atoms: B, Al, Si, P, As, Se, Sn, Sb, Te, Pb, Bi
result:  10,000 accepted from 13,523 scanned  (74.0% acceptance rate)
```

**Vocabulary growth:**
| Version | Corpus | Vocab size | Stereo tokens |
|---------|--------|-----------|--------------|
| v3.0 | 25 seed SMILES | 54 tokens | ~4 |
| v3.2+ | **10,000 ChEMBL SMILES** | **95 tokens** | **11** |

The larger corpus exposes the vocabulary builder to real medicinal chemistry scaffolds: benzodiazepines, quinolines, piperazines, lactams, sulfonamides, etc.

**Three structural penalties (v3.2):**

| Penalty | Trigger | Formula | Rationale |
|---------|---------|---------|-----------|
| `size_penalty` | n_heavy > 25 | -0.15 × (n_heavy - 25) | Chains grow unbounded without this |
| `repeat_penalty` | same token > 4× | -0.4 × (max_count - 4) | Penalizes `[C@H1]` repeated 20× |
| `stereo_penalty` | stereo centers > 6 | -0.3 × (n_stereo - 6) | Pathological stereocenters |

**v3.2 bug — reward collapse (4% valid):** The repeat/stereo penalties combined with an early-return guard (`if rep_pen + stereo_pen < -1.5: return -2.0`) triggered for ~96% of episodes with the 95-token vocab (a single token repeated 5× exceeds the `max_token_repeat=4` threshold). This was a catastrophic signal collapse.

---

**v3.3 — rebalanced penalties, no early returns (fix for reward collapse):**

All penalties rewritten as soft deductions added to a unified `penalties` variable. No early return exists for structural penalties — every episode reaches full reward computation.

| Parameter | v3.2 | v3.3 | Rationale |
|-----------|------|------|-----------|
| `max_token_repeat` | 4 | 8 | 95-token vocab: a token repeated 5× is normal |
| `max_stereo_centers` | 6 | 8 | Allow natural drug stereocenters |
| `max_heavy_atoms` | 25 | 30 | Allow larger fragments |
| `repeat_penalty_coef` | 0.4 | 0.1 | Soft deduction |
| `stereo_penalty_coef` | 0.3 | 0.1 | Soft deduction |
| `size_penalty_coef` | 0.15 | 0.05 | Soft deduction |
| `carbon_penalty` | -1.5 | -0.5 | Soft deduction (was too harsh) |
| `cumul_penalty` | -1.0 | -0.5 | Soft deduction |

**v3.3 results (2000 episodes):**

| Metric | Value |
|--------|-------|
| Valid molecules | **1,750 / 2,000 (87.5%)** |
| Best reward | **3.618** |
| Best SMILES | `Cl[C+1]=[C+1]/S\Br` |
| Training stability | Stable (loss 0.01–0.02, no collapse) |
| ε at end | 0.050 |

**v3.3 diagnosis — formal charge exploitation:**
The best molecule `Cl[C+1]=[C+1]/S\Br` reveals a new hacking pattern: the agent discovered that **formal charges** (`[C+1]`) generate non-trivial QED scores in RDKit. Formal charges alter atom electronegativities in the Gaussian desirability functions used by QED, producing scores that do not reflect real drug-likeness. Additionally, halogens (Cl, Br) in combination with short chains create molecules that satisfy Lipinski MW while scoring well on LogP. These are chemically unrealistic: formally charged carbon atoms are not stable under physiological conditions.

---

**v3.4 — drug-likeness refinement:**

Three new penalty terms added directly to the `penalties` accumulator:

```python
# Formal charges: [C+1], [N+1], [O+1], [S+1] etc.
charged_atoms = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
if charged_atoms > 0:
    penalties -= charged_atoms * 0.4          # -0.4 per charged atom

# Isotope labels: [11C], [125I] — filtered from vocab too
if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
    penalties -= 0.5                           # fixed penalty

# Excess halogens: F=9, Cl=17, Br=35, I=53
n_halogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in {9, 17, 35, 53})
if n_halogens > 2:
    penalties -= (n_halogens - 2) * 0.3       # -0.3 per extra halogen
```

**Vocabulary filter:** Isotopic tokens (regex `\[\d+`) removed from `SELFIESVocabulary` at construction — vocab drops 95 → 91 tokens (4 isotopic tokens removed).

**v3.4 results (2000 episodes):**

| Metric | Value |
|--------|-------|
| Valid molecules | **1,443 / 2,000 (72.2%)** |
| Best reward | **3.484** |
| Best SMILES | `I/[C@@]/[C@H1]=C\I` |
| Mean reward (last 50 ep) | ~0.0 (marginal) |

**v3.4 diagnosis:** Charged-carbon and isotope exploits eliminated. New hacks: (1) **diiodo scaffold** — two iodine atoms sit exactly at `max_halogens=2`, combining high MW with non-zero QED; (2) **polyyne chains** — `C#CC#CC#CC#C` (5 triple bonds) evade detection because alkyne count was not penalized. Valid% declining trend indicates reward signal starting to weaken.

---

**v3.5 — halogen tightening + alkyne penalty:**

| Parameter | v3.4 | v3.5 | Rationale |
|-----------|------|------|-----------|
| `max_halogens` | 2 | 1 | Iodine pairs removed; 1 halogen is common in real drugs |
| `halogen_penalty_coef` | 0.3 | 0.5 | Stronger deterrent |
| `max_alkynes` | — | 1 | New: count C#C (carbon-carbon only); 1 alkynyl tolerated |
| `alkyne_penalty_coef` | — | 0.4 | -0.4 per excess C#C triple bond |

```python
# Polyyne penalty (new in v3.5)
n_alkynes = sum(1 for b in mol.GetBonds()
                if b.GetBondTypeAsDouble() == 3.0
                and b.GetBeginAtom().GetAtomicNum() == 6
                and b.GetEndAtom().GetAtomicNum() == 6)
if n_alkynes > max_alkynes:
    penalties -= (n_alkynes - max_alkynes) * alkyne_penalty_coef
```

**v3.5 also fixes the `SyntaxWarning: invalid escape sequence '\B'`** — docstring converted to raw string `r"""..."""`. TensorFlow/absl INFO logs suppressed via `TF_CPP_MIN_LOG_LEVEL=3` and `logging.getLogger("absl").setLevel(ERROR)`.

**v3.5 results (2000 episodes):**

| Metric | Value |
|--------|-------|
| Valid molecules | **1,254 / 2,000 (62.7%)** |
| Best reward | **2.354** |
| Best SMILES | `N/[C@@]\N/[C@@]\Br` |
| Mean reward (last 50 ep) | **−0.1 to −0.6 (negative)** |

**v3.5 diagnosis — reward starvation:** Mean reward turned consistently negative after ε-decay. The agent is learning to *minimize penalties* rather than *maximize drug-likeness*. Root cause: cumulative penalties now routinely subtract 0.5–1.5 from every molecule, while the maximum QED term is only +2.0. Without rings, a molecule that avoids all penalties still scores ≤ 1.5. The acyclic stereochain hack (`N/[C@@]\N/[C@@]\Br`) re-emerges because it has no rings (no `arom_bonus`), no halogens above threshold, no charges — the least-penalized structure, not the most drug-like.

---

**v3.6 — reward rebalancing + obligatory ring filter (current version):**

**Root cause fix:** Positive signal raised, acyclic structures explicitly penalized.

| Parameter | v3.5 | v3.6 | Rationale |
|-----------|------|------|-----------|
| `qed_weight` | 2.0 | **3.0** | Dominant positive signal: QED=0.7 → +2.1, above most penalty stacks |
| `acyclic_penalty` | — | **-0.6** | Any molecule with zero rings (aromatic or aliphatic) penalized immediately |
| `max_alkynes` | 1 | **0** | No C#C carbon-carbon triple bonds tolerated |
| `alkyne_penalty_coef` | 0.4 | **0.5** | Stronger deterrent |
| `max_halogens` | 1 | **2** | Relaxed back to avoid reward starvation |
| `halogen_penalty_coef` | 0.5 | **0.4** | Slightly reduced |
| `stereo_penalty_coef` | 0.1 | **0.05** | Avoid penalizing legitimate ring stereocenters |
| `max_stereo_centers` | 8 | **12** | Natural drug-like molecules can have many stereocenters |

**Full v3.6 reward function:**
```
R = -0.5                                      if SELFIES decode fails
  + -0.2                                      if n_heavy < 5
  + carbon_penalty   (-0.5)                   if C_frac < 25%
  + cumul_penalty    (-0.5)                   if count(=C=) ≥ 3
  + size_penalty                              if n_heavy > 30  (−0.05/atom)
  + repeat_penalty                            if max_rep_tok > 8  (−0.1/excess)
  + stereo_penalty                            if stereo_centers > 12  (−0.05/excess)
  + charge_penalty                            −0.4 × n_charged_atoms
  + isotope_penalty  (-0.5)                   if any isotope label
  + halogen_penalty                           −0.4 × max(0, n_halogens − 2)
  + alkyne_penalty                            −0.5 × n_C#C_bonds  (all penalized)
  + acyclic_penalty  (-0.6)                   if ring_info.NumRings() == 0
  + 3.0  × QED(mol)                           ← dominant signal
  + 0.5  × exp(-(logP-2.0)²/4)
  + 0.8  × min(n_arom_rings, 3)/3
  + 1.0                                       if Lipinski Rule of 5 satisfied
  + 0.8  × exp(-(IC50-(-1.5))²/2)
  + 0.4  × (1 - max_Tanimoto_sim)
  ∈ [-1.0, 10.0]
```

**Maximum achievable reward breakdown (v3.6, ideal drug-like molecule):**

| Term | Max value | Achieved when |
|------|----------|--------------|
| QED ×3.0 | +3.0 | QED = 1.0 (theoretical max) |
| LogP gaussian | +0.5 | LogP = 2.0 exactly |
| Aromatic bonus | +0.8 | ≥3 aromatic rings |
| Lipinski | +1.0 | MW≤500, HBD≤5, HBA≤10, LogP≤5 |
| IC50 | +0.8 | IC50 = -1.5 log µM exactly |
| Diversity | +0.4 | No similar molecule seen before |
| **Total** | **+6.5** | Neutral, cyclic, no alkynes, ≤2 halogens, Lipinski-compliant |

A typical drug-like molecule (QED≈0.7, 2 arom. rings, Lipinski pass): R ≈ 2.1 + 0.3 + 0.53 + 1.0 = **+3.9** — well above the acyclic stereochain baseline (~0.5).

---

#### 3.5 — Version comparison summary

| Version | Corpus | Vocab | Key change | Valid % | Best molecule | R_best | Mean R (end) |
|---------|--------|-------|-----------|---------|--------------|--------|-------------|
| v1 | 25 seed | 33 (SMILES) | Initial implementation | ~2% | *(empty — decode bug)* | — | — |
| v2 | 25 seed | 33 (SMILES) | Action masking | 50% | `P=P` (trivial) | 3.79 | — |
| v3.0 | 25 seed | 54 (SELFIES) | SELFIES representation | 99.2% | polysulfide/cumulene | 3.67 | +1.2 |
| v3.1 | 25 seed | 54 (SELFIES) | carbon + cumul + arom filters | ~85% | stereocarbon chains | ~3.5 | — |
| v3.2 | 10k ChEMBL | 95 (SELFIES) | corpus + size/repeat/stereo | **4%** | *(collapse)* | — | −2.0 |
| v3.3 | 10k ChEMBL | 95 (SELFIES) | soft penalties, no early return | **87.5%** | `Cl[C+1]=[C+1]/S\Br` | 3.618 | +0.7 |
| v3.4 | 10k ChEMBL | 91 (SELFIES) | charge + isotope + halogen | 72.2% | `I/[C@@]/[C@H1]=C\I` | 3.484 | ~0.0 |
| v3.5 | 10k ChEMBL | 91 (SELFIES) | max_halogens=1 + alkyne penalty | 62.7% | `N/[C@@]\N/[C@@]\Br` | 2.354 | **−0.3** |
| v3.6 | 10k ChEMBL | 91 (SELFIES) | qed×3 + acyclic_penalty | ~64% | stereochain + cyclopropane | ~2.7 | ~−0.3 |
| v3.7 | 10k ChEMBL | 91 (SELFIES) | nonarom_penalty, max_repeat=4 | 57.6% | `[C@H1][C@@][N+1]/O\I.[C@H1]I` | **2.649** | **−0.4** |
| v3.8 | 10k ChEMBL | 91 (SELFIES) | disconnect rejet, penalties×5 | 58.1% | `[C@]#SBr` | **−0.2** | −0.4 |
| v3.9 | 10k ChEMBL | 37 (SELFIES) | vocab blacklist (regex) | 67.2% | `C\CP#S\[C@]#N` | 3.137 | −0.4 |
| v3.10 | 10k ChEMBL | 50 (SELFIES) | vocab corpus-only whitelist | 66.8% | `Br\OC[OH0]/I` | 2.886 | −0.3 |
| v3.11 | 10k ChEMBL | 50 (SELFIES) | post-decode atom check (Cl/Br/I) | 57.6% | `[N@@]\S\S\N/F` | 2.424 | −0.4 |
| **v4.0** | 10k ChEMBL | ~45 (SELFIES) | F banned + reward shaping + 10k ep | 41.1% | `CC(=O)Nc1ccc(F)cc1` | **2.667** | ~0.0 (frozen ep.200) |
| **v5.0** | 10k ChEMBL | ~45 (SELFIES) | warm-start buffer + ε_min=0.15 | 60.4% | `C1CC(=O)NC(=O)N1` (acyclic) | **3.153** | +0.5 |
| **v5.1** | 10k ChEMBL | ~45 (SELFIES) | `[=Branch1]` reward fix (+0.20) | *pending* | *pending* | *pending* | *pending* |

**Trend analysis (v3.6–v5.1):** After v3.5, the critical insight was that the exploit source is not the *penalties* but the *vocabulary and SELFIES decoder*. Three structural issues were identified and fixed iteratively:

1. **Disconnected fragments** (`SMILES with '.'`) — SELFIES can generate multi-component SMILES; RDKit validates each component independently giving artificially high QED. Fixed in v3.8 with immediate −1.0 rejection.
2. **SELFIES decoder semantic drift** — even if a token (e.g. `[Br]`) is absent from the vocabulary, the SELFIES 2.x decoder can produce the corresponding atom from other token combinations. Vocab-level blacklisting alone is insufficient. Fixed in v3.11 with post-decode atomic number check.
3. **Reward starvation vs. reward exploitation trade-off** — aggressive penalties (v3.8: `nonarom=−2.0`, `charge=−2.0/atom`) push `Moy50` permanently negative, preventing the DQN from learning any positive signal. The 2000-episode budget is insufficient for convergence when the density of positive-reward states is very low. v4.0 addresses this with reward shaping (+0.03/aromatic token during episode) and 10,000 episodes.

---

## 4. Detailed Metric Interpretation

### 4.1 Pre-training RMSE (normalized space)

The 8 target descriptors are each normalized independently to (μ=0, σ=1) before training. Therefore:

- **RMSE = 1.0** → model predicts the mean for all samples (zero learning)
- **RMSE = 0.5** → error is half the standard deviation of each descriptor
- **RMSE = 0.208** (achieved) → average cross-descriptor prediction error ≈ 21% of σ. Strong for 8-task simultaneous regression on structurally diverse compounds.

The multi-task objective forces the GNN to encode features that generalize across molecular properties — LogP (lipophilicity), TPSA (polarity), MW (size), QED (global drug-likeness) — rather than overfitting to one.

### 4.2 IC50 RMSE (QSAR, normalized space)

IC50 is transformed as `log1p(max(IC50_µM, 0.001))` then z-scored. Raw CCLE IC50 values span 0.0001–400,374 µM; after log1p transformation the distribution has mean=2.67, std=1.85.

```
CCLE IC50 post-log1p std ≈ 1.85
Normalized RMSE = 0.47  (previous, invalid run with random drug features)
→ Absolute error ≈ 0.47 × 1.85 ≈ 0.87 log1p-µM
```

**Context:** Published multimodal QSAR models on CCLE (DeepDR, MOLI, tCNNs) report Pearson r ≈ 0.70–0.85 on held-out cell lines. The previous run used random drug vectors, which means the model learned omics → IC50 patterns only. Corrected results (103,477 triplets, real SMILES for 201/266 drugs, mutations included) are pending the current rerun at batch_size=16.

### 4.3 KL Loss (VAE)

```
KL divergence = β · Σ_d [ KL(q(z_d|x) ‖ N(0,1)) ]
             = 2.0 × 64.0 / 128 = 1.0 nat per dimension (before free_bits clamp)
After free_bits=0.5 clamp: effective KL ≈ 0.5 per dimension
Total KL ≈ 0.5 × 128 = 64.0  ✓
```

**Interpretation:** Each of the 128 latent dimensions carries exactly 0.5 nats of information about the cell line's omics profile. This is the intended operating point: the full latent capacity is used (no posterior collapse), and the regularization prevents memorization of cell line identities.

If KL < 10: posterior collapse (most dimensions unused). If KL > 100: VAE acts as autoencoder (overfits, no generalization). KL = 64.0 is in the healthy range.

### 4.4 DQN Reward Hacking — Diagnosis

Reward hacking is a well-documented failure mode in RL-based molecular generation (see Guimaraes et al. 2017, Olivecrona et al. 2017). The agent optimizes the proxy reward function, not the underlying scientific objective. Each version of the DQN revealed a new hacking strategy:

| Version | Hacking strategy | Root cause | Fix applied |
|---------|-----------------|-----------|-------------|
| v2 | `P=P`, `SSS` (trivial molecules) | Short sequences always valid; Lipinski MW satisfied | `min_heavy_atoms = 5` |
| v3.0 | `[S+1]=[S+1]=[S+1]...` (polysulfides) | RDKit QED non-zero for charged S chains | `carbon_frac ≥ 0.25` |
| v3.0 | `C=C=C=C=C=C=` (cumulenes) | Cyclic cumulenes pass Lipinski MW + ring count | `count(=C=) < 3` |
| v3.1 | `[C@H1][C@H1][C@H1]...` (stereo chains) | Repeated stereocarbon tokens fill max_len | `repeat_penalty` + `stereo_penalty` + `size_penalty` |
| v3.2 | *(reward collapse)* | Early-return guard fired for 96% of episodes | Remove early return; soften all coefficients |
| v3.3 | `Cl[C+1]=[C+1]/S\Br` (formal charges) | `[C+1]` alters QED Gaussian desirability values | `charge_penalty` −0.4/atom + isotope filter |
| v3.4 | `I/[C@@]/[C@H1]=C\I` (diiodo scaffold) | 2 iodines at exactly `max_halogens=2` threshold | `max_halogens=1`, `alkyne_penalty` added |
| v3.5 | `N/[C@@]\N/[C@@]\Br` (acyclic stereochain) | No rings → no `arom_bonus` deducted, fewest penalties | `acyclic_penalty=-0.6`, `qed_weight` 2.0→3.0 |
| v3.7 | `[C@H1][C@@][N+1]/O\I.[C@H1]I` (disconnected fragment) | SELFIES generates multi-component SMILES (dot notation); RDKit validates each fragment independently | Immediate −1.0 if `'.' in smiles` (v3.8) |
| v3.8 | `[C@]#SBr` (reward starvation, no positive signal) | `nonarom_penalty=−2.0` + `charge=−2.0/atom` → Moy50 permanently negative; agent learns nothing | Revert to soft penalties; filter vocab at source (v3.9) |
| v3.9 | `C\CP#S\[C@]#N` (triple-bond sulfur) | Regex blacklist removed `Ring`/`Branch` tokens → only 37 tokens, no aromatic cycles possible | Corpus-only whitelist (v3.10) |
| v3.10 | `Br\OC[OH0]/I` (Br/I despite vocab ban) | SELFIES 2.x decoder semantic substitution: absent tokens can appear via grammar expansion | Post-decode atomic number check `{17,35,53}` (v3.11) |
| v3.11 | `[N@@]\S\S\N/F` (fluorine exploit) | F (atomicNum=9) still in vocab; 5-atom organofluorine has QED=0.325 → +0.97 QED×3 term | F added to forbidden set `{9,17,35,53}` (v4.0) |
| v3.0–v3.11 | Best SMILES frozen after episode 50 | 2,000 episodes insufficient for DQN convergence when reward-positive states are sparse | 10,000 episodes + reward shaping +0.03/aromatic token (v4.0) |
| **v4.0** | All Top-5 acyclic (arom_rings=0) despite 93.5% corpus aromatic | SELFIES `[=Branch1]` token is the aromatic closure signal — DQN never receives +reward for it specifically; reward shaping targeted `[Ring1]` not `[=Branch1]` | Chirurgical fix: `[=Branch1]`/`[#Branch1]` → +0.20 reward (v5.1) |
| **v5.0** | Valid=60.4% but all Top-5 still acyclic | Warm-start buffer fills replay with seed SMILES trajectories but seed SMILES also encode `[=Branch1]` — low ε_min=0.15 keeps exploring without reinforcing the aromatic token specifically | Target `[=Branch1]` explicitly in step reward (v5.1) |

**General pattern:** Three distinct failure modes were identified across versions:

- **Reward exploitation** (v2–v3.7): the agent finds a structural shortcut that scores high on the proxy reward but is chemically meaningless. Each fix narrows the exploit space.
- **Reward starvation** (v3.2, v3.8): over-aggressive penalties push the entire reward landscape negative; the DQN gradient signal vanishes. Balance between positive terms and penalties is critical.
- **Vocabulary-level leakage** (v3.9–v3.11): SELFIES 2.x grammar allows token combinations to decode into atoms not present in the vocabulary. Lexical filtering alone is insufficient — validation must happen at the decoded SMILES level via RDKit atom checks.

### 4.5 Transfer Learning Effect

Loading ChEMBL pre-trained GNN weights before QSAR training provides:

| Layer | What it encodes | Transfer benefit |
|-------|----------------|-----------------|
| `node_embed` | Atomic environment: hybridization, aromaticity, charge | Avoids learning atom types from scratch on small IC50 data |
| `gcn_proj_1` + `ln1` | 1-hop neighborhood aggregation | Ring membership, local bonding patterns |
| `node_proj` + `ln2` | 2-hop neighborhood aggregation | Fragment-level features (phenyl, amine, carbonyl) |

Without pre-training, the drug encoder starts at random initialization and must learn molecular chemistry from IC50 supervision alone — a hard optimization with noisy labels and random drug features (current limitation). Pre-training provides a warm start aligned with molecular property distributions from 100k diverse ChEMBL structures.

---

## 5. Dataset Description

### CCLE (Cancer Cell Line Encyclopedia) — cBioPortal v2019

| File | Content | Raw dimensions | Dimensions used |
|------|---------|---------------|----------------|
| `data_drug_treatment_ic50.txt` | IC50 (µM), 1 drug × cell line per cell | 266 drugs × 1,068 cell lines | 266 × 647 common |
| `data_mrna_seq_rpkm.txt` | RNA-seq RPKM, 1 gene per row | 56,319 genes × cells | Top 978 by variance → **(647, 978)** |
| `data_cna.txt` | Copy number (discrete + continuous) | 23,312 genes × cells | Top 426 by variance → **(647, 426)** |
| `data_mutations.txt` | Somatic mutations (MAF format) | ~100k mutation records | Binary matrix **(647, 735)**, sparsity=0.844, mean 115 mutations/cell (**fix P2: `comment='#', on_bad_lines='skip'`, full barcode match**) |

Cell line naming: `CELLNAME_TISSUE` (e.g., `K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE`).
Modality alignment: sorted intersection of cell line IDs across IC50, GEx, CNA, mutations → **647 common cell lines**.
IC50 range: 0.0001–400,374 µM; post-log1p: mean=2.67, std=1.85.
Valid triplets after SMILES filter (201/266 drugs with real SMILES): **103,477**.
Omics cache: `Dataset/ccle_broad_2019/omics_cache_gex978_cna426.npz`.

Gene selection rationale:
- **978 GEx genes**: corresponds to the L-1000 landmark gene space used in CMap/LINCS — these high-variance genes capture the most transcriptional diversity and are biologically interpretable
- **426 CNA genes**: top-variance copy number genes enrich for known oncogenes (ERBB2, MYC, CCND1) and tumor suppressors (CDKN2A, RB1, TP53) with frequent amplifications/deletions

### ChEMBL 36

| Use | Filter | Size |
|-----|--------|------|
| GNN pre-training | ≤60 heavy atoms, RDKit-sanitizable, all 8 descriptors finite | 100,000 |
| DQN SELFIES vocabulary (v3.2) | QED≥0.3, 8–40 heavy atoms, carbon required, no metals | **10,000** |

Full ChEMBL 36 SDF: 2,854,815 compounds, 7.4 GB, stored at `Dataset/chembl_36.sdf`.

---

## 6. GPU Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4000 Ada Generation |
| VRAM | 20,475 MiB |
| CUDA | 13.0 (runtime, WSL2 CUDA passthrough) |
| TensorFlow | 2.15.0 |
| SELFIES | 2.1.1 |
| RDKit | 2024.x |
| Strategy | `tf.distribute.MirroredStrategy` (1 replica) |
| OS | Ubuntu 24.04 LTS (WSL2 on Windows 11 Pro) |
| Python | 3.11 |
| Conda env | `TwinCell` (anaconda3) |

**Note on batch size:** BiInt training with batch_size=32 triggers a `SelectV2 ResourceExhaustedError` on this GPU despite 20,475 MiB VRAM. Reduced to batch_size=16 for the corrected run.

---

## 7. Project Structure

```
Twin/
├── fullPipeline.py              # Bi-Int model, QuatVAE, CCLE loader, QSAR training, PPO
│                                #   --loss-mode kl|cross_entropy|both  (CE wins: r=0.713)
│                                #   --beta-anneal  (linear β: 0→2.0 over 10 epochs)
│                                #   load_ccle_real_data: loads real drug SMILES if CSV present
│                                #   MAF parser fixed (comment='#', on_bad_lines='skip')
├── chembl_pretrain.py           # GNN self-supervised pre-training on ChEMBL 100k
├── dqn_optimizer.py             # Double DQN — SELFIES v5.1 (current)
│                                #   v5.1: [=Branch1] step reward +0.20 (aromatic closure fix)
│                                #   v5.0: warm-start buffer, ε_min=0.15
│                                #   v4.0: 10k episodes, F/Cl/Br/I banned, reward shaping
├── brics_dqn_optimizer.py       # BRICS fragment-based DQN (NEW — structural solution)
│                                #   BRICSVocabulary (freq≥5 fragments from ChEMBL 10k)
│                                #   BRICSDQNOptimizer with brics_success_bonus, frag_diversity
├── compare_vae_losses.py        # Runs kl/cross_entropy/both × N epochs, saves CSV
│                                #   --fast (20k), --batch-size, --split-mode random|leave_drug_out
├── fetch_drug_smiles.py         # Maps 266 CCLE drug names → SMILES via PubChem REST API
│                                #   output: Dataset/ccle_drug_smiles.csv (~90 sec, 266 drugs)
├── graphga_biint_optimizer.py   # GraphGA evolutionary drug optimizer
├── reinvent_biint_optimizer.py  # REINVENT-style conditional policy gradient
├── reinvent_optimizer.py        # Simplified REINVENT
├── inference.py                 # Inference API wrapper
├── api_server.py                # FastAPI REST server
├── smiles_tokenizer.json        # Legacy SMILES vocabulary (33 tokens, PPO only)
├── pretrained_drug_encoder.keras       # Full Keras model (ChEMBL pre-trained encoder)
├── pretrained_weights/
│   ├── chembl_drug_encoder.weights.h5  # Transferred GNN weights
│   └── pretrain_meta.json              # Descriptor normalization stats + training metadata
├── dqn_weights_v3/              # v3.x SELFIES DQN weight snapshots
├── dqn_weights_v4/              # v4.0 (10k episodes, r=2.667, all acyclic)
├── dqn_weights_v5.0/            # v5.0 (warm-start, r=3.153, still acyclic)
├── dqn_weights_v5.1/            # v5.1 (pending — aromatic token fix)
├── archive/                     # Obsolete scripts kept for reference
├── notebooks/
│   └── evaluation.ipynb         # Training curves, GraphGA visualization, DQN version comparison
├── logs_chembl.txt              # ChEMBL pre-training full log (10 epochs)
├── logs_dqn.txt / logs_dqn_v5.1.txt / logs_brics_dqn.txt   # DQN run logs
├── reports/
│   ├── 2026-05-17_session_report.md    # Full session report (DQN v3.4→v4.0)
│   └── reviewer_response_2026-05-18.md # Response to reviewer feedback
├── COMMANDES.md                 # Step-by-step Ubuntu execution commands
├── Dataset/
│   ├── chembl_36.sdf            # ChEMBL 36 full SDF (2.85M molecules, 7.4 GB — gitignored)
│   ├── ccle_drug_smiles.csv     # CCLE drug → SMILES mapping (produced by fetch_drug_smiles.py)
│   ├── vae_loss_comparison.csv  # compare_vae_losses.py results (kl/CE/both × splits)
│   └── ccle_broad_2019/         # CCLE cBioPortal v2019 (IC50, GEx, CNA, mutations)
└── README.md
```

---

## 8. Reproduction Commands

```bash
# Open Ubuntu WSL
wsl -d Ubuntu
cd ~/Twin && source venv_tf/bin/activate
```

### Step 1 — ChEMBL GNN Pre-training
```bash
nohup python3 chembl_pretrain.py > ~/Twin/logs_chembl.txt 2>&1 &
tail -f ~/Twin/logs_chembl.txt
# ~15 min on RTX 4000 | 100k molecules | 10 epochs | best val RMSE: 0.208
```

### Step 1b — Map CCLE Drug SMILES via PubChem (required for generalization)
```bash
python3 fetch_drug_smiles.py
# ~90 seconds | 266 drugs | output: Dataset/ccle_drug_smiles.csv
# Maps drug names → canonical SMILES; automatically used by load_ccle_real_data()
```

### Step 2 — QSAR Training on Real CCLE Data
```bash
# Option A: cross-entropy loss (best on random split — Pearson r=0.713)
python3 fullPipeline.py --loss-mode cross_entropy --no-ppo
# Option B: both losses + β-annealing (best expected generalization)
python3 fullPipeline.py --loss-mode both --beta-anneal --no-ppo --epochs 20
# Loads ChEMBL weights + CCLE data + real drug SMILES → 20 epochs IC50
# ~10 min on RTX 4000 | 137k triplets
```

### Step 2b — VAE Loss Mode Comparison
```bash
# Fast run (20k samples, ~5 min)
python3 compare_vae_losses.py --epochs 5 --fast --batch-size 256
# True generalization test (leave-drug-out split)
python3 compare_vae_losses.py --epochs 5 --fast --batch-size 256 --split-mode leave_drug_out
# Full run (137k samples, ~45-60 min)
python3 compare_vae_losses.py --epochs 10
```

### Step 3 — Full Pipeline with PPO drug generation
```bash
python3 fullPipeline.py --loss-mode cross_entropy --epochs 20
```

### Step 4a — DQN Drug Generation (SELFIES v5.1 — chirurgical aromatic fix)
```bash
nohup python3 dqn_optimizer.py > ~/Twin/logs_dqn_v5.1.txt 2>&1 &
tail -f ~/Twin/logs_dqn_v5.1.txt
# v5.1: [=Branch1] step reward +0.20, warm-start 500 ep, ε_min=0.15
# 5,000 episodes | ~20-30 min on RTX 4000
```

### Step 4b — BRICS DQN (fragment-based, structural solution)
```bash
nohup python3 brics_dqn_optimizer.py > ~/Twin/logs_brics_dqn.txt 2>&1 &
tail -f ~/Twin/logs_brics_dqn.txt
# Fragment assembly MDP | 5,000 episodes | ~20-30 min
# output: Dataset/brics_dqn_results.csv
```

### Step 5 — GraphGA Optimization
```bash
python3 graphga_biint_optimizer.py
```

### Monitor GPU
```bash
nvidia-smi
ps aux | grep python | grep -v grep
```

---

## 9. RL Methods Comparison

| | PPO | REINVENT | GraphGA | Double DQN v4.0 (this work) |
|---|---|---|---|---|
| **Paradigm** | Policy gradient (on-policy) | Policy gradient (off-policy KL) | Evolutionary | Q-learning (off-policy) |
| **Mol. representation** | char-level SMILES (LSTM) | char-level SMILES | Molecular graph | **SELFIES tokens** |
| **Memory** | None (on-policy) | None | Population (50 mol.) | Replay buffer (20k transitions) |
| **Exploration** | Entropy regularization | Temperature annealing | Mutation + crossover | ε-greedy: 1.0 → 0.05 / 20k steps |
| **Validity guarantee** | ~70% (pre-trained policy) | ~80% (pre-trained) | ~90% | **~100% (SELFIES grammar)** |
| **Stability** | High variance, collapse ~ep.70 | Medium variance | High | Target network prevents overestimation; reward starvation risk if penalties > positive signal |
| **Reward signal** | IC50 + validity | IC50 × validity | IC50 + QED + SA + Lipinski | IC50 + QED×3 + LogP + Lipinski + arom + diversity + reward shaping (v4.0) |
| **Reward hacking** | `P=P` type trivial mols | — | Limited (graph operators) | 12 distinct exploits identified v2–v3.11; root causes: (1) soft penalties, (2) SELFIES decoder semantic drift, (3) insufficient episodes for convergence |
| **Best honest result** | Partial valid SMILES | — | QED: 0.71–0.93, MW: 269–347 Da | v4.0: 41.1% valid, R=2.667 (all acyclic); v5.0: 60.4% valid, R=3.153 (still acyclic); v5.1 pending |
| **Corpus for generation** | 25 seed SMILES | — | — | **10,000 ChEMBL SMILES (QED≥0.3)** |
| **Key diagnosis** | — | — | — | SELFIES `[=Branch1]` token = aromatic closure; no reward assigned → arom_rings=0 across all versions v3–v5. Fix: v5.1 +0.20 step reward; structural fix: BRICS DQN |

---

## 10. Known Limitations & Next Steps

### Current Limitations

| Issue | Status | Root cause | Scientific impact |
|-------|--------|-----------|------------------|
| 65/266 CCLE drugs still missing SMILES | Open | No PubChem match after 3-level lookup | ~24% of drugs use fallback zero vectors |
| BiInt training results not yet available | In progress | GPU OOM at batch_size=32 → fixed to 16, rerun running | Corrected Pearson r (random/leave-drug-out/leave-cell-out) pending |
| Leave-drug-out result with corrected data unknown | Pending BiInt rerun | Previous r=−0.35 was invalid (random vectors + zero mutations) | Cannot claim generalization until rerun completes |
| Baseline comparison table not yet run | Pending | `baseline_models.py` not yet executed | R²/Pearson r comparison row for Ridge/RF/MLP/XGBoost missing |
| SELFIES DQN arom_rings=0 (v3–v5) | Fix in v5.1 — `[=Branch1]` step reward +0.20 | `[=Branch1]` is SELFIES aromatic closure token; DQN never reinforced it | BRICS DQN is the structural solution; v5.1 is a chirurgical fix |
| ChEMBL pre-training on 100k / 2.8M | Open | WSL2 RAM limit (OOM at 500k) | GNN sees only 3.5% of available chemical space |
| DQN uses synthetic IC50 oracle | Open | Real BiInt model not yet stable enough to serve as oracle | Generated molecules optimized against a proxy reward |
| GitHub push blocked | Pending | Old tokens compromised; new token needed | Latest commits not yet on remote |

### Prioritized Next Steps

1. **[Immediate] Wait for BiInt 20-epoch rerun to complete** — batch_size=16 on 103,477 real-SMILES triplets + mutations
   ```bash
   tail -f ~/Twin/run_log.txt
   ```
2. **[Critical] Run baseline comparison** — establishes R² floor for Ridge/RF/MLP/XGBoost
   ```bash
   python3 baseline_models.py
   ```
3. **[Critical] Evaluate leave-drug-out and leave-cell-out splits** — true generalization test on corrected data
   ```bash
   python3 compare_vae_losses.py --epochs 10 --split-mode leave_drug_out
   python3 compare_vae_losses.py --epochs 10 --split-mode leave_cell_out
   ```
4. **[High] Run DQN v5.1** — test chirurgical `[=Branch1]` aromatic fix; expected first Top-5 aromatic molecule
   ```bash
   nohup python3 dqn_optimizer.py > ~/Twin/logs_dqn_v5.1.txt 2>&1 &
   ```
5. **[High] Run BRICS DQN** — structural solution to SELFIES aromatic failure; fragments are drug-like by construction
   ```bash
   nohup python3 brics_dqn_optimizer.py > ~/Twin/logs_brics_dqn.txt 2>&1 &
   ```
6. **[High] Set up new GitHub token and push** — commits d503903–present not yet on remote
7. **[Medium] Full pipeline with CE + β-annealing** — train 20 epochs on corrected 103k triplets
   ```bash
   python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 --no-ppo
   ```
8. **[Medium] Resolve remaining 65 missing SMILES** — manual SMILES curation or alternative name lookup
9. **[Future] Contrastive learning** — SimCLR/NT-Xent on drug embeddings; bypasses VAE latent space limitations
10. **[Future] Scale ChEMBL pre-training** to 500k–1M molecules via streaming SDF reader with HDF5 feature cache
11. **[Future] Transformer Q-network** — replace 3-layer MLP with causal Transformer; captures long-range SELFIES token dependencies
12. **[Future] Multi-task IC50** — add GI50, AUC, Z-score heads to MLP prediction head

---

## 11. References

1. Mnih et al., *"Human-level control through deep reinforcement learning"*, Nature 2015
2. Van Hasselt et al., *"Deep Reinforcement Learning with Double Q-learning"*, AAAI 2016
3. Olivecrona et al., *"Molecular de novo design through deep reinforcement learning"*, J. Cheminformatics 2017
4. Guimaraes et al., *"Objective-Reinforced Generative Adversarial Networks (ORGAN)"*, arXiv 2017
5. Jumper et al., *"Highly accurate protein structure prediction with AlphaFold"*, Nature 2021
6. Partin et al., *"Multitask drug-cell interaction learning"*, Scientific Reports 2023
7. Bickerton et al., *"Quantifying the chemical beauty of drugs"* (QED), Nature Chemistry 2012
8. Ertl & Schuffenhauer, *"Estimation of synthetic accessibility score"*, J. Cheminformatics 2009
9. Lipinski et al., *"Experimental and computational approaches to estimate solubility and permeability"*, Adv. Drug Delivery Reviews 2001
10. Barretina et al., *"The Cancer Cell Line Encyclopedia enables predictive modelling"*, Nature 2012
11. Krenn et al., *"Self-Referencing Embedded Strings (SELFIES)"*, Mach. Learn.: Sci. Technol. 2020
12. Kingma & Welling, *"Auto-Encoding Variational Bayes"*, ICLR 2014
13. Higgins et al., *"β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"*, ICLR 2017

---

*All experiments run on Ubuntu 24.04 LTS (WSL2) with NVIDIA RTX 4000 Ada (20,475 MiB VRAM), CUDA 13.0, TensorFlow 2.15.0, SELFIES 2.1.1, RDKit 2024, Python 3.11, conda env TwinCell.*
*Source code: `fullPipeline.py` (model + CCLE loader + P1-P3 fixes), `chembl_pretrain.py` (GNN pre-training), `dqn_optimizer.py` (DQN v5.1), `baseline_models.py` (P5 baselines).*
*Last updated: 21 May 2026.*

---

## Reviewer Feedback — Implementations & Results (2026-05-18)

Based on expert reviewer feedback: *"try BRICS tokenization and cross-entropy as loss function — you will get better results than KL divergence".*

---

### Cross-Entropy VAE Loss — Experimental Results

**Motivation:** Binary cross-entropy (BCE) reconstruction is more appropriate than implicit MSE for omics data — especially for sparse mutation features (binary 0/1) and bounded gene expression profiles. BCE penalizes systematic reconstruction errors proportionally to feature sparsity, exposing more biologically meaningful latent dimensions.

**Three modes implemented** in `fullPipeline.py` via `--loss-mode`:

| Mode | Reconstruction | Regularization | Description |
|------|---------------|---------------|-------------|
| `kl` (original default) | Implicit MSE | β·KL (β=2.0) | Baseline — unchanged |
| `cross_entropy` | Binary CE on min-max normalized omics | None | Pure autoencoder |
| `both` | Binary CE + MSE | β·KL | Strongest regularization |

#### Random split results (20k samples, 5 epochs, `--fast --batch-size 256`)

| Mode | Val RMSE | Pearson r | Interpretation |
|------|----------|-----------|----------------|
| `kl` (baseline) | 0.848 | 0.546 | β=2.0 too strong — crushes IC50 signal |
| `cross_entropy` | **0.702** | **0.713** | ✅ Best — latent space encodes pharmacological correlations |
| `both` | 0.747 | 0.689 | KL + CE combined — middle ground |

**Scientific interpretation — why CE wins on random split:**
- β=2.0 KL forces all embeddings toward N(0,I), erasing drug-specific and cell-line-specific structure
- BCE reconstruction acts as a rich autoencoder: the latent space is free to encode true pharmacological correlations (which genes respond to which chemical scaffolds)
- Pearson r=0.713 is **competitive with published multimodal CCLE models** (0.65–0.75 range) on 20k samples / 5 epochs

#### Leave-drug-out split results (true generalization test — no drug shared between train and val)

| Mode | Val RMSE | Pearson r | Drop vs random |
|------|----------|-----------|---------------|
| `kl` | 1.369 | **−0.354** | −0.900 |
| `cross_entropy` | 1.357 | **−0.327** | −1.040 |
| `both` | **1.181** | **−0.125** | **−0.814** (smallest drop) |

**Critical finding — the model memorizes, it does not generalize:**

Pearson r turns **negative** on unseen drugs — the model predicts IC50 in the wrong direction for drugs never seen during training. This is not noise: it is active counter-generalization. The root cause is that **drug features are currently random vectors** (`np.random.randn`). The model has memorized a random fingerprint per drug; for an unseen drug, the random vector is statistically uncorrelated with any pharmacological signal.

**Why `both` resists best (r=−0.125 vs −0.35):** KL regularization pushes the latent space toward N(0,I), preventing per-drug memorization and preserving some generality across the omics branch. Pure CE (no KL) learns more specialized per-drug representations — more powerful on seen drugs, more brittle on unseen ones.

**Fix implemented:** `load_ccle_real_data()` now loads real drug SMILES from `Dataset/ccle_drug_smiles.csv` (produced by `fetch_drug_smiles.py` via PubChem API). With real molecular features, the GNN can generalize via structural similarity — Imatinib and Dasatinib share the piperazinyl-benzamide scaffold, so their IC50 profiles should correlate.

#### β-Annealing (new in `BiIntTrainer`)

To combine the best of both worlds (CE reconstruction quality + KL generalization), `BiIntTrainer` now supports **linear β-annealing**:

```
β: 0.0 → 2.0  over vae_anneal_epochs=10 epochs
```

Start with pure reconstruction (β=0, learns pharmacological signal first), then gradually add KL regularization. This prevents early KL collapse while still reaching the healthy KL=64 operating point.

```bash
python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 30 --no-ppo
```

---

### BRICS Tokenization for DQN (`brics_dqn_optimizer.py`)

**Motivation:** SELFIES DQN v3–v5 consistently generated acyclic molecules (arom_rings=0 in all Top-5) despite 93.5% of ChEMBL corpus being aromatic.

**Root cause of SELFIES failure:** In SELFIES 2.x, benzene encodes as `[C][=C][C][=C][C][=C][Ring1][=Branch1]` — the aromatic ring closure uses the `[=Branch1]` token (20,539 occurrences in corpus). The DQN cannot associate this abstract SELFIES token with aromaticity from reward alone. No reward was given specifically for `[=Branch1]`, so the agent never discovered aromatic rings despite the vocabulary containing all necessary tokens.

**BRICS solution:** Replace atom-by-atom generation with **scaffold assembly**. Each action is a complete medicinal chemistry fragment:

| Aspect | SELFIES DQN v3–v5 | BRICS DQN |
|--------|-------------------|-----------|
| Action | Single abstract token | Complete scaffold fragment |
| Benzene | 8 tokens, aromatic closure implicit | 1 token: `[*:1]c1ccccc1` (visibly aromatic) |
| Drug-likeness | Reward-enforced only | Fragments are from drug-like ChEMBL molecules by construction |
| Retrosynthesis | Not guaranteed | BRICS rules follow 16 retrosynthesis bond-breaking patterns |

**Additional rewards in BRICS DQN:**
- `brics_success_bonus=+0.3` when `AllChem.BRICS.BRICSBuild` succeeds
- `fragment_diversity=+0.2×(unique_fragments/total_fragments)`

**Status:** Implemented (`brics_dqn_optimizer.py`, 860 lines) — not yet executed. Run with:
```bash
python3 brics_dqn_optimizer.py > logs_brics_dqn.txt 2>&1
# ~20-30 min | 5,000 episodes | output: Dataset/brics_dqn_results.csv
```

---

### DQN v4.0 Complete Results (10,000 episodes)

```
Valid: 4,110 / 10,000 (41.1%)     Best reward: 2.667 (ep. 200, then frozen)
ε final: 0.050                     Moy50 final: ~0.0
Top-5 molecules: all acyclic (arom_rings=0), all with QED<0.3
```

**Diagnosis — aromatic token not reinforced:**
- F/Cl/Br/I successfully eliminated (post-decode atom check)
- Reward shaping gave +0.03 for `[Ring1]` and `[Ring2]` — but ring closures in SELFIES 2.x require `[Ring1][=Branch1]` pair; `[Ring1]` alone encodes aliphatic ring closure only
- `[=Branch1]` (the true aromatic closure token) had no positive reward — the DQN never specifically reinforced it
- Best SMILES frozen after episode 200: convergence failure despite 10k episodes

**Key finding:** The SELFIES representation itself is the barrier — not the reward function. `[=Branch1]` is a SELFIES-specific abstraction with no chemical meaning; it cannot be learned from IC50/QED reward signals.

---

### DQN v5.0 — Warm-Start Buffer

```
Valid: 6,040 / 10,000 (60.4%)     Best reward: 3.153
ε: 1.0 → 0.15 (ε_min raised from 0.05)
Warm-start: 500 episodes from SEED_SMILES BRICS decomposition
Top-5: still all acyclic (arom_rings=0) — same root cause as v4.0
```

**Diagnosis:** Warm-start improved validity (60.4% vs 41.1%) but did not fix aromatic generation. The warm-start trajectories contain `[=Branch1]` tokens in SEED_SMILES SELFIES encodings, but without a specific reward attached to that token, the DQN does not learn to select it during generation.

---

### DQN v5.1 — Chirurgical Aromatic Token Fix

**Fix:** Direct positive reward for SELFIES aromatic tokens at each generation step:

```python
if token in ("[=Branch1]", "[#Branch1]"):
    step_reward = +0.20   # aromatic closure — strongest step signal
elif token in ("[Ring1]", "[Ring2]"):
    step_reward = +0.10   # ring closure
elif token in ("[=C]", "[=N]"):
    step_reward = +0.05   # double bond (aromatic context)
else:
    step_reward = 0.0
```

**Rationale:** `[=Branch1]` is the aromatic ring closure token in SELFIES 2.x (20,539 occurrences in 10k ChEMBL, vs `[Ring1]` alone for aliphatic rings). By rewarding it directly at generation time — not just at episode end — the DQN receives an immediate learning signal for every aromatic ring it closes.

**Status:** Implemented — awaiting execution.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/evaluation.ipynb` | Training curves, GraphGA candidates visualization, DQN version comparison |
