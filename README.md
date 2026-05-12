# Bi-Int Digital Twin for Drug Discovery and IC50 Prediction

## Overview
This project implements a comprehensive **Bipartite Interaction Transformer (Bi-Int)** digital twin system for cell line drug screening and IC50 prediction using multi-omics data. The system integrates molecular drug features with omics embeddings through advanced neural architectures and includes reinforcement learning for de novo drug generation.

**Key Features:**
- **Molecular Encoding**: BRICS fragmentation + Pre-trained GNN (ChEMBL) for drug representation
- **Omics Integration**: Quaternion-fused Variational Autoencoder (VAE) for multi-modal data (Gene Expression, Mutations, Copy Number Variations)
- **Interaction Modeling**: Bi-Int blocks with bidirectional cross-attention and triangular updates
- **Prediction**: Regression head for log IC50 values
- **Drug Generation**: PPO-based reinforcement learning for conditional SMILES optimization
- **Alternative Optimization**: GraphGA genetic algorithm for robust molecular evolution
- **Validation Pipeline**: RDKit-based chemical property analysis (QED, SA, MW, LogP)
- **Pre-training**: ChEMBL-based initialization of drug encoder for improved molecular representations

The pipeline currently uses synthetic CCLE-like data for demonstration but is designed for integration with real datasets (CCLE, GDSC, TCGA).

## Architecture Summary
- **Drug Input**: SMILES strings → BRICS fragments → Pre-trained GNN (ChEMBL) → Drug embeddings (D)
- **Omics Input**: GEx/Mut/CNV → Quaternion VAE → Omics embeddings (O)
- **Fusion**: Bi-Int blocks (Row-Cross Attention, Col-Cross Attention, Triangular Updates) → IC50 prediction
- **RL Generation**: PPO-conditioned LSTM/Transformer for SMILES sampling
- **Fitness Oracle**: Bi-Int model provides IC50 predictions for optimization
- **Inputs**: Drug SMILES → atom features; Omics (GEx, Mut, CNV).
- **Fusion**: Quaternion algebra for omics; Bi-Int for drug-omics interactions.
- **RL Generation**: PPO-conditioned LSTM/Transformer for SMILES sampling
- **Fitness Oracle**: Bi-Int model provides IC50 predictions for optimization

## Project Structure

```
api_server.py
chembl_pretrain.py
env.yml
FIXES_APPLIED.md
fullPipeline.py
graphga_biint_optimizer.py
graphga_population.txt
graphga_ranked_population.csv
graphga_top_candidates.csv
graphga_validated_candidates.csv
inference.py
load_smiles_data.py
pretrained_drug_encoder.keras
README.md
reinvent_biint_optimizer.py
reinvent_optimizer.py
sanitize_population.py
simple_reinvent.py
smiles_data.txt
smiles_sanitizer.py
smiles_tokenizer.json
test_rdkit.py
transformer_smiles_gen.py
validate_graphga_candidates.py
__pycache__/
Dataset/
	chembl_36.sdf
	chembl_36.sdfZone.Identifier
	ccle_broad_2019/
		data_clinical_patient.txt
		data_clinical_patient.txtZone.Identifier
		data_clinical_sample.txt
		data_clinical_sample.txtZone.Identifier
		data_cna_hg19.seg
		data_cna_hg19.segZone.Identifier
		data_cna.txt
		data_cna.txtZone.Identifier
		data_drug_treatment_auc.txt
		data_drug_treatment_auc.txtZone.Identifier
		data_drug_treatment_ic50.txt
		data_drug_treatment_ic50.txtZone.Identifier
		data_drug_treatment_zscore.txt
		data_drug_treatment_zscore.txtZone.Identifier
		data_gene_panel_matrix.txt
		data_gene_panel_matrix.txtZone.Identifier
		data_mrna_seq_rpkm_zscores_ref_all_samples.txt
		data_mrna_seq_rpkm_zscores_ref_all_samples.txtZone.Identifier
		data_mrna_seq_rpkm_zscores_ref_diploid_samples.txt
		data_mrna_seq_rpkm_zscores_ref_diploid_samples.txtZone.Identifier
		data_mrna_seq_rpkm.txt
		data_mrna_seq_rpkm.txtZone.Identifier
		data_mutations.txt
		data_mutations.txtZone.Identifier
		data_protein_quantification_zscores.txt
		data_protein_quantification_zscores.txtZone.Identifier
		data_protein_quantification.txt
		data_protein_quantification.txtZone.Identifier
		data_sv.txt
		data_sv.txtZone.Identifier
		LICENSE
		LICENSEZone.Identifier
		meta_clinical_patient.txt
		meta_clinical_patient.txtZone.Identifier
		meta_clinical_sample.txt
		meta_clinical_sample.txtZone.Identifier
		meta_cna_hg19_seg.txt
		meta_cna_hg19_seg.txtZone.Identifier
		meta_cna.txt
		meta_cna.txtZone.Identifier
		...
```

## Usage
Run the full pipeline from the `Twin` directory using one of the following modes:

- `pretrained` : use the ChEMBL drug encoder initialization
- `baseline`   : train the Bi-Int model with a randomly initialized drug encoder
- `compare`    : run both baseline and pretrained modes back-to-back and compare final IC50 validation RMSE

Example commands:

```bash
# Run the normal pipeline with ChEMBL pre-training
bash run_pipeline.sh

# Run fullPipeline.py in baseline mode
source venv_tf/bin/activate && python3 fullPipeline.py --mode baseline --epochs 20

# Run the comparison mode
source venv_tf/bin/activate && python3 fullPipeline.py --mode compare --epochs 20
```

## Completed Components ✅

### 1. Bi-Int Digital Twin Model
- **Architecture**: Full implementation with quaternion layers, cross-attention blocks, and MLP prediction head
- **Training**: Converged to RMSE ~1.8 log µM on synthetic data (20 epochs)
- **Parameters**: 9.4M trainable parameters
- **Performance**: Stable convergence with KL regularization

### 2. IC50 Prediction Pipeline
- **Drug Screening**: Batch prediction over compound libraries
- **Virtual Gene Knockouts**: In-silico perturbation analysis with delta IC50 computation
- **Sensitivity Profiling**: Cell line-specific IC50 predictions
- **Inference API**: `DigitalTwinInference` class for production deployment

### 3. Data Infrastructure
- **SMILES Tokenizer**: Parallel encoding/decoding with vocabulary persistence (`smiles_tokenizer.json`)
- **Dataset Generation**: Synthetic CCLE/GDSC-compatible batches
- **Pre-training Data**: 49,996 ChEMBL molecules for drug encoder warm-up

### 4. ChEMBL Pre-training
- **Dataset**: 49,996 molecules from ChEMBL 36 SDF (`Dataset/chembl_36.sdf`)
- **Featurizer**: BRICS fragmentation + simple GCN with residual connections
- **Training**: 5 epochs, MSE loss (targets set to 0.0 as no IC50 values extracted)
- **Weights**: Saved in `pretrained_weights/chembl_drug_encoder.weights.h5`
- **Integration**: Automatically loaded in `fullPipeline.py` for drug encoder initialization

### 5. Reinforcement Learning Setup
- **PPO Framework**: Actor-critic with clipped objectives, entropy regularization, GAE
- **Pre-training**: 100 epochs supervised learning (loss: 3.51 → 0.90)
- **Curriculum Learning**: Progressive entropy decay (0.15 → 0.03) and temperature annealing
- **Hyperparameters**: Tuned for stability (clip ε=0.2, vf_coef=0.5)

### 6. GraphGA Optimization (Alternative to RL)
- **Genetic Algorithm**: Population-based evolution on SMILES graphs
- **Operations**: Mutation, crossover, selection with elitism
- **Fitness Function**: Multi-objective (IC50 + QED + SA + Lipinski penalties)
- **Validation**: 50 generations completed with 90% validity rate
- **Results**: Drug-like molecules (QED 0.71-0.93, MW 269-347 Da, LogP 1.0-3.3)

### 7. Chemical Validation Pipeline
- **RDKit Integration**: Validity checking, canonicalization, property calculation
- **Metrics**: QED, Synthetic Accessibility (SA), Molecular Weight, LogP, Ring count
- **Filtering**: PAINS alerts, Lipinski violations, structural penalties
- **Output**: CSV exports with ranked candidates and top selections
- **RDKit Integration**: Validity checking, canonicalization, property calculation
- **Metrics**: QED, Synthetic Accessibility (SA), Molecular Weight, LogP, Ring count
- **Filtering**: PAINS alerts, Lipinski violations, structural penalties
- **Output**: CSV exports with ranked candidates and top selections

## Current Status and Issues ⚠️

### PPO Drug Generation (Ongoing Issue)
- **Problem**: Mode collapse to invalid/short SMILES despite fixes
- **Symptoms**: Entropy drops rapidly (1.0 → 0.6), rewards remain negative (-0.9 → -0.5)
- **Root Cause**: LSTM capacity limits for SMILES grammar; insufficient exploration vs exploitation balance
- **Attempts**: Entropy curriculum, reward shaping (penalize alkanes, bonus cycles), pre-training expansion
- **Current Output**: Invalid sequences like `11(1`, `H(21`, `=))==()C==C)((CCC==(1(OC==1()C`

### GraphGA Success
- **Status**: Fully functional alternative with validated results
- **Advantage**: 100% validity guarantee, no training required
- **Limitation**: Slower than RL for large explorations
- **Current Output**: Mostly invalid SMILES (e.g., `?`, unmatched parentheses)
- **Limitation**: Slower than RL for large explorations

## Remaining Steps 📋

### Immediate Priorities (Next 1-2 Weeks)
1. **Fix PPO Generation**
   - Implement Transformer-based SMILES generator (replace LSTM)
   - Add stronger token masking (allow only chemically valid continuations)
   - Integrate SELFIES representation for robust molecular sampling
   - Test convergence to positive rewards and valid drug-like SMILES

2. **Molecular Docking Integration**
   - Install AutoDock Vina or Glide
   - Implement docking pipeline for top candidates
   - Compute binding poses, affinities, and interaction fingerprints
   - Validate against known targets (e.g., EGFR, BRAF for cancer cell lines)

3. **Enhanced Validation**
   - Install `rdkit-sascorer` for true Synthetic Accessibility scores
   - Add ADMET property predictions (toxicity, solubility, permeability)
   - Implement diversity metrics (Tanimoto similarity, scaffold analysis)

### Medium-term Goals (1-3 Months)
4. **Real Data Integration**
   - Replace synthetic data with actual CCLE/GDSC datasets
   - Handle missing values, normalization, and batch effects
   - Validate on held-out test sets with known IC50 values

5. **Model Improvements**
   - Hyperparameter optimization (architecture search)
   - Multi-task learning (IC50 + toxicity prediction)
   - Uncertainty quantification with ensemble methods

6. **Production Deployment**
   - Containerize with Docker
   - Build REST API for inference
   - Implement caching and batch processing
   - Add monitoring and logging

### Long-term Vision (3-6 Months)
7. **Advanced Architectures**
   - Integrate pre-trained molecular models (ChemBERTa, MolFormer)
   - Implement diffusion models for molecular generation
   - Add 3D structure prediction and conformation analysis

8. **Clinical Translation**
   - Partner with pharma for real drug screening campaigns
   - Validate predictions against wet-lab assays
   - Publish results in peer-reviewed journals

## Installation and Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.15.0 (CPU mode enforced)
- RDKit 2023+
- CUDA-compatible GPU (optional, but recommended for training)

### Dependencies
```bash
pip install tensorflow==2.15.0 tensorflow-probability rdkit-pypi numpy pandas scikit-learn
# Optional for enhanced SA scoring
pip install rdkit-sascorer
# For docking
pip install meeko vina  # or conda install -c conda-forge autodock-vina
```

### Environment Setup
```bash
# Create conda environment
conda env create -f env.yml
conda activate TwinCell

# Force CPU usage (if no GPU)
export CUDA_VISIBLE_DEVICES=-1
```

### Running the Pipeline
```bash
# Full pipeline (training + RL + inference)
python3 fullPipeline.py

# GraphGA optimization only
python3 graphga_biint_optimizer.py

# Validation of candidates
python3 validate_graphga_candidates.py

# Sanitization
python3 sanitize_population.py graphga_ranked_population.csv
```

## Results and Validation

### Model Performance
- **Training RMSE**: 1.8168 → 1.6280 log µM (improving)
- **Inference Speed**: ~50ms per prediction (batch=16)
- **Virtual KO**: Correctly computes sensitivity shifts

### GraphGA Optimization Results
- **Generations**: 50 completed
- **Validity Rate**: ~90% after filtering
- **Top Molecules**:
  - QED: 0.710-0.926 (excellent drug-likeness)
  - MW: 269-347 Da (optimal range)
  - LogP: 1.0-3.3 (balanced lipophilicity)
  - Composite Score: 2.7-4.0

## Results and Validation

### Model Performance
- **Training RMSE**: 1.8168 → 1.6280 log µM (improving)
- **Inference Speed**: ~50ms per prediction (batch=16)
- **Virtual KO**: Correctly computes sensitivity shifts

### GraphGA Optimization Results
- **Generations**: 50 completed
- **Validity Rate**: ~90% after filtering
- **Top Molecules**:
  - QED: 0.710-0.926 (excellent drug-likeness)
  - MW: 269-347 Da (optimal range)
  - LogP: 1.0-3.3 (balanced lipophilicity)
  - Composite Score: 2.7-4.0

### PPO Status - Latest Improvements ✅
**Episode 1 Run (Before Fixes):**
- Entropy collapsed: 1.0 → 0.07 by episode 70
- Rewards remained deeply negative throughout (-0.5 to -1.0)
- Best SMILES: Invalid fragments like `11(1`, `=))==()C==C)`

**Episode 2 Run (After Bootstrap + Entropy Floor):**
- Entropy maintained high: 0.4 → 2.7 (no collapse!)
- Pre-training improved: Loss 3.5 → 0.38 (2x better convergence)
- Best SMILES: Partially valid (`CCCCCCCCCNCN1CCC1CCCCC`, `CCOCCCCCCN`)
- **Still negative rewards** because strict penalties still applied

**Episode 3 Run (After Bootstrap + Curriculum Learning):**
- **New feature**: Curriculum reward scaling (1.0 → 3.0 over 200 episodes)
- **Early episodes** (1-60): Generous rewards for ANY valid SMILES (+0.2 bootstrap)
- **Late episodes** (60-200): Strict drug-likeness penalties
- **Expected result**: Monotonic reward improvement, valid SMILES by episode 200

**How It Works:**
1. Episodes 1-60: Policy learns to generate ANY valid SMILES structure (entropy high)
2. Episodes 61-200: Policy fine-tunes for drug-like properties (rings, QED, LogP)
3. Hard penalties (ring==0) only kick in after 50% training

This mimics human learning: first learn alphabet, then write sentences, then write poetry.

## Future Improvements

### Recommended Fixes for PPO
1. **Architecture Upgrade**:
   ```python
   class TransformerSMILESGenerator(Model):
       # Multi-head attention + positional encoding
       # Cross-attention to omics latent z
   ```

2. **Representation Enhancement**:
   - Switch to SELFIES (Self-referencing Embedded Strings) for robust generation
   - Add chemical grammar constraints

3. **Reward Engineering**:
   - Include docking scores in reward function
   - Add diversity bonuses to prevent mode collapse

### Alternative Approaches
- **Pre-trained Models**: Fine-tune ChemGPT or MolGen on domain data
- **Specialized Libraries**: Integrate REINVENT or GuacaMol for production-ready optimization

---

## WORK
Voici le résumé détaillé de toutes les étapes réalisées dans le projet :

1. **Architecture Bi-Int**
   - Conception et implémentation d'un modèle Bi-Int pour la prédiction d'IC50.
   - Intégration de BRICS molecular featurization + GNN pour le décodage des molécules.
   - Création d'un VAE quaternion pour fusionner les données multi-omiques (GEx, Mut, CNV).
   - Ajout de blocs de cross-attention bidirectionnelle et de mises à jour triangulaires.

2. **Pipeline de prédiction**
   - Mise en place d'un pipeline d'inférence pour le screening de molécules.
   - Implémentation de la simulation de knock-out virtuel pour mesurer l'impact sur l'IC50.
   - Gestion de la génération de données synthétiques CCLE-like pour tests.

3. **Préparation des données SMILES**
   - Création d'un vocabulaire SMILES et d'un tokenizer parallèle.
   - Chargement de 47 SMILES valides pour l'initialisation et le warmup.
   - Alignement du batch size entre les trajectoires SMILES et les latents omiques.

4. **Renforcement & génération**
   - Mise en place d'un framework PPO avec pré-entraînement supervisé.
   - Ajout d'un training curriculum : température progressive et masking de logits.
   - Détection des limites du LSTM pour la génération de SMILES valides.

5. **Transition GraphGA**
   - Ajout de `graphga_biint_optimizer.py` pour remplacer l'approche REINVENT.
   - Mise en place d'une optimisation génétique sur SMILES valides.
   - Utilisation de Bi-Int comme oracle de fitness pour IC50.
   - Ajout de mutation/crossover, sélection et génération de population.

6. **Validation chimique**
   - Création de `validate_graphga_candidates.py` pour valider les candidats RDKit.
   - Calcul de QED, SA (fallback) et logP.
   - Sauvegarde des résultats complets et des top candidats filtrés.

9. **Résultats GraphGA finaux**
   - 50 générations exécutées avec succès
   - Meilleurs scores fitness : 2.7-4.0 (amélioration progressive)
   - Molécules générées : propriétés drug-like excellentes
     - QED : 0.710-0.926 (très bonnes valeurs >0.7)
     - MW : 269-347 Da (poids moléculaire optimal)
     - LogP : 1.0-3.3 (lipophilie équilibrée)
   - Taux succès : ~90% de candidats valides après filtrage
   - SA : calcul fallback (0.000) - améliorable avec rdkit-sascorer

10. **Validation et analyse**
    - Script `validate_graphga_candidates.py` mis à jour pour lire résultats générés
    - Fonction `analyser_candidats()` pour analyse propre QED/MW/LogP
    - Export CSV : `graphga_validated_candidates.csv`, `graphga_top_candidates.csv`
    - Tri par score composite : QED + SA - logP_penalty

11. **Nettoyage et sanitisation**
    - Script `sanitize_population.py` pour post-traitement
    - Suppression SMILES invalides (kekulization, valence, aromaticité)
    - Double-vérification via SMILES canonique
    - Désactivation logs RDKit pour console propre

12. **Améliorations futures**
    - Installation rdkit-sascorer pour vraie SA
    - Augmentation générations (100+) pour convergence
    - Ajustement paramètres évolution (mutation, crossover)
    - Intégration vraie base de données médicaments

8. **Dépendances et environnement**
   - Gestion des versions critiques : NumPy, TensorFlow, TensorFlow Probability.
   - Contournement d'erreurs GPU JIT en forçant l'exécution CPU pour le script GraphGA.

---

## Next Steps (For Future Improvement)

### **Option 2: Implement Transformer Architecture** (Recommended)
Replace LSTM with Transformer for better sequence generation:
```python
class TransformerSMILESGenerator(Model):
    # Multi-head self-attention + cross-attention to z
    # Proven superior for sequence generation tasks
```

### **Option 3: GraphGA + Bi-Int oracle** (recommandé)
- Genetic Algorithm sur graphes moléculaires — [STONED, GEAM, GraphGA]
- Pas de prior à entraîner, opère directement sur les graphes
- Mutations / crossover sur SMILES valides → 100% validité garantie
- Rapide à intégrer : ton Bi-Int devient simplement la fonction fitness
- Récent : GraphGA (2022), GEAM (2023)
- Flux recommandé :
  - Population initiale SMILES (drug-like)
  - Mutation / Crossover (SELFIES)
  - Oracle → Bi-Int IC50 + QED + SA
  - Sélection des meilleurs candidats avec un tri composite (QED + SA - logP_penalty)
  - Renforcer le critère QED >= 0.7 pour les candidats retenus
  - Répéter 50-100 générations

### **Option 4: Specialized Library**
- Use GUACAMOL or other molecular optimization toolkits
- GraphGA is simple and robust for current setup
- Production-ready alternative if custom GA is insufficient

---

## Validation
- Use `validate_graphga_candidates.py` to verify RDKit validity and compute QED/SA for top GraphGA molecules.
- Automatically reads from `graphga_ranked_population.csv` (top 10 candidates)
- Run:
  ```bash
  python3 validate_graphga_candidates.py
  ```
- The `analyser_candidats()` function provides clean QED/MW/LogP analysis for valid SMILES lists.
- Results: Excellent drug-like properties (QED 0.7-0.9, MW 270-350 Da, LogP 1-3)

## Sanitization
- Use `sanitize_population.py` for post-processing GraphGA results
- Removes invalid SMILES with chemical errors
- Run:
  ```bash
  python3 sanitize_population.py graphga_ranked_population.csv
  ```
- Outputs: `*_sanitized.csv` and `*_valid_only.csv`

---

## Deployment Notes

### **Option 2: Implement Transformer Architecture** (Recommended)
Replace LSTM with Transformer for better sequence generation:
```python
class TransformerSMILESGenerator(Model):
    # Multi-head self-attention + cross-attention to z
    # Proven superior for sequence generation tasks
```

### **Option 3: Integrate Pre-trained Model**
- Use fine-tuned ChemBERTa or GPT2 on SMILES datasets
- Requires ~100MB model download
- Significantly better generalization

### **Option 4: Specialized Library**
- Integrate REINVENT (Geitner et al., 2020)
- Or use GUACAMOL for molecular optimization
- Production-ready but requires external dependencies

---

## Deployment Notes
1. **Save Trained Model**: Use `model.save_weights()` before production
2. **Real Data Integration**: Replace synthetic generator with actual CCLE/GDSC data
3. **GPU Optimization**: Model currently targets RTX 4000 Ada (17GB VRAM)
4. **Inference API**: Wrap DigitalTwinInference class as REST endpoint

## Troubleshooting
- **Out of Memory**: Reduce `batch_size` from 32 to 16
- **Slow Training**: Enable mixed precision: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- **SMILES Invalidity**: Increase pre-training epochs or implement Transformer

---

## Requirements
- Python 3.8+
- TensorFlow 2.10+ (with TensorFlow Probability)
- RDKit (optional, falls back to mock featurization)
- NumPy, Keras

## Setup and Run
1. Install dependencies: `pip install tensorflow tensorflow-probability rdkit-pypi numpy`
2. Run the pipeline: `python fullPipeline.py`
   - Builds model, trains on synthetic data, runs PPO optimization, and performs inference demo.

## Data
- Uses synthetic data mimicking CCLE/GDSC formats.
- For real data: Replace `generate_synthetic_ccle_batch` with actual loaders for CCLE omics and GDSC IC50.

## Notes for Experts
- Quaternion layers enable algebraic fusion of heterogeneous omics modalities.
- Bi-Int blocks draw inspiration from AlphaFold2 triangular updates for structured interactions.
- RL reward is negative predicted IC50 (lower IC50 = higher reward).
- Model scales to ~9M parameters; optimize for GPU training.

For questions or contributions, refer to the code comments in `fullPipeline.py`.# Twin
