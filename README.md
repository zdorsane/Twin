# Bi-Int Digital Twin — AI Platform for Drug Discovery & IC50 Prediction

## Overview

Ce projet implémente un **Digital Twin** biologique complet pour le criblage de médicaments sur lignées cellulaires cancéreuses et la prédiction d'IC50 à partir de données multi-omiques réelles (CCLE).

Le pipeline combine :
- Un encodeur moléculaire GNN pré-entraîné sur **100 000 molécules ChEMBL**
- Un VAE quaternion pour la fusion multi-omique (GEx + CNA + Mutations)
- Des blocs Bi-Int avec cross-attention bidirectionnelle (inspirés d'AlphaFold2)
- Une prédiction IC50 entraînée sur **137 182 triplets réels** (drogue × lignée cellulaire)
- Plusieurs méthodes d'optimisation RL pour la génération de molécules candidates

---

## Résultats obtenus

### Pré-entraînement ChEMBL (GNN)

| Epoch | Train RMSE | Val RMSE | Val MAE |
|-------|-----------|---------|--------|
| 1     | 0.4875    | 0.3519  | 0.2451 |
| 5     | 0.2687    | 0.2503  | 0.1747 |
| 9     | 0.2140    | 0.2088  | 0.1476 |
| 10    | 0.2100    | 0.2155  | 0.1525 |

**100 000 molécules ChEMBL** traitées sur GPU RTX 4000 Ada (17710 MB VRAM).
L'encodeur GNN apprend des représentations chimiques transférables (LogP, TPSA, MW, QED, HBD, HBA, NumRings).

---

### Entraînement QSAR sur données CCLE réelles

| Epoch | Train RMSE | Val RMSE |
|-------|-----------|---------|
| 1     | 0.7754    | 0.5749  |
| 5     | 0.5125    | 0.4943  |
| 10    | 0.4818    | 0.4847  |
| 15    | 0.4712    | 0.4720  |
| 20    | 0.4635    | 0.4723  |

**Interprétation :**
- Train RMSE ≈ Val RMSE → pas d'overfitting, bonne généralisation
- Convergence stable sur 20 epochs
- KL stable à 64.0 → espace latent VAE bien régularisé
- 137 182 triplets réels (266 drogues × 647 lignées cellulaires communes)

---

## Architecture

```
[Drug SMILES]
      ↓
GNN pré-entraîné ChEMBL (100k molécules)
      ↓
Drug Embedding

[GEx (978 gènes)] + [CNA (426 gènes)]
      ↓
OmicsVAE (encodeur quaternion)
      ↓
Cell Embedding (latent z, dim=128)

Drug Embedding + Cell Embedding
      ↓
Bi-Int Blocks (cross-attention bidirectionnelle + mises à jour triangulaires)
      ↓
IC50 prédit (log µM)
```

Ce que le modèle apprend :

```
(Molécule médicament) + (Expression génique) + (CNA)
             ↓
     Sensibilité tumorale
             ↓
          IC50
```

---

## Données utilisées

### CCLE (Cancer Cell Line Encyclopedia) — données réelles

| Fichier | Description | Dimensions |
|---------|-------------|-----------|
| `data_mrna_seq_rpkm.txt` | Expression génique (RPKM) | 56 319 gènes × 1 068 lignées |
| `data_cna.txt` | Variations du nombre de copies | 23 312 gènes × 1 068 lignées |
| `data_drug_treatment_ic50.txt` | IC50 mesurés | 266 drogues × 1 068 lignées |
| `data_mutations.txt` | Mutations somatiques (MAF) | — |

Après alignement des lignées communes : **647 lignées × 266 drogues = 137 182 triplets valides**.

### ChEMBL 36

- `Dataset/chembl_36.sdf` : 2 854 815 molécules (7.4 GB)
- Pré-entraînement sur les 100 000 premières molécules valides (≤60 atomes)
- Descripteurs cibles : LogP, TPSA, MW, HBD, HBA, QED, NumRings, NumAromaticRings

---

## GPU — Environnement

| Composant | Valeur |
|-----------|--------|
| GPU | NVIDIA RTX 4000 Ada Generation |
| VRAM | 17 710 MB |
| CUDA | 12.3 |
| TensorFlow | 2.21.0 |
| OS | Ubuntu 24.04 (WSL2) |

---

## Structure du projet

```
Twin/
├── fullPipeline.py                  # Modèle principal Bi-Int + entraînement QSAR + PPO
├── chembl_pretrain.py               # Pré-entraînement GNN sur ChEMBL (100k molécules)
├── inference.py                     # Pipeline d'inférence
├── dqn_optimizer.py                 # Optimiseur DQN (Deep Q-Network)
├── reinvent_biint_optimizer.py      # Optimiseur REINVENT-style (policy gradient)
├── graphga_biint_optimizer.py       # Optimiseur algorithme génétique sur graphes
├── api_server.py                    # Serveur REST pour l'inférence
├── smiles_tokenizer.json            # Vocabulaire SMILES persisté
├── pretrained_drug_encoder.keras    # Poids GNN pré-entraîné ChEMBL
├── pretrained_weights/
│   ├── chembl_drug_encoder.weights.h5
│   └── pretrain_meta.json
├── COMMANDES.md                     # Toutes les commandes Ubuntu pour reproduire
├── Dataset/
│   ├── chembl_36.sdf                # 2.8M molécules ChEMBL
│   └── ccle_broad_2019/             # Données CCLE réelles
└── README.md
```

---

## Commandes pour reproduire

### Étape 1 — Pré-entraîner sur ChEMBL

```bash
cd ~/Twin && source venv_tf/bin/activate
nohup python3 chembl_pretrain.py > ~/Twin/logs_chembl.txt 2>&1 & echo "PID: $!"
tail -f ~/Twin/logs_chembl.txt
```

### Étape 2 — Entraîner le modèle QSAR sur CCLE

```bash
cd ~/Twin && source venv_tf/bin/activate
python3 fullPipeline.py --no-ppo
```

### Étape 3 — Pipeline complet avec PPO (génération moléculaire)

```bash
python3 fullPipeline.py
```

### Autres optimiseurs

```bash
python3 dqn_optimizer.py              # DQN drug generation
python3 graphga_biint_optimizer.py    # GraphGA evolutionary optimizer
python3 reinvent_biint_optimizer.py   # REINVENT policy gradient
```

---

## Méthodes d'optimisation RL

### PPO — Proximal Policy Optimization
- Acteur-critique avec objectifs clippés
- Pré-entraînement supervisé (behavior cloning) sur SMILES valides
- Curriculum learning : décroissance progressive de la température

### GraphGA — Algorithme Génétique
- Évolution de population sur SMILES valides (mutation, croisement, sélection)
- Bi-Int comme oracle de fitness : IC50 + QED + SA + pénalités Lipinski
- Résultats : QED 0.71–0.93, MW 269–347 Da, LogP 1.0–3.3

### DQN — Deep Q-Network
- Double DQN avec replay buffer (20 000 transitions)
- ε-greedy décroissant (1.0 → 0.05)
- Récompense multi-critère : IC50 + validité chimique + diversité Tanimoto

### REINVENT-style RL
- Policy gradient conditionné sur l'embedding omique z
- Récompense : IC50 prédit × validité RDKit

---

## Niveau scientifique

Ce pipeline est comparable aux approches utilisées dans :
- **Drug response prediction** (GDSC, PRISM, CTRPv2)
- **QSAR multimodal** (multi-omics + structure chimique)
- **Pharmacogénomique IA** (precision oncology)
- **De novo drug design** par apprentissage par renforcement

---

## Étapes réalisées

1. Architecture Bi-Int complète (quaternion VAE, cross-attention bidirectionnelle, mises à jour triangulaires)
2. Tokenizer SMILES persisté
3. Pré-entraînement GNN sur 100 000 molécules ChEMBL (GPU RTX 4000)
4. Chargement des vraies données CCLE (137 182 triplets IC50 réels)
5. Entraînement QSAR multimodal — Val RMSE final : **0.4723**
6. Transfer learning ChEMBL → QSAR validé
7. PPO, GraphGA, REINVENT, DQN implémentés
8. Validation chimique (QED, SA, LogP, MW, Lipinski)
9. Support GPU complet (WSL2 + CUDA 12.3 + TF 2.21)

---

## Prochaines étapes

1. Corriger le chargement des mutations (format MAF non standard)
2. Augmenter le nombre de molécules ChEMBL (500k → 1M avec streaming)
3. Intégration docking moléculaire (AutoDock Vina) dans la récompense RL
4. Remplacement LSTM → Transformer pour la génération PPO
5. Multi-tâche : prédiction simultanée IC50 + toxicité + solubilité
6. Déploiement REST API (FastAPI + Docker)

---

## Références

- Mnih et al., *Human-level control through deep RL*, Nature 2015
- Van Hasselt et al., *Double DQN*, AAAI 2016
- Olivecrona et al., *Molecular de novo design through deep RL*, J. Cheminformatics 2017
- Jumper et al., *AlphaFold2 triangular updates*, Nature 2021
- Partin et al., *Bi-Int cross-attention for drug-omics*, 2023
