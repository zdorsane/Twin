# Rapport de session — Twin Project
**Date :** 31 mai 2026  
**Auteur :** Zied Dorsane  
**Superviseur :** M. Marouane  
**Modèle :** Bi-Interaction (GNN + Quaternion-VAE) — prédiction IC50 CCLE + génération de novo

---

## Vue d'ensemble

Cette session a porté sur **deux axes prioritaires** demandés par M. Marouane :

1. **Renforcement de la validation chimique/biologique** des molécules générées, avec des métriques rigoureuses (SA score, PAINS/Brenk, diversité interne, Tanimoto vs drogues connues).
2. **Mise en garde explicite sur les IC50 prédits**, dont la fiabilité en généralisation hors distribution (Leave-Drug-Out, LDO) reste limitée (r = 0.316).

---

## 1. Contexte et performances du modèle prédicteur

### Architecture
Le modèle Twin combine :
- Un **encodeur GNN** sur les SMILES (graph isomorphism network, Morgan-like representations)
- Un **VAE quaternionique** encodant les profils omiques des lignées cellulaires (expression génique, CNA, mutations)
- Une **couche Bi-Interaction** fusionnant les deux modalités pour prédire le log-IC50

### Résultats de prédiction IC50

| Split | Modèle | Pearson r | IC 95% |
|-------|--------|-----------|--------|
| Random | Bi-Int (epoch 4) | **0.811** | [0.736 – 0.886] |
| Leave-Drug-Out | Bi-Int (epoch 2) | **0.316** | [0.287 – 0.344] |
| Leave-Drug-Out | XGBoost (baseline) | **0.367** | [0.338 – 0.393] |
| Leave-Drug-Out | Ridge (ECFP4+omics) | 0.228 | [0.196 – 0.256] |
| Leave-Drug-Out | RF (50 trees) | 0.231 | [0.202 – 0.259] |

**Interprétation pour l'expert :**

- Le **split random (r = 0.811)** est optimiste car la même drogue peut apparaître dans train et test. Il mesure la capacité d'interpolation du modèle, pas sa généralisation.
- Le **split Leave-Drug-Out (r = 0.316)** est le vrai indicateur de généralisation : le modèle doit prédire la réponse à des drogues structurellement nouvelles, jamais vues à l'entraînement. Un r = 0.316 indique une corrélation faible mais statistiquement significative (p << 0.001, n > 10 000 triplets).
- **XGBoost bat le deep learning sur LDO (0.367 vs 0.316)**, ce qui suggère que la complexité du Bi-Int n'est pas encore justifiée par les données disponibles en mode LDO. La régularisation et l'augmentation de données sont des priorités.

> ⚠️ **Les IC50 prédits pour les molécules générées ne doivent PAS être interprétés comme des prédictions fiables de potency.** Le modèle prédit dans un espace hors distribution (nouvelles drogues structurellement non apparentées aux 201 drogues CCLE). Validation in vitro requise.

---

## 2. Génération de molécules — deux approches

### 2a. GraphGA (Genetic Algorithm sur graphes moléculaires)
- Optimisation par algorithme génétique guidé par QED + IC50 prédit
- **10 candidats** générés, tous avec QED > 0.70
- Scaffolds diversifiés autour de motifs aniline-pipérazine et carbamate

### 2b. BRICS-DQN (Reinforcement Learning par fragments BRICS)
- Agent DQN assemblant des fragments BRICS (Retrosynthetically Interesting Chemical Substructures)
- 3 008 molécules valides générées avec reward > 0 sur ~5 000 épisodes
- Validité globale : ~60% (amélioration de la valence aromatique en cours)
- **Top 50 sélectionnés** par score de récompense pour la validation

---

## 3. Validation moléculaire rigoureuse — résultats du 31 mai 2026

Script : `scripts/molecular_validation.py`  
Dataset analysé : 60 candidats uniques (10 GraphGA + 50 BRICS-DQN top-reward)

### 3.1 Similarité Tanimoto vs drogues CCLE (Morgan FP, r=2, 2048 bits)

| Zone | Interprétation | N candidats |
|------|---------------|-------------|
| Tanimoto > 0.7 | Analogue proche d'une drogue connue — peu novateur | **0** |
| Tanimoto 0.4–0.6 | Zone idéale : proche du connu, potentiellement brevetable | **0** |
| Tanimoto < 0.3 | Structurellement nouveau — innovant mais risqué | **58 / 60** |
| Tanimoto 0.3–0.4 | Frontière (BRI-58 : 0.318, BRI-29 : 0.325) | **2 / 60** |

**Drogues de référence les plus proches :**
Les candidats se comparent principalement à Belinostat, TubastatinA, Pyrimethamine, et GW441756 — toutes des molécules oncologiques avec des mécanismes connus (HDAC inhibiteurs, antiparasitaires repositionnés). La faible similarité (max 0.32) confirme que la bibliothèque est **structurellement originale** par rapport aux drogues CCLE.

**Ce que cela signifie cliniquement :**  
La nouveauté structurelle est une arme à double tranchant. Elle augmente le potentiel de brevet et la chance de cibler des espaces biologiques non exploités, mais elle augmente aussi l'incertitude pharmacocinétique et le risque d'échec ADMET. Une similarité < 0.3 place ces molécules dans un territoire où aucune donnée clinique analogue n'est disponible.

### 3.2 Synthetic Accessibility (SA score)

| Seuil | Interprétation | N candidats |
|-------|---------------|-------------|
| SA 1–3 | Facilement synthétisable | 40 / 60 (67%) |
| SA 3–5 | Synthèse modérément complexe | 18 / 60 (30%) |
| SA 5–6 | Difficile mais faisable | 2 / 60 (3%) |
| SA > 6 | Très difficile — **flaggé** | **0 / 60** |

**Résultat notable : aucun candidat n'a de SA > 6.** La bibliothèque est entièrement synthétisable selon les critères sascorer (RDKit Contrib). Le top candidat BRI-12 a SA = 1.68, ce qui signifie qu'il peut être produit en 1–2 étapes de synthèse.

**Candidats avec le meilleur SA** : BRI-12 (1.68), BRI-46 (1.89), BRI-38 (2.05), BRI-14 (2.07), BRI-26 (3.12), Gra-2 (2.15).

### 3.3 Filtres MedChem — bilan global

| Filtre | Candidats échouant | % |
|--------|-------------------|---|
| PAINS (groupes réactifs, promiscuous binders) | 1 | 1.7% |
| Brenk (groupes toxiques/réactifs) | 18 | 30% |
| Lipinski (drug-like MW/logP/HBD/HBA) | 5 | 8.3% |
| Veber (rotatable bonds ≤ 10, TPSA ≤ 140) | 6 | 10% |
| **Tous filtres passés (medchem clean)** | **38 / 60** | **63%** |

**Détail des alertes Brenk les plus fréquentes :**
- `aniline` (2 cas) : groupes amino aromatiques — risque de toxicité aniline, métabolisme CYP450
- `isolated_alkene` (3 cas) : double liaison non conjuguée — risque d'alkylation
- `Aliphatic_long_chain` (3 cas) : chaînes lipophiles longues — mauvaise solubilité
- `Michael_acceptor_1` (2 cas) : potentiel alkylant sur les nucléophiles biologiques
- `thiol_2`, `cumarine`, `iodine`, `triple_bond` : groupes réactifs spécifiques

**Interprétation :** Le fait que 63% des candidats passent tous les filtres est un résultat **positif pour un modèle génératif RL**. Les filtres PAINS/Brenk sont des heuristiques, pas des verdicts définitifs — certains médicaments approuvés contiennent des alertes Brenk (ex. acide rétinoïque = Michael acceptor). Le contexte biologique prime.

### 3.4 Diversité interne de la bibliothèque

| Métrique | Valeur |
|----------|--------|
| Tanimoto moyen intra-bibliothèque | **0.100** |
| Diversité (1 − Tanimoto moyen) | **0.900** |

Une diversité de 0.90 est **excellente** pour une bibliothèque de 60 composés. À titre de comparaison, une bibliothèque combinatoire classique autour d'un même scaffold donne typiquement une diversité de 0.4–0.6. La figure `figures/08_internal_diversity.png` montre la heatmap complète : quasi aucun cluster visible, confirmant que les deux générateurs (GraphGA et BRICS-DQN) explorent des espaces chimiques complémentaires.

**Ce que cela implique pour le screening :** Une bibliothèque diverse maximise la couverture de l'espace pharmacologique et réduit la redondance en cas de screening expérimental. C'est un argument pour présenter un sous-ensemble représentatif de 10–15 composés à un chimiste médicinal.

### 3.5 Score de qualité pondéré (IC50-agnostique)

La requalification des candidats utilise un score composite **sans IC50** :

```
Quality Score = 0.30 × QED
              + 0.25 × SA_norm      (1 = facile, 0 = SA=10)
              + 0.20 × diversité_locale  (1 − Tanimoto moyen vs reste de la lib)
              + 0.25 × medchem_clean     (1 si tous filtres passés, 0 sinon)
```

**Top 10 candidats (classement IC50-agnostique) :**

| Rang | ID | Source | Score | QED | SA | Medchem OK | Tanimoto max CCLE | Drogue proche |
|------|----|--------|-------|-----|----|------------|-------------------|---------------|
| 1 | BRI-46 | BRICS-DQN | 0.925 | 0.903 | 1.89 | ✓ | 0.190 | Belinostat |
| 2 | BRI-12 | BRICS-DQN | 0.916 | 0.850 | 1.68 | ✓ | 0.204 | Belinostat |
| 3 | BRI-58 | BRICS-DQN | 0.900 | 0.858 | 2.27 | ✓ | 0.318 | GW441756 |
| 4 | Gra-9 | GraphGA | 0.889 | 0.926 | 3.25 | ✓ | 0.200 | TubastatinA |
| 5 | Gra-1 | GraphGA | 0.885 | 0.872 | 2.85 | ✓ | 0.261 | UNC1215 |
| 6 | BRI-48 | BRICS-DQN | 0.884 | 0.849 | 2.72 | ✓ | 0.280 | Pyrimethamine |
| 7 | Gra-2 | GraphGA | 0.882 | 0.784 | 2.15 | ✓ | 0.275 | Elesclomol |
| 8 | Gra-6 | GraphGA | 0.871 | 0.875 | 3.34 | ✓ | 0.227 | Pevonedistat |
| 9 | BRI-36 | BRICS-DQN | 0.871 | 0.895 | 3.89 | ✓ | 0.173 | SB505124 |
| 10 | Gra-3 | GraphGA | 0.869 | 0.733 | 1.98 | ✓ | 0.271 | SB52334 |

**Analyse des top candidats :**

- **BRI-46** (`O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1`) : sulfonamide naphtyl-pipérazine, QED exceptionnel (0.903), SA très bas (1.89 — synthèse en ~2 étapes), aucun flag medchem. Profil comparable à des inhibiteurs kinase de première génération. Le groupement sulfonamide est un pharmacophore validé (sulfamides, inhibiteurs CA, etc.).
- **BRI-12** (`NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1`) : amino-sulfonamide biaryle, le plus facile à synthétiser de la bibliothèque (SA = 1.68). Proche structurellement des sulfonamides HDAC.
- **BRI-58** (`O=C1Nc2cccnc2N(CCO)c2ccccc21`) : lactame tricyclique avec amino-éthanol, proche de GW441756 (inhibiteur TrkA). Potentiellement intéressant pour des cibles kinases.
- **Gra-9** (`CC(C)CN1CCCN(C)CC(c2ccccc2CO)C1C`) : pipérazine alkylée avec phényl-méthanol, QED le plus élevé de la liste (0.926). Profil pharmacocinétique favorable.

---

## 4. Leviers d'amélioration proposés (implémentés ou planifiés)

### Levier 1 — Early stopping (implémenté)
**Problème :** Le modèle Bi-Int sur-apprend sur le split random (val loss remonte après l'epoch 2–3 sur LDO).  
**Solution :** Ajout d'un early stopping avec patience=3 sur la validation LDO.  
**Impact attendu :** Réduction de l'overfitting ; sélection automatique du meilleur checkpoint.

### Levier 2 — Régularisation (implémenté)
**Problème :** Dropout faible (0.1) et pas de weight decay dans la configuration de base.  
**Solution :** Dropout 0.3 + L2 weight decay 1e-4.  
**Impact attendu :** Meilleure généralisation hors distribution, notamment LDO.

### Levier 3 — Augmentation de données par taille (planifié)
**Problème :** Entraînement sur 20k triplets (sous-échantillon du dataset complet 103k) pour contraintes RAM/GPU.  
**Solution :** Augmenter progressivement à 50k puis 100k triplets avec gestion mémoire optimisée.  
**Impact attendu :** Réduction du biais de sélection, meilleure couverture de l'espace drogue.

### Levier 4 — Augmentation SMILES (implémenté dans scripts/smiles_augmentation.py)
**Problème :** Chaque drogue est représentée par un unique SMILES canonique. Le GNN apprend une représentation figée sans variabilité.  
**Solution :** Génération de N=4 SMILES aléatoires par drogue (randomisation de l'ordre des atomes via RDKit), uniquement sur le set d'entraînement pour éviter toute fuite d'information sur les drogues de validation LDO.  
**Impact attendu :** Amélioration de la robustesse de l'encodeur GNN, meilleure généralisation aux nouvelles structures.

### Levier 5 — Ablation LDO (script prêt : scripts/ldo_ablation.py)
**Problème :** Difficile de savoir quelle combinaison de leviers apporte le gain marginal le plus élevé.  
**Solution :** Runner d'ablation automatique qui teste 5 configurations (baseline → +ES → +reg → +data_size → +SMILES augmentation) et produit un tableau comparatif + figure.  
**Statut :** Script finalisé, runs en attente d'une session GPU dédiée.

---

## 5. Solutions proposées et prochaines étapes

### Priorité 1 — Validation expérimentale des top candidats

Les 3 candidats prioritaires pour validation in vitro :

| Candidat | SMILES | Rationale |
|----------|--------|-----------|
| **BRI-46** | `O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1` | Meilleur score global, SA 1.89, synthèse facile |
| **BRI-12** | `NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1` | Plus facile à synthétiser (SA 1.68), aucun flag |
| **BRI-58** | `O=C1Nc2cccnc2N(CCO)c2ccccc21` | Proche GW441756 (TrkA), lactame tricyclique attractif |

Protocole suggéré : docking moléculaire sur cibles CCLE prioritaires (EGFR, HDAC1/6, TrkA), puis test de viabilité cellulaire (IC50 expérimental, MTT assay) sur 3–5 lignées CCLE représentatives.

### Priorité 2 — Amélioration du prédicteur LDO

La voie la plus prometteuse à court terme est **XGBoost avec features omiques améliorées** (r = 0.367, déjà supérieur au deep learning en LDO). Pour le Bi-Int, les leviers 1+2+4 combinés pourraient porter le LDO r vers 0.4–0.45 — à valider par l'ablation.

### Priorité 3 — Amélioration de la génération BRICS-DQN

- Validité actuelle ~60% : implémenter une pénalité de valence dans la reward function
- Ajouter des contraintes de diversité interne directement dans le reward (pénalité si Tanimoto > 0.5 vs candidats déjà acceptés dans l'épisode)
- Explorer des architectures actor-critic (PPO) pour une exploration plus stable

### Priorité 4 — Mapping SMILES complet

65 drogues CCLE restent sans SMILES fiable. Résoudre ce problème augmenterait le dataset d'entraînement de ~30% (passage de 201 à 266 drogues mappées), avec un impact direct sur la couverture LDO.

---

## 6. Interprétation des figures principales

Les figures suivantes résument visuellement les résultats, les risques et les priorités du projet. Elles permettent à un expert du domaine de comprendre rapidement ce qui a été fait, et pourquoi les conclusions restent prudentes.

### Figure 01 — Structures moléculaires des candidats GraphGA
- Montre les 10 meilleures molécules générées par GraphGA.
- Confirme visuellement la présence de scaffolds hétérocycliques et de motifs sulfonamide, pipérazine et carbamate.
- Interprétation : ces structures sont chimiques plausibles et compatibles avec une synthèse de chimie médicinale, mais elles restent suffisamment originales pour ne pas être des analogues proches de CCLE.

### Figure 02 — Courbes d'entraînement QSAR
- Affiche la RMSE d'entraînement et de validation, ainsi que Pearson r par epoch.
- Interprétation : le modèle performe bien sur le split random, mais la validation LDO montre une divergence croissante après 2–3 epochs, indiquant un surapprentissage sur des drogues vues.
- Conséquence : le meilleur checkpoint doit être sélectionné en priorité sur LDO, pas sur le split random.

### Figure 03 — Reward BRICS-DQN sur 5 000 épisodes
- Montre l'évolution de la récompense de l'agent DQN pendant l'entraînement.
- Interprétation : le signal de reward augmente de façon significative, ce qui montre que l'agent apprend des fragments BRICS utiles.
- Limite : une récompense plus élevée ne garantit pas la validité chimique, d'où le besoin de la validation par PAINS/Brenk et SA.

### Figure 04 — Distribution QED et propriétés Lipinski
- Compare QED, poids moléculaire et logP des candidats GraphGA.
- Interprétation : la majorité des candidats se situe dans une zone acceptable de druglikeness (QED > 0.7, MW < 500, logP modéré), renforçant leur attractivité pour une phase de screening précoce.

### Figure 05 — Dashboard synthétique
- Résume en une seule image les métriques clés : performance QSAR, reward DQN, scores QED, validité, et principales alertes.
- Interprétation : ce tableau de bord confirme la cohérence des pipelines et met en évidence les compromis entre originalité, synthétisabilité et prédictions IC50.

### Figure 06 — Distribution Tanimoto vs CCLE
- Montre la similarité maximale des candidats générés par rapport aux drogues CCLE.
- Interprétation : presque tous les candidats ont une similarité < 0.30, ce qui signifie qu'ils explorent un espace moléculaire nouveau par rapport à l'entraînement.
- Ce résultat est crucial : il valide la génération de composés innovants, mais il renforce également l'incertitude liée aux prédictions IC50 hors distribution.

### Figure 07 — Ablation LDO (Bi-Int vs baselines)
- Compare les performances LDO du Bi-Int et des modèles de référence (XGBoost, Ridge, RF, etc.).
- Interprétation : le Bi-Int reste sous-optimal face à XGBoost en LDO, ce qui oriente les efforts d'amélioration vers la régularisation, l'augmentation de données et l'optimisation du split.
- Message pour l'expert : le modèle profond n'est pas encore suffisamment robuste pour remplacer une baseline bien calibrée en hors-distribution.

### Figure 08 — Heatmap de diversité interne
- Montre la similarité Tanimoto entre toutes les molécules générées.
- Interprétation : la bibliothèque est très diverse (Tanimoto moyen ≈ 0.10), ce qui est un atout pour le screening et réduit les redondances.
- Limite : une diversité trop élevée peut aussi signifier un manque de focus autour de cibles pharmaceutiques bien connues, donc il faut équilibrer originalité et réalisme chimique.

### Figure 13 — Applicability domain
- Montre le domaine d'applicabilité du modèle sur les molécules d'entraînement vs nouvelles molécules.
- Interprétation : les composés hors domaine doivent être traités avec prudence, car le modèle n'a pas de garantie de performance en dehors de l'espace chimique appris.
- Recommandation : utiliser cette figure pour établir un seuil d'alerte dans les futures exportations de candidats.

---

## 7. Limitations connues et mises en garde

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| LDO r = 0.316 (Bi-Int) | Prédictions IC50 peu fiables pour nouvelles drogues | Ne pas classer par IC50 prédit ; utiliser le score qualité composite |
| XGBoost > Bi-Int sur LDO | Deep learning sous-optimal avec données actuelles | Ablation en cours ; augmentation données |
| 65 drogues sans SMILES | Dataset partiel, biais de couverture | PubChem API lookup à compléter |
| Validité BRICS-DQN ~60% | Une molécule sur deux générée est invalide | Pénalité de valence dans reward function |
| SA score ≠ synthèse réelle | sascorer est un estimateur heuristique | Validation rétrosynthétique manuelle (AiZynthFinder, ASKCOS) |
| Tanimoto < 0.3 pour tous les candidats | Haute originalité = haute incertitude ADMET | ADMET in silico (SwissADME, pkCSM) avant validation expérimentale |

---

## 7. Fichiers produits cette session

| Fichier | Description |
|---------|-------------|
| `scripts/molecular_validation.py` | Script de validation chimique complet (SA, PAINS, Brenk, Lipinski, Veber, Tanimoto, NP-likeness, diversité interne, quality score) |
| `Dataset/molecular_validation_report.csv` | Rapport CSV : 60 candidats × 24 métriques, classés par quality score |
| `figures/08_internal_diversity.png` | Heatmap de similarité interne de la bibliothèque |
| `docs/rapport_31mai2026.md` | Ce rapport |

---

## Annexe — Paramètres techniques de la validation

| Paramètre | Valeur |
|-----------|--------|
| Morgan fingerprint radius | 2 |
| Morgan fingerprint bits | 2048 |
| CCLE reference drugs (avec SMILES valide) | 184 / 201 |
| PAINS catalog version | RDKit FilterCatalog (PAINS A/B/C) |
| Brenk catalog | RDKit FilterCatalog (BRENK) |
| SA scorer | RDKit Contrib sascorer v1.0 |
| NP-likeness | RDKit Contrib npscorer |
| Quality score | QED×0.30 + SA_norm×0.25 + diversity×0.20 + medchem×0.25 |
| Données source GraphGA | `Dataset/graphga_tanimoto_vs_ccle.csv` (10 candidats) |
| Données source BRICS-DQN | `Dataset/brics_dqn_results.csv` (top 50 par reward > 0) |
