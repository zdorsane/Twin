# Rapport de session — Twin Project
**Date :** 31 mai & 1–2 juin 2026 (mis à jour le 2 juin 2026)
**Auteur :** Zied Dorsane
**Superviseur :** M. Marouane
**Modèle :** Bi-Interaction (GNN + Quaternion-VAE) — prédiction IC50 CCLE + génération de novo

---

## Vue d'ensemble

Ce rapport couvre l'ensemble du travail réalisé sur le projet Twin depuis la session du
31 mai 2026, incluant les mises à jour du 1er et 2 juin 2026. Il documente :

1. **Validation chimique/biologique** des molécules générées (31 mai)
2. **Infrastructure de fiabilité** : sauvegarde checkpoint, correction VAE, alignement features (1er juin)
3. **Analyses d'interprétabilité** : biomarqueurs ncRNA et codants, domaine d'applicabilité, incertitude MC Dropout (1er juin)

Pour l'interprétation détaillée de chaque figure, voir
[docs/FIGURE_INTERPRETATIONS.md](FIGURE_INTERPRETATIONS.md).

---

## 1. Architecture et performances du modèle prédicteur

### 1.1 Architecture Bi-Int

Le modèle Twin combine trois encodeurs spécialisés :
- Un **encodeur GNN** sur les SMILES (pré-entraîné sur ChEMBL, puis fine-tuné sur CCLE)
- Un **VAE quaternionique** encodant les profils omiques des lignées cellulaires
  (978 gènes GEx + 426 CNA + 735 mutations → vecteur latent z ∈ ℝ¹²⁸)
- **4 blocs d'attention croisée bipartite (Bi-Int)** fusionnant les représentations
  drogue et omique par attention croisée symétrique + mises à jour triangulaires

**Paramètres :** 9 255 070 paramètres entraînables.
**Inférence :** déterministe — le VAE retourne µ (moyenne posterieure) sans
reparamétrisation stochastique en `training=False`.

### 1.2 Résultats de prédiction IC50

| Split | Modèle | Pearson r | IC 95% | Notes |
|-------|--------|-----------|--------|-------|
| Random | Bi-Int (epoch 4) | **0.811** | [0.736, 0.886] | Interpolation — même drogues train/val |
| Leave-Drug-Out | **XGBoost** | **0.367** | [0.338, 0.393] | Meilleure généralisation |
| Leave-Drug-Out | Bi-Int (epoch 2) | 0.316 | [0.287, 0.344] | Généralisation structurelle |
| Leave-Drug-Out | RF (50 trees) | 0.231 | [0.202, 0.259] | |
| Leave-Drug-Out | Ridge ECFP4+omics | 0.228 | [0.196, 0.256] | |
| Leave-Cell-Out | Bi-Int (epoch 4) | 0.766 | — | Run partiel (6 epochs) |

**Interprétation des splits :**

- **Random (r = 0.811) :** les mêmes drogues apparaissent dans train et validation.
  Le modèle mémorise des corrélations drogue-spécifiques. Ce chiffre mesure la
  capacité d'interpolation, pas la généralisation.
- **Leave-Drug-Out (r = 0.316) :** les drogues de validation n'ont jamais été vues à
  l'entraînement. C'est la métrique honnête pour évaluer la capacité du modèle à
  prédire la réponse à de nouvelles structures chimiques. Un r = 0.316 indique une
  corrélation faible mais statistiquement significative (p << 0.001, n > 3 000).
  XGBoost (r = 0.367) surpasse le modèle profond sur ce split, ce qui indique que la
  complexité du Bi-Int n'est pas encore justifiée à cette échelle de données (16k
  triplets d'entraînement).

> ⚠️ **Les IC50 prédits pour les molécules générées ne doivent pas être interprétés
> comme des prédictions fiables de potency.** Ces molécules sont structurellement
> nouvelles (Tanimoto < 0.30 vs drogues CCLE), le modèle opère hors distribution.
> Validation in vitro indispensable.

---

## 2. Génération de molécules — deux approches

### 2.1 GraphGA (Algorithme génétique sur graphes moléculaires)

Optimisation guidée par QED + IC50 prédit sur une population de molécules.
**10 candidats** générés, QED moyen 0.833, tous Lipinski-compliant.

### 2.2 BRICS-DQN (Reinforcement Learning par fragments)

Agent DQN assemblant des fragments BRICS (sous-structures rétrosynthétiquement
accessibles). 3 008 molécules valides sur ~5 000 épisodes, validité ~60%.
**Top 50 sélectionnés** par score de récompense.

---

## 3. Validation moléculaire rigoureuse (31 mai 2026)

### 3.1 Similarité Tanimoto vs drogues CCLE

| Zone Tanimoto | Interprétation | N candidats |
|--------------|----------------|-------------|
| > 0.7 | Analogue proche — peu novateur | 0 / 60 |
| 0.4 – 0.6 | Zone idéale | 0 / 60 |
| 0.3 – 0.4 | Frontière | 2 / 60 |
| < 0.3 | Structurellement nouveau | **58 / 60** |

Similarité maximale : 0.318 (BRI-58 vs GW441756). La bibliothèque est
**structurellement originale** par rapport aux 184 drogues CCLE avec SMILES valide.

### 3.2 Synthetic Accessibility (SA score, sascorer RDKit)

- SA 1–3 (facilement synthétisable) : **40/60 (67%)**
- SA 3–5 (complexe mais faisable) : **18/60 (30%)**
- SA 5–6 (difficile) : **2/60 (3%)**
- SA > 6 (très difficile — flaggé) : **0/60**

Meilleur SA : BRI-12 (SA = 1.68, synthèse estimée en 1–2 étapes).

### 3.3 Filtres MedChem

| Filtre | Candidats échouant | % |
|--------|-------------------|---|
| PAINS (groupes réactifs/promiscuous) | 1 | 1.7% |
| Brenk (toxiques/réactifs) | 18 | 30% |
| Lipinski (Ro5 : MW, logP, HBD, HBA) | 5 | 8.3% |
| Veber (rotatable bonds, TPSA) | 6 | 10% |
| **Tous filtres passés (medchem clean)** | **38/60** | **63%** |

### 3.4 Diversité interne

Tanimoto moyen intra-bibliothèque : **0.10** → diversité = **0.90**.
Exceptionnelle pour une bibliothèque générée — confirme que GraphGA et BRICS-DQN
explorent des espaces chimiques complémentaires.

### 3.5 Top 5 candidats (score qualité IC50-agnostique)

| Rang | ID | Source | Score | QED | SA | SMILES |
|------|----|--------|-------|-----|----|----|
| 1 | BRI-46 | BRICS-DQN | 0.925 | 0.903 | 1.89 | `O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1` |
| 2 | BRI-12 | BRICS-DQN | 0.916 | 0.850 | 1.68 | `NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1` |
| 3 | BRI-58 | BRICS-DQN | 0.900 | 0.858 | 2.27 | `O=C1Nc2cccnc2N(CCO)c2ccccc21` |
| 4 | Gra-9 | GraphGA | 0.889 | 0.926 | 3.25 | `CC(C)CN1CCCN(C)CC(c2ccccc2CO)C1C` |
| 5 | Gra-1 | GraphGA | 0.885 | 0.872 | 2.85 | `CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1` |

---

## 4. Infrastructure de fiabilité (1er juin 2026)

### 4.1 Sauvegarde du checkpoint modèle

**Problème :** `run_pipeline()` ne sauvegardait jamais le modèle entraîné.
Toute inférence post-entraînement nécessitait un réentraînement complet.

**Correction :** ajout dans `src/fullPipeline.py` :
```python
model.save_weights(os.path.join(log_dir, "biint_ic50_model.weights.h5"))  # prioritaire
model.save(os.path.join(log_dir, "biint_ic50_model.keras"))                 # tentative
json.dump(HP, open(os.path.join(log_dir, "hp_snapshot.json"), "w"))         # config
```

**Validation :** `scripts/test_model_loading.py --self-test` → écart max = 0.00e+00
(identité numérique exacte, 37.4 MB, 9 255 070 paramètres).

### 4.2 Correction du bug d'inférence stochastique du VAE

**Problème :** `reparameterize()` appelait `tf.random.normal()` même en
`training=False`. Les prédictions post-`load_weights` différaient du modèle original
(écart max 1.24 en unités z-score).

**Correction :**
```python
def reparameterize(self, mu, log_var, training=False):
    if not training:
        return mu  # inférence déterministe sur la moyenne posterieure μ
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps
```

### 4.3 Alignement features GEx (désalignement silencieux)

**Problème :** l'index du CSV RNA-seq contient 1 965 gènes dupliqués. La
recomputation de `top_gex` sans déduplication produisait 988 colonnes au lieu
de 978, causant un `ValueError` silencieux à la forward pass.

**Correction :** `scripts/_ccle_loader.py` charge directement depuis le cache NPZ
(matrices identiques à l'entraînement). Vérification : corrélation = 1.0000 entre
la liste recalculée et le cache.

---

## 5. Analyses d'interprétabilité (1er juin 2026)

Toutes calculées sur le checkpoint LDO (r = 0.210, epoch 1).
Méthode : **Gradient × Input** sur GEx (978 dimensions), 150 paires validation LDO.

> **Limite :** un modèle à r = 0.210 peut apprendre des corrélations artéfactuelles.
> Ces résultats seront recalculés sur le checkpoint random (r = 0.811).

### 5.1 Biomarqueurs ncRNA (Figure 09–10)

76 transcrits non-codants identifiés parmi les 978 features. Résultats :

- **Top-1 ncRNA :** RNU1-28P (rang 34/978) — pseudogène snRNA U1, rôle non documenté
- **VIM-AS1 :** rang 8/76 — transcrit antisens de Vimentine (EMT — pertinent biologiquement)
- **H19 :** rang 61/76 (rang 769/978), importance 0.000323
- **GAS5 :** rang 70/76 (rang 916/978), importance 0.000090

H19 et GAS5 sont en bas du classement car le modèle LDO (r=0.210) n'a pas convergé
vers les mécanismes biologiques pertinents. Ces résultats seront réévalués sur le
checkpoint random.

### 5.2 Biomarqueurs codants (Figure 11)

902 gènes codants analysés. Top-5 : CCND3, OST4, ARL6IP1, CTGF, THY1.

**Biomarqueurs oncologiques canoniques dans le top-20 : 0/20.**

Gènes avec pertinence biologique indirecte : CTNNB1 (β-caténine, voie Wnt),
AREG (ligand EGFR, résistance anti-EGFR), SQSTM1 (autophagie, résistance chimio).

### 5.3 Domaine d'applicabilité (Figure 13)

Tanimoto max (Morgan FP, r=2, 2048 bits) de chaque drogue de validation vs
161 drogues d'entraînement LDO :

| Niveau | Seuil | N drogues val | % |
|--------|-------|-------------|---|
| 🟢 RELIABLE | ≥ 0.6 | 4 | 10% |
| 🟡 CAUTION | 0.4–0.6 | 4 | 10% |
| 🔴 UNRELIABLE | < 0.4 | **32** | **80%** |

**Résultat attendu en LDO** — confirme que le split est rigoureux. L'alerte
Tanimoto est opérationnelle en production dès maintenant, indépendamment de la
performance du modèle.

### 5.4 Incertitude MC Dropout (Figure 12)

200 paires × 30 passes forward (dropout=10%, training=True) :

- Seuil d'alerte (σ) : **0.1975**
- HIGH_UNCERTAINTY : **11/200 (5.5%)**

Le modèle est **trop confiant** malgré r=0.210. Dropout à 10% insuffisant pour
une estimation bayésienne robuste. Combinaison obligatoire avec l'alerte Tanimoto.

---

## 6. Figures produites

| Figure | Fichier | Description |
|--------|---------|-------------|
| 01 | `figures/phase1_training_generation/01_molecular_structures.png` | Structures RDKit 2D des candidats GraphGA |
| 02 | `figures/phase1_training_generation/02_training_curves.png` | Courbes d'entraînement QSAR (random split) |
| 03 | `figures/phase1_training_generation/03_dqn_reward.png` | Reward BRICS-DQN sur 5 000 épisodes |
| 04 | `figures/phase1_training_generation/04_qed_lipinski.png` | QED et propriétés Lipinski (GraphGA top-10) |
| 05 | `figures/phase1_training_generation/05_dashboard.png` | Dashboard synthétique |
| 06 | `figures/phase2_validation_ablation/06_tanimoto_distribution.png` | Tanimoto candidats vs CCLE |
| 07 | `figures/phase2_validation_ablation/07_ldo_ablation.png` | Ablation LDO : Bi-Int vs baselines |
| 08 | `figures/phase2_validation_ablation/08_internal_diversity.png` | Heatmap diversité interne 60 candidats |
| 09 | `figures/phase3_interpretability_reliability/09_ncrna_importance.png` | ★ Top-20 ncRNA par importance Gradient×Input |
| 10 | `figures/phase3_interpretability_reliability/10_ncrna_vs_drugs.png` | ★ Heatmap importance ncRNA × drogues |
| 11 | `figures/phase3_interpretability_reliability/11_coding_biomarkers.png` | ★ Top-20 gènes codants, biomarqueurs surlignés |
| 12 | `figures/phase3_interpretability_reliability/12_uncertainty_distribution.png` | ★ MC Dropout : distribution σ et IC 95% |
| 13 | `figures/phase3_interpretability_reliability/13_applicability_domain.png` | ★ Domaine d'applicabilité Tanimoto |

★ = nouvelles figures (1er juin 2026)

---

## 7. Prochaines étapes prioritaires

1. **Relancer biomarqueurs (Figs 09–11) sur checkpoint random (r = 0.811)** —
   attendu : EGFR, KRAS, BRAF dans le top-20 codants.
2. **Rapport de fiabilité consolidé** : pour 3 paires représentatives (fiable /
   prudence / hors domaine), combiner IC50 prédit ± IC, alerte Tanimoto, σ Dropout.
3. **Relancer ablation LDO** depuis `src/` : early stopping, dropout 0.3, GNN freeze,
   40k triplets.
4. **Finaliser run LCO** pour grille de comparaison complète (random / LDO / LCO).

---

## 8. Limitations et mises en garde

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| LDO r = 0.316, XGBoost r = 0.367 | Deep learning sous-optimal | Régularisation, augmentation données |
| Biomarqueurs sur checkpoint LDO (r=0.210) | Attributions peu fiables biologiquement | Recalcul sur checkpoint random |
| MC Dropout trop confiant (5.5% alertes) | Sous-estimation incertitude | Augmenter dropout, combiner avec Tanimoto |
| 80% drogues LDO hors domaine | Prédictions non fiables pour nouvelles drogues | Alerte Tanimoto systématique |
| 65/266 drogues sans SMILES | Couverture partielle | PubChem API lookup à compléter |
| Validité BRICS-DQN ~60% | Molécules invalides | Pénalité valence dans reward function |
| SA score heuristique | ≠ synthèse réelle | AiZynthFinder, ASKCOS pour validation |

---

## Annexe — Paramètres clés

| Paramètre | Valeur |
|-----------|--------|
| GEx features | 978 gènes top-variance (après déduplication) |
| CNA features | 426 gènes top-variance |
| Mutations features | 735 gènes top-mutés |
| Latent dim VAE | 128 |
| Bi-Int blocs | 4 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Dropout | 10% (à augmenter) |
| Loss mode | cross_entropy (meilleur sur CCLE benchmark) |
| Pré-entraînement | ChEMBL GNN (drogue encoder uniquement) |
| Dataset | CCLE Broad 2019, sous-échantillon 20k / 103k triplets |
| Attribution method | Gradient × Input sur GEx |
| MC Dropout N | 30 passes |
| Tanimoto FP | Morgan, rayon 2, 2048 bits |
