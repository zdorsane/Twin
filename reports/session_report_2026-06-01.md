# Rapport de session — 31 mai & 1er juin 2026
## Bi-Int : Infrastructure de fiabilité, analyses d'interprétabilité & biomarqueurs thérapeutiques

**Projet :** Twin — Prédicteur multimodal de réponse aux drogues (CCLE)  
**Auteur :** Zied Dorsane  
**Superviseur :** M. Marouane  
**Commits couverts :** `8f270f8` → `6e17f2e` (31 mai–1er juin 2026)  
**Matériel :** NVIDIA RTX 4000 Ada, 20 475 MiB VRAM — Ubuntu 24.04 LTS (WSL2)  
**Figures produites aujourd'hui :** 09, 10, 11, 12, 13 (+ figures 01–08 existantes)

---

## 1. Résumé exécutif

La journée du 1er juin a porté sur **deux axes** : (1) corriger les problèmes
d'infrastructure bloquants qui empêchaient toute analyse post-entraînement, et
(2) produire les premières analyses d'interprétabilité et d'estimation d'incertitude
sur le modèle Bi-Int entraîné. Cinq nouvelles figures ont été générées (09–13).

**Résultats clés :**
- Le modèle Bi-Int pondère davantage des loci génomiques non-annotés (RP11-, AC-, RNU)
  que les lncRNA classiques (H19 rang 61/76, GAS5 rang 70/76) — cohérent avec une
  convergence partielle sur le split LDO (r = 0.210).
- 80 % des drogues de validation sont hors du domaine d'applicabilité (Tanimoto < 0.4
  vs drogues d'entraînement) — résultat attendu en LDO mais quantifié pour la première
  fois.
- L'incertitude MC Dropout est faible (seuil 0.198, 5.5 % de paires à haute
  incertitude) — le modèle est trop confiant, ce qui confirme le besoin d'alertes
  d'applicabilité.
- **Prochaine étape prioritaire :** relancer les biomarqueurs sur le checkpoint
  random (r = 0.811) pour des attributions biologiquement interprétables.

---

## 2. Travail du 31 mai — Livrables et restructuration (rappel)

### 2.1 Fonctionnalités vérifiées

| # | Fonctionnalité | État | Preuve |
|---|---------------|------|--------|
| 1 | Mapping SMILES via PubChem | ✅ | `Dataset/ccle_drug_smiles.csv` — 201/266 drogues |
| 2 | Chargement mutations (MAF) | ✅ | Parser MAF — 735 gènes, 647/647 lignées couvertes |
| 3 | Splits Random / LDO / LCO | ✅ | `--split-mode` CLI — 3 modes opérationnels |
| 4 | Baselines RF / XGBoost / MLP / Ridge | ✅ | `src/baseline_models.py` + CI bootstrap — 12 résultats |
| 5 | BRICS-DQN | ✅ | `src/brics_dqn_optimizer.py` — 50 épisodes générés |
| 6 | Terminologie prudente | ✅ | README et rapports : « prototype de recherche » |

### 2.2 Scripts livrés

| Script | Fonction |
|--------|----------|
| `scripts/ldo_ablation.py` | Ablation LDO systématique |
| `scripts/bootstrap_ci.py` | IC bootstrap n=1 000 sur toutes baselines |
| `scripts/tanimoto_analysis.py` | Similarité Tanimoto GraphGA vs CCLE |
| `scripts/molecular_validation.py` | Validation chimique (SA, PAINS, Brenk, Lipinski) |
| `scripts/smiles_augmentation.py` | Enumération SMILES |

### 2.3 Diagnostic ncRNA

76 transcrits non-codants identifiés parmi les 978 gènes top-variance du modèle,
dont H19 (index 48) et GAS5 (index 545). Analyse d'importance lancée aujourd'hui.

---

## 3. Travail du 1er juin — Infrastructure et analyses

### 3.1 Problèmes critiques corrigés

#### A. Absence de sauvegarde du modèle entraîné

**Problème :** `run_pipeline()` produisait uniquement des CSV de métriques. Toute
inférence post-entraînement nécessitait un réentraînement complet.

**Correction :** ajout dans `src/fullPipeline.py` après `trainer.fit()` :
```python
model.save_weights(os.path.join(log_dir, "biint_ic50_model.weights.h5"))
model.save(os.path.join(log_dir, "biint_ic50_model.keras"))
json.dump(HP, open(os.path.join(log_dir, "hp_snapshot.json"), "w"))
```
Argument `--save-model` / `--no-save-model` ajouté au CLI.

#### B. Inférence stochastique du VAE

**Problème :** `reparameterize()` appelait `tf.random.normal()` même en
`training=False`. Résultat : après `load_weights`, les prédictions différaient du
modèle original (écart maximal 1.24 sur une échelle z-score de σ=1).

**Correction :**
```python
def reparameterize(self, mu, log_var, training=False):
    if not training:
        return mu  # inférence déterministe sur la moyenne posterieure
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps
```

**Validation :** `scripts/test_model_loading.py --self-test` → écart max = 0.00e+00
(identité numérique exacte, 37,4 MB, 9 255 070 paramètres).

#### C. Désalignement des features GEx lors du chargement

**Problème :** l'index du CSV RNA-seq contient 1 965 gènes dupliqués. Les scripts
d'analyse qui reconstruisaient `top_gex` à partir du CSV brut obtenaient 988 colonnes
au lieu de 978, causant un `ValueError` à la forward pass du modèle.

**Correction :** création de `scripts/_ccle_loader.py` qui charge directement depuis
le cache NPZ (`omics_cache_gex978_cna426.npz`) — même matrice exacte que celle vue
pendant l'entraînement — et de `Dataset/top978_gex_genes.txt` (liste de gènes
vérifiée, corrélation = 1.0000 avec le cache).

### 3.2 Run LDO avec checkpoint

```
src/fullPipeline.py --mode pretrained --loss-mode cross_entropy
  --split-mode leave_drug_out --epochs 15 --early-stopping 3
  --log-dir logs/ldo_checkpoint --no-ppo --save-model
```

| Epoch | Val RMSE | Pearson r |
|-------|----------|-----------|
| 1 | 1.017 | 0.210 ← meilleur |
| 2 | 1.107 | 0.193 |
| 3 | 1.228 | 0.049 |
| 4 | 1.273 | 0.130 → arrêt (patience=3) |

Checkpoint sauvegardé : `logs/ldo_checkpoint/biint_ic50_model.weights.h5` (37 MB).
Vérification : LOAD TEST PASSED, prédictions sans NaN/Inf.

---

## 4. Analyses d'interprétabilité et de fiabilité (figures 09–13)

Toutes les analyses utilisent le checkpoint LDO (r = 0.210, epoch 1).  
Méthode d'attribution : **Gradient × Input** sur le vecteur GEx (978 dimensions),
150 paires (drogue, lignée) échantillonnées aléatoirement dans le split de validation.

> ⚠️ **Limite importante :** les attributions présentées ici sont calculées sur un
> modèle LDO à r = 0.210. Un modèle peu performant peut apprendre des corrélations
> artéfactuelles. Ces résultats sont à interpréter avec prudence ; les biomarqueurs
> seront recalculés sur le checkpoint random (r = 0.811) pour comparaison.

---

### 4.1 Figure 09 — Importance des transcrits non-codants (ncRNA)

**`figures/09_ncrna_importance.png`**

**Méthode :** pour chaque paire de validation (drogue *d*, lignée cellulaire *c*),
on calcule le gradient de la prédiction IC50 par rapport au vecteur d'expression
génique, puis on multiplie par la valeur d'entrée (Gradient × Input). L'importance
d'un gène est la moyenne des valeurs absolues sur les 150 paires. Ce score mesure
à quel point une variation d'expression de ce gène change la prédiction du modèle.
On restreint ensuite aux 76 transcrits non-codants identifiés parmi les 978 features.

**Résultats :**

| Rang/76 | Gène | Rang/978 | Importance | Catégorie |
|---------|------|----------|------------|-----------|
| 1 | RNU1-28P | 34 | 0.001350 | snRNA U1 (pseudogène) |
| 2 | RMRP | 57 | 0.001221 | ARN ribonucléase MRP |
| 3 | RP11-3P17.5 | 69 | 0.001184 | locus génomique non-annoté |
| 4–20 | RP11-*, AC*, VIM-AS1… | 98–271 | — | loci non-annotés |
| **61** | **H19** | **769** | **0.000323** | **oncogène — résistance chimio** |
| **70** | **GAS5** | **916** | **0.000090** | **suppresseur de tumeur** |

**Interprétation pour l'expert :** le modèle pondère principalement des loci
génomiques anonymes (nomenclature RP11-, AC-, AL-, Z-) qui correspondent à des
transcrits non-référencés dans les bases fonctionnelles. Ces loci ont une variance
d'expression élevée entre lignées cellulaires, ce qui les rend informatifs
statistiquement sans nécessairement refléter un mécanisme biologique. H19 et GAS5,
les deux lncRNA à rôle oncologique établi, se trouvent en bas du classement (rangs
61 et 70/76). Cela indique que le modèle, insuffisamment convergé sur LDO (r = 0.210),
n'a pas appris à exploiter les lncRNA biologiquement pertinents. Ce résultat est
**attendu et non alarmant** : il justifie l'étape suivante, à savoir relancer
l'analyse sur le checkpoint random (r = 0.811) où le modèle a réellement convergé.

---

### 4.2 Figure 10 — Heatmap importance ncRNA × drogues

**`figures/10_ncrna_vs_drugs.png`**

Chaque cellule représente l'importance normalisée d'un ncRNA (colonnes, top-10) pour
une drogue donnée (lignes, top-15 par importance ncRNA moyenne). La normalisation est
par colonne (0 = importance minimale pour ce ncRNA, 1 = maximale).

**Interprétation :** la heatmap permet d'identifier si certaines drogues activent
préférentiellement certains ncRNA. Une heatmap uniforme (toutes colonnes similaires
entre drogues) indiquerait que le modèle ne distingue pas les profils ncRNA selon
la drogue — ce qui serait cohérent avec un modèle peu convergé. Une heatmap
différenciée (certaines drogues fortement spécifiques à certains ncRNA) suggérerait
une spécificité drug-ncRNA apprise. L'interprétation fine nécessite de regarder
la figure.

---

### 4.3 Figure 11 — Importance des gènes codants

**`figures/11_coding_biomarkers.png`**

**Résultats :**

| Rang | Gène | Importance | Biomarqueur connu |
|------|------|------------|-------------------|
| 1 | CCND3 | 0.001902 | — (cycline D3, prolifération) |
| 2 | OST4 | 0.001892 | — (sous-unité oligosaccharyltransférase) |
| 3 | ARL6IP1 | 0.001867 | — (trafic vésiculaire) |
| 4 | CTGF | 0.001826 | — (facteur de croissance du tissu conjonctif) |
| 5 | THY1 | 0.001783 | — (marqueur cellules souches/fibrose) |
| 8 | SQSTM1 | 0.001660 | — (autophagie, stress oxydatif) |
| 13 | AREG | 0.001551 | — (ligand EGF, résistance aux anti-EGFR) |
| 18 | APP | 0.001498 | — (précurseur amyloïde) |
| 24 | CTNNB1 | 0.001411 | — (β-caténine, voie Wnt) |

**Biomarqueurs oncologiques canoniques (EGFR, KRAS, TP53…) dans top-20 : 0/20.**

**Interprétation pour l'expert :** l'absence de biomarqueurs oncologiques classiques
dans le top-20 est le marqueur le plus clair d'une convergence insuffisante du modèle
sur LDO. Un modèle ayant appris la biologie de la pharmacorésistance devrait pondérer
fortement EGFR pour les inhibiteurs d'EGFR, BRAF pour les inhibiteurs de BRAF, etc.
Quelques gènes du top-20 ont néanmoins une pertinence biologique indirecte : CTNNB1
(β-caténine, voie Wnt — impliquée dans la résistance aux taxanes), AREG (ligand EGF
— surexprimé dans les tumeurs résistantes aux anti-EGFR), SQSTM1 (autophagie —
mécanisme de résistance aux chimiothérapies). Ces co-occurrences restent à confirmer
sur un modèle mieux convergé.

---

### 4.4 Figure 12 — Incertitude MC Dropout

**`figures/12_uncertainty_distribution.png`**

**Méthode :** pour 200 paires de validation, on effectue N=30 passages forward
avec `training=True` (dropout actif). On calcule la moyenne, l'écart-type et
l'intervalle de confiance 95 % des 30 prédictions. Le seuil d'alerte est fixé à
médiane + 1×écart-type des écarts-types observés.

**Résultats :**

| Métrique | Valeur |
|----------|--------|
| Seuil d'alerte (std) | 0.1975 |
| Paires HIGH_UNCERTAINTY | 11 / 200 (5,5 %) |
| Paires OK | 189 / 200 (94,5 %) |

**Interprétation pour l'expert :** un taux de 5,5 % de paires à haute incertitude
est **anormalement bas** pour un modèle à r = 0.210. Cela indique que le modèle est
**trop confiant** : malgré ses mauvaises prédictions, le dropout ne génère pas de
variance élevée entre les 30 passes. Deux explications possibles : (1) le taux de
dropout est faible (10 % dans HP), insuffisant pour créer une dispersion significative ;
(2) le modèle a appris une représentation peu sensible aux sous-réseaux activés.
Ce résultat souligne l'importance de combiner MC Dropout avec l'alerte Tanimoto :
l'incertitude interne du modèle ne suffit pas à détecter les prédictions hors domaine.
Un modèle peut être certain *et* incorrect lorsqu'il extrapole hors distribution.

---

### 4.5 Figure 13 — Domaine d'applicabilité (Tanimoto)

**`figures/13_applicability_domain.png`**

**Méthode :** pour chaque drogue de validation, on calcule la similarité Tanimoto
maximale (Morgan FP, rayon 2, 2048 bits) avec l'ensemble des drogues d'entraînement.
Seuils : ≥ 0.6 = fiable, 0.4–0.6 = prudence, < 0.4 = hors domaine.

**Résultats sur les 40 drogues de validation LDO :**

| Niveau | Seuil | N | % |
|--------|-------|---|---|
| 🟢 RELIABLE | Tanimoto ≥ 0.6 | **4** | **10 %** |
| 🟡 CAUTION | 0.4 – 0.6 | **4** | **10 %** |
| 🔴 UNRELIABLE | < 0.4 | **32** | **80 %** |

**Interprétation pour l'expert :** ce résultat est **attendu par construction** pour
un split Leave-Drug-Out : les drogues de validation sont choisies pour ne pas se
chevaucher avec les drogues d'entraînement. Le Tanimoto < 0.4 pour 80 % d'entre elles
confirme qu'elles sont structurellement distinctes — c'est précisément la définition
du LDO. L'intérêt de cette analyse n'est donc pas de diagnostiquer un problème, mais
de **quantifier la difficulté de la tâche** : le modèle doit prédire l'IC50 de
molécules dont le scaffold est inédit. Les 4 drogues RELIABLE (Tanimoto ≥ 0.6)
correspondent à des drogues similaires à des analogues vus à l'entraînement
(variantes de même famille chimique) ; ce sont probablement les meilleures prédictions
du modèle sur ce split.

Cette alerte est **immédiatement utilisable en production** : toute nouvelle drogue
soumise au modèle doit passer par ce test Tanimoto avant de présenter sa prédiction.
Si max_tanimoto < 0.4, afficher l'alerte 🔴 HORS DOMAINE — prédiction non fiable.

---

## 5. État d'avancement global

| Composant | État | Note |
|-----------|------|------|
| Architecture Bi-Int | ✅ | GNN + Quaternion-VAE + 4 blocs Bi-Int |
| QSAR random split | ✅ | r = 0.811 [0.736, 0.886] — epoch 4 |
| QSAR LDO | ✅ | r = 0.316 [0.241, 0.391] — epoch 2 (run précédent) |
| QSAR LCO | ⚠️ | 6 epochs, best r = 0.766 epoch 4 — non finalisé |
| Baselines + CI bootstrap | ✅ | 12 modèles × 3 splits |
| Validation chimique GraphGA + BRICS-DQN | ✅ | 60 molécules, 0 PAINS |
| Sauvegarde checkpoint | ✅ | Corrigé + testé (0.00e+00) |
| Inférence déterministe (fix VAE) | ✅ | `reparameterize` retourne μ en `training=False` |
| Biomarqueurs ncRNA (Fig 09–10) | ✅ | Sur checkpoint LDO r=0.210 — à refaire sur random |
| Biomarqueurs codants (Fig 11) | ✅ | 0/20 marqueurs connus — cohérent avec r=0.210 |
| MC Dropout incertitude (Fig 12) | ✅ | 5.5 % haute incertitude — modèle trop confiant |
| Domaine d'applicabilité (Fig 13) | ✅ | 80 % UNRELIABLE en LDO — quantifié |
| Biomarqueurs sur checkpoint random | ⏳ | Prochaine étape prioritaire |
| Rapport consolidé fiabilité | ⏳ | Après relance sur random |

---

## 6. Prochaines étapes prioritaires

1. **Relancer biomarqueurs (Figs 09–11) sur checkpoint random (r = 0.811)** — seul
   modèle suffisamment convergé pour des attributions biologiquement interprétables.
   Attendu : EGFR, KRAS, BRAF dans le top-20 codants si le modèle a appris les
   voies de signalisation.

2. **Rapport de fiabilité consolidé** (`scripts/reliability_report.py`) — pour 3
   paires représentatives : fiable / prudence / hors domaine.

3. **Compléter la grille de comparaison** : finaliser le run LCO pour avoir les 3
   splits comparables (random, LDO, LCO) avec early stopping rigoureux.

4. **Relancer ablation LDO** depuis `src/` (les 4 configs avaient échoué lors de
   la réorganisation) : early stopping patience=3, dropout 0.3, GNN freeze, 40k
   triplets.

---

## 7. Limitations honnêtes

- **Biomarqueurs calculés sur un modèle peu convergé (LDO r = 0.210)** — les
  attributions Gradient × Input ne reflètent pas nécessairement la biologie réelle ;
  elles mesurent ce que *ce modèle* a appris, qui peut être du bruit.
- **Importance ≠ causalité** — un poids élevé indique une corrélation statistique,
  pas un mécanisme.
- **MC Dropout sous-estime l'incertitude** — dropout à 10 % insuffisant pour une
  estimation bayésienne robuste. Un taux de 30–50 % ou une approche Deep Ensemble
  serait plus fiable.
- **65/266 drogues sans SMILES** — exclues de toutes les analyses moléculaires.
- **Ablation LDO incomplète** — 4/5 configs échouées lors de la réorganisation `src/`.
