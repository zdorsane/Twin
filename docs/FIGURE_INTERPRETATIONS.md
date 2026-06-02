# Interprétation des figures — Projet Twin
## Guide expert : figures 01–13 organisées par phase de travail

**Projet :** Twin — Prédicteur multimodal de réponse aux drogues (CCLE)  
**Auteur :** Zied Dorsane | **Superviseur :** M. Marouane  
**Dernière mise à jour :** 2 juin 2026  
**Audience :** expert en apprentissage automatique appliqué à la biologie computationnelle

---

## Contexte général

Le modèle **Bi-Int** prédit le log-IC50 (µM) d'une drogue sur une lignée cellulaire
cancéreuse à partir de deux entrées : la structure chimique de la drogue (GNN pré-entraîné
sur ChEMBL) et le profil omique de la lignée (VAE quaternionique sur 978 GEx + 426 CNA +
735 mutations). Les deux représentations sont fusionnées par 4 blocs d'attention croisée
bipartite (Bi-Int). **9 255 070 paramètres entraînables.**

**Dataset :** CCLE Broad 2019 — 647 lignées × 201 drogues avec SMILES → 103 477
triplets valides, sous-échantillonnés à 20 000 pour les contraintes mémoire.

**Trois splits d'évaluation :**
- **Random :** train et val partagent les mêmes drogues et lignées → mesure l'interpolation.
- **Leave-Drug-Out (LDO) :** drogues de val jamais vues à l'entraînement → mesure la généralisation structurelle (**métrique honnête**).
- **Leave-Cell-Out (LCO) :** lignées de val jamais vues → mesure la généralisation omique.

---

## Phase 1 — Entraînement, prédiction et génération moléculaire
### Figures 01 · 02 · 03 · 04 · 05 · nb_01–08
### *24 mai 2026 — commits `4f15675`, `fca2b05`*

Cette phase établit les performances de base du modèle prédicteur et des deux générateurs
moléculaires. Elle produit les courbes d'entraînement, les résultats de comparaison avec les
baselines classiques, et les premières visualisations des molécules générées.

---

### Figure 01 — Structures 2D des candidats GraphGA
**`figures/phase1_training_generation/01_molecular_structures.png`** | Généré par : `notebooks/evaluation.ipynb`

**Ce que montre la figure :** représentations RDKit 2D des 10 meilleures molécules
générées par l'algorithme génétique GraphGA, annotées avec leur QED et score composite.

**Interprétation :** Les scaffolds couvrent des motifs benzylaminopipérazine,
tricarbamates, acétamide-biphényle et lactames tricycliques — aucun n'est un analogue
proche d'une drogue CCLE connue (Tanimoto max 0.261). Ces structures sont chimiquement
plausibles et compatibles avec une synthèse de chimie médicinale courante. Le candidat
#1 (QED = 0.872, MW = 303 Da, scaffold benzylaminopipérazine-cyclopropyl) est le plus
attractif pour une synthèse préliminaire : complexité modérée, profil électronique
favorable pour des interactions avec des sites de liaison polaires (kinases, HDAC).

---

### Figure 02 — Courbes d'entraînement QSAR (split random)
**`figures/phase1_training_generation/02_training_curves.png`** | Généré par : `notebooks/evaluation.ipynb`

**Ce que montre la figure :** évolution de la RMSE train/validation et du coefficient
de Pearson r par epoch sur le split random (4 epochs).

| Epoch | Train RMSE | Val RMSE | Pearson r |
|-------|-----------|----------|-----------|
| 1 | 0.959 | 0.854 | 0.506 |
| 2 | 0.767 | 0.899 | 0.631 |
| 3 | 0.669 | 0.594 | 0.791 |
| 4 | 0.606 | **0.588** | **0.811** |

**Interprétation :** Convergence rapide et régulière. Le gradient norm diminue de
moitié entre l'epoch 1 et 4 (26.2 → 9.4), signe d'une optimisation stable. Le
r = 0.811 en epoch 4 témoigne d'une forte capacité d'interpolation sur des drogues
vues à l'entraînement.

**Mise en garde critique :** Ce r = 0.811 est **optimiste par construction** — les
mêmes drogues apparaissent dans train et validation. En split LDO, le même modèle
descend à r = 0.316 dès l'epoch 2 avant de sur-apprendre. La courbe illustre
donc la mémorisation, pas la généralisation.

---

### Figure 03 — Reward BRICS-DQN sur 5 000 épisodes
**`figures/phase1_training_generation/03_dqn_reward.png`** | Généré par : `notebooks/evaluation.ipynb`

**Ce que montre la figure :** évolution de la récompense de l'agent DQN, avec les
moyennes par blocs de 100 épisodes et le taux de validité chimique.

**Interprétation :** L'agent apprend progressivement à assembler des fragments BRICS
(sous-structures rétrosynthétiquement accessibles) pour maximiser un score composite
(QED + IC50 prédit + pénalités de valence). La progression du reward confirme un
apprentissage effectif. La dispersion élevée reflète la stochasticité de l'assemblage
par fragments. Le taux de validité chimique (~60 %) indique qu'une molécule sur deux
est invalide (valence incorrecte, aromaticité mal formée) — l'introduction d'une
pénalité de valence dans la fonction de récompense est la prochaine amélioration.

---

### Figure 04 — Distribution QED et propriétés Lipinski (GraphGA top-10)
**`figures/phase1_training_generation/04_qed_lipinski.png`** | Généré par : `notebooks/evaluation.ipynb`

**Ce que montre la figure :** QED, poids moléculaire (MW) et logP des 10 candidats
GraphGA, comparés aux limites de la règle de Lipinski.

**Résultats :** QED moyen 0.833 (0.710–0.926), MW moyen 304 Da, logP moyen 1.87.
Tous les candidats respectent la règle de Ro5.

**Interprétation :** Un QED > 0.7 est généralement considéré "drug-like" (Bickerton
et al., *Nature Chemistry* 2012). Ces propriétés sont comparables à celles de
petites molécules en phase clinique précoce. Le faible logP moyen (1.87) est
favorable pour la solubilité aqueuse et la perméabilité membranaire, réduisant
le risque d'échec ADMET précoce.

---

### Figure 05 — Dashboard synthétique
**`figures/phase1_training_generation/05_dashboard.png`** | Généré par : `notebooks/evaluation.ipynb`

**Ce que montre la figure :** tableau de bord résumant en un panneau unique les
métriques clés — courbes d'entraînement, comparaison baselines, reward DQN,
distribution QED, pré-entraînement ChEMBL.

**Interprétation :** Ce dashboard met en évidence les deux tensions centrales du
projet : (1) le modèle Bi-Int interpole bien (r = 0.811 random) mais généralise
mal (r = 0.316 LDO) ; (2) les molécules générées ont de bonnes propriétés
drug-like mais leurs IC50 prédites sont hors distribution. Les deux informations
doivent être lues ensemble pour former un jugement équilibré.

---

### Figures notebook (nb\_01–08)
**`figures/phase1_training_generation/nb_01_ccle_summary.png`** à **`figures/phase1_training_generation/nb_08_dashboard.png`**
Généré par : `notebooks/evaluation.ipynb`

Versions haute résolution et détaillées des analyses principales :
résumé CCLE (couverture SMILES, dimensions omiques, distribution splits), courbes
de pré-entraînement ChEMBL (val RMSE = 0.2187 à epoch 9), analyse QSAR par split,
comparaison des 5 modèles × 3 splits, reward DQN avec taux de validité par bloc.
Ces figures servent à la documentation technique et à la reproduction des résultats
sans relancer l'entraînement (valeurs de fallback dans le notebook).

---

## Phase 2 — Validation chimique rigoureuse et ablation LDO
### Figures 06 · 07 · 08
### *25 mai & 31 mai 2026 — commits `94a2513`, `0a4a37e`, `3b1d054`*

Cette phase répond aux demandes de l'encadrant : valider rigoureusement les
candidats générés avec des métriques chimiques standardisées (PAINS, Brenk,
SA score, diversité interne) et analyser les leviers d'amélioration LDO.

---

### Figure 06 — Distribution Tanimoto des candidats vs drogues CCLE
**`figures/phase2_validation_ablation/06_tanimoto_distribution.png`** | Généré par : `scripts/tanimoto_analysis.py`

**Ce que montre la figure :** pour chaque candidat GraphGA (top-10), le Tanimoto
maximum calculé avec les 184 drogues CCLE ayant un SMILES valide (Morgan FP, rayon 2,
2048 bits). Histogramme de distribution.

**Résultats :** Tanimoto max médian ~0.22. Aucun candidat au-dessus de 0.30.
Drogue CCLE la plus proche : UNC1215 (Tanimoto = 0.261 vs candidat #1).

**Interprétation :** Un Tanimoto < 0.30 définit des composés "structurellement
nouveaux" selon les conventions médicinales computationnelles (Willett, 2011). Cette
originalité est à double tranchant : elle maximise le potentiel de brevet mais place
les candidats dans une zone où le modèle Bi-Int extrapole (voir Figure 13). Les IC50
prédites pour ces candidats sont des indicateurs ordinaux (classement relatif), non
des valeurs absolues fiables.

---

### Figure 07 — Ablation LDO : configurations Bi-Int vs baselines
**`figures/phase2_validation_ablation/07_ldo_ablation.png`** | Généré par : `scripts/ldo_ablation.py`

**Ce que montre la figure :** comparaison Pearson r LDO entre le Bi-Int en
configuration baseline et les modèles de référence. Note : les 4 configurations
améliorées (early stopping, régularisation, GNN freeze, 40k triplets) ont échoué
lors de l'exécution suite à la réorganisation `src/`.

**Résultats disponibles :**

| Configuration | LDO r | LDO RMSE |
|--------------|-------|---------|
| Bi-Int baseline (epoch 1) | 0.535 | 0.843 |
| Bi-Int (epoch 2, best run) | **0.316** | 0.983 |
| XGBoost (cible) | 0.367 | 0.938 |

**Interprétation :** Le r = 0.535 à l'epoch 1 représente la performance avant
overfitting — le modèle sur-apprend dès l'epoch 2. XGBoost (r = 0.367) reste
supérieur au meilleur checkpoint LDO Bi-Int (r = 0.316), confirmant que la
complexité du modèle profond n'est pas encore justifiée par le volume de données
(16k triplets d'entraînement). Ce constat oriente les efforts vers la régularisation,
l'augmentation de données et un early stopping rigoureux plutôt que vers une
complexification de l'architecture.

---

### Figure 08 — Heatmap de diversité interne (60 candidats)
**`figures/phase2_validation_ablation/08_internal_diversity.png`** | Généré par : `scripts/molecular_validation.py`

**Ce que montre la figure :** matrice symétrique 60×60 de similarité Tanimoto entre
tous les candidats générés (10 GraphGA + 50 BRICS-DQN top-reward). Couleurs chaudes
= similaires, froides = distincts.

**Résultats :** Tanimoto moyen intra-bibliothèque = **0.10** → diversité = **0.90**.
Aucun cluster visible dans la heatmap.

**Interprétation :** Une diversité de 0.90 sur 60 composés est exceptionnelle — une
chimiothèque combinatoire classique autour d'un seul scaffold donne typiquement
0.4–0.6. L'absence de clusters confirme que GraphGA et BRICS-DQN explorent des
régions complémentaires de l'espace chimique. Cette diversité est un argument fort
pour présenter un sous-ensemble de 10–15 composés à un chimiste médicinal : la
bibliothèque maximise la couverture pharmacologique et réduit la redondance en cas
de screening expérimental.

**Contexte MedChem global :** 38/60 candidats passent tous les filtres (PAINS, Brenk,
Lipinski, Veber). Les alertes Brenk les plus fréquentes sont les amines aromatiques
(risque métabolique CYP450) et les chaînes aliphatiques longues (solubilité). Ces
alertes sont des heuristiques, non des verdicts — certains médicaments approuvés
contiennent des groupes Brenk.

---

## Phase 3 — Interprétabilité et fiabilité du modèle
### Figures 09 · 10 · 11 · 12 · 13
### *1er juin 2026 — commit `d38ca9e`*

Cette phase produit les premières analyses d'interprétabilité (biomarqueurs
génomiques par attribution Gradient × Input) et les alertes de fiabilité (domaine
d'applicabilité Tanimoto, incertitude MC Dropout). Toutes calculées sur le checkpoint
LDO (r = 0.210, epoch 1).

> ⚠️ **Limite importante :** les attributions sont calculées sur un modèle à
> r = 0.210. Un modèle insuffisamment convergé peut apprendre des corrélations
> artéfactuelles. Ces résultats seront recalculés sur le checkpoint random
> (r = 0.811) pour comparaison et validation biologique.

---

### Figure 09 — Importance des transcrits non-codants (ncRNA)
**`figures/phase3_interpretability_reliability/09_ncrna_importance.png`** | Généré par : `scripts/ncrna_biomarker_analysis.py`

**Méthode :** Gradient × Input sur le vecteur GEx (978 dimensions). Pour chaque
paire de validation (drogue *d*, lignée *c*), le gradient de la prédiction IC50 par
rapport aux 978 features d'expression est multiplié par la valeur d'entrée.
L'importance d'un gène est la moyenne des valeurs absolues sur **150 paires**
échantillonnées dans le split de validation LDO. Restreint aux 76 transcrits
non-codants identifiés parmi les 978 features (RP11-, AC-, AL-, LINC-, MIR-, RNU-,
SNORD-, SNHG-, ainsi que les ncRNA nommés : H19, GAS5, RMRP, SNHG5, VIM-AS1…).

**Résultats :**

| Rang/76 | Gène | Rang/978 | Importance | Catégorie biologique |
|---------|------|----------|------------|----------------------|
| 1 | RNU1-28P | 34 | 0.001350 | Pseudogène snRNA U1 |
| 2 | RMRP | 57 | 0.001221 | ARN composant RNase MRP mitochondriale |
| 8 | VIM-AS1 | 166 | 0.001017 | Antisens de Vimentine (EMT) |
| 54 | SNHG5 | 715 | 0.000431 | Hôte snoRNA — oncogène (prolifération) |
| **61** | **H19** | **769** | **0.000323** | **lncRNA oncogène — résistance chimio** |
| **70** | **GAS5** | **916** | **0.000090** | **lncRNA suppresseur de tumeur** |

**Interprétation :** Le modèle pondère prioritairement des loci génomiques de
nomenclature RP11- et des pseudogènes snRNA — transcrits à haute variance
d'expression entre lignées mais sans rôle fonctionnel documenté. H19 et GAS5,
les deux lncRNA à rôle oncologique établi, se trouvent en bas du classement
(rangs 61 et 70/76). Ce résultat est cohérent avec une convergence partielle
(r = 0.210) : le modèle exploite des corrélations statistiques de variance élevée
plutôt que des mécanismes biologiques de pharmacorésistance.

**VIM-AS1** (rang 8) mérite une attention particulière : transcrit antisens de
VIM (Vimentine), protéine surexprimée dans la transition épithélio-mésenchymateuse
(EMT). Son importance relative pourrait refléter un signal biologique réel lié à
la plasticité phénotypique des lignées cellulaires.

**Prochaine étape :** recalculer sur le checkpoint random (r = 0.811). Si H19 et
GAS5 remontent dans le classement avec le modèle convergé, cela constituera une
validation biologique de l'apprentissage.

---

### Figure 10 — Heatmap importance ncRNA × drogues
**`figures/phase3_interpretability_reliability/10_ncrna_vs_drugs.png`** | Généré par : `scripts/ncrna_biomarker_analysis.py`

**Ce que montre la figure :** matrice (top-15 drogues × top-10 ncRNA), valeurs
normalisées par colonne. Couleur chaude = importance relative élevée de ce ncRNA
pour cette drogue.

**Interprétation :** Cette figure permet de détecter une **spécificité drug-ncRNA** :
certaines drogues activent-elles préférentiellement certains ncRNA dans la prédiction
du modèle ? Une heatmap uniforme (lignes similaires entre elles) indique que le
modèle ne discrimine pas les profils ncRNA selon la drogue — attendu pour un modèle
peu convergé. Une heatmap différenciée (blocs de couleur) suggérerait une
spécificité drug-ncRNA apprise. Cette analyse sera plus informative après recalcul
sur le checkpoint random, où le modèle a appris des représentations plus
discriminantes.

---

### Figure 11 — Importance des gènes codants (top-20)
**`figures/phase3_interpretability_reliability/11_coding_biomarkers.png`** | Généré par : `scripts/coding_biomarker_analysis.py`

**Méthode :** même Gradient × Input que Figure 09, restreint aux 902 gènes codants
(978 − 76 ncRNA). Biomarqueurs oncologiques canoniques surlignés en rouge
(EGFR, KRAS, BRAF, TP53, PIK3CA, ERBB2, ABL1, MYC, BCL2, PTEN…).

**Résultats :**

| Rang | Gène | Importance | Pertinence oncologique |
|------|------|------------|------------------------|
| 1 | CCND3 | 0.001902 | Cycline D3 — contrôle G1/S, prolifération |
| 4 | CTGF | 0.001826 | CCN2 — EMT, résistance thérapies ciblées |
| 5 | THY1 | 0.001783 | CD90 — marqueur cellules souches tumorales |
| 8 | SQSTM1 | 0.001660 | p62 — autophagie, résistance au stress oxydatif |
| 13 | AREG | 0.001551 | Amphiréguline — ligand EGFR, résistance anti-EGFR |
| 24 | CTNNB1 | 0.001411 | β-caténine — voie Wnt, résistance aux taxanes |

**Biomarqueurs oncologiques canoniques dans le top-20 : 0/20.**

**Interprétation :** L'absence d'EGFR, KRAS, BRAF, TP53 dans le top-20 est le
signal le plus clair d'une convergence insuffisante. Un modèle ayant appris la
pharmacorésistance devrait pondérer EGFR pour les inhibiteurs d'EGFR (Afatinib,
Erlotinib), BRAF pour les inhibiteurs BRAF (PLX-4720), ABL1 pour les inhibiteurs
BCR-ABL. Leur absence confirme que le checkpoint LDO n'a pas appris les voies de
signalisation pharmacologiquement pertinentes.

Cependant, plusieurs gènes ont une pertinence indirecte non négligeable : AREG est
un mécanisme de résistance au cetuximab/erlotinib documenté ; CTNNB1 est associé
à la résistance aux taxanes dans le cancer du sein ; SQSTM1 est impliqué dans
l'autophagie comme mécanisme de résistance à la chimiothérapie. Ces cooccurrences
suggèrent que le modèle capte partiellement des signaux biologiques, sans avoir
encore convergé vers les mécanismes clés.

---

### Figure 12 — Incertitude MC Dropout
**`figures/phase3_interpretability_reliability/12_uncertainty_distribution.png`** | Généré par : `scripts/uncertainty_mc_dropout.py`

**Méthode :** 200 paires de validation, N = 30 passages forward avec `training=True`
(dropout = 10 % actif). Calcul de σ (écart-type), IC 95 % (percentiles 2.5–97.5)
par paire. Seuil d'alerte = médiane(σ) + écart-type(σ) sur les 200 paires.

**Résultats :**

| Métrique | Valeur |
|----------|--------|
| σ médian | ~0.13 |
| Seuil d'alerte | **0.1975** |
| HIGH_UNCERTAINTY (σ > seuil) | **11 / 200 (5.5 %)** |
| IC 95 % moyen (amplitude) | ~0.68 unités z-score |

**Interprétation :** Un taux de 5.5 % de paires à haute incertitude est anormalement
bas pour un modèle à r = 0.210. Ce phénomène — un réseau confiant malgré de mauvaises
prédictions — est bien documenté (Gal & Ghahramani, 2016) : les réseaux de neurones
sont calibrés pour produire des sorties stables, y compris lors d'extrapolations hors
distribution. Le dropout à 10 % est insuffisant pour une estimation bayésienne robuste
(recommandation : 20–50 %).

**Exemple concret :** la paire (NG-25, BT483\_BREAST) présente ic50\_true = 6.07
vs ic50\_mean = 1.94 — erreur de ~4 unités z-score — avec σ = 0.205 (alerte
HIGH\_UNCERTAINTY). Ce cas illustre que l'alerte MC Dropout est utile mais tardive
sur des prédictions profondément incorrectes.

**Implication pratique :** MC Dropout seul est insuffisant comme alerte de fiabilité.
Il doit être **combiné obligatoirement avec l'alerte Tanimoto** (Figure 13) : un
modèle peut être confiant (σ faible) et simultanément opérer hors de son domaine
d'applicabilité. Les deux métriques sont complémentaires et non substituables.

---

### Figure 13 — Domaine d'applicabilité (Tanimoto)
**`figures/phase3_interpretability_reliability/13_applicability_domain.png`** | Généré par : `scripts/applicability_domain.py`

**Méthode :** pour chaque drogue de validation LDO (40 drogues), calcul du Tanimoto
maximum avec les 161 drogues d'entraînement (Morgan FP, rayon 2, 2048 bits).
Seuils : ≥ 0.6 = fiable, [0.4–0.6] = prudence, < 0.4 = hors domaine.

**Résultats sur les 40 drogues de validation :**

| Niveau | Seuil | N | % | Signification |
|--------|-------|---|---|---------------|
| 🟢 RELIABLE | ≥ 0.6 | 4 | **10 %** | Analogue structurel d'une drogue d'entraînement |
| 🟡 CAUTION | 0.4–0.6 | 4 | **10 %** | Proximité partielle — résultat à vérifier |
| 🔴 UNRELIABLE | < 0.4 | **32** | **80 %** | Drogue structurellement nouvelle — hors domaine |

Les 4 drogues RELIABLE (Tanimoto = 1.0) sont des variants stéréochimiques ou
formes prodrogue d'une même molécule présente dans le train (GSK269962A, JQ1,
Refametinib). Ce sont les prédictions les plus fiables sur ce split.

**Interprétation :** Ce résultat est attendu par définition du split LDO. Son
intérêt n'est pas de diagnostiquer un problème mais de le **quantifier précisément**.
Le modèle doit prédire l'IC50 de molécules dont le scaffold est inédit dans 80 %
des cas — ce qui explique mécaniquement le r = 0.316.

**Utilisation en production — immédiatement opérationnelle :**
```
si max_tanimoto(drogue, drogues_train) < 0.4  →  🔴 HORS DOMAINE — prédiction non fiable
si 0.4 ≤ max_tanimoto < 0.6                   →  🟡 PRUDENCE — interpréter avec réserve
si max_tanimoto ≥ 0.6                          →  🟢 FIABLE — prédiction dans le domaine appris
```

Cette alerte est **robuste indépendamment de la performance du modèle** — elle
repose sur la distance chimique, pas sur la précision des prédictions. C'est la
mesure de fiabilité la plus immédiatement exploitable du projet.

---

## Synthèse par phase

| Phase | Figures | Période | Message central |
|-------|---------|---------|----------------|
| **1 — Entraînement & génération** | 01–05, nb\_01–08 | 24 mai 2026 | r=0.811 (random) vs r=0.316 (LDO) ; 38/60 candidats MedChem-clean ; diversité 0.90 |
| **2 — Validation chimique & ablation** | 06, 07, 08 | 25–31 mai 2026 | Candidats structurellement nouveaux (Tanimoto < 0.30) ; XGBoost > Bi-Int en LDO ; bibliothèque très diverse |
| **3 — Interprétabilité & fiabilité** | 09–13 | 1er juin 2026 | H19/GAS5 non prioritaires sur LDO checkpoint ; MC Dropout trop confiant ; 80 % des nouvelles drogues hors domaine |

**Lecture transversale :** les trois phases forment un argument progressif.
La Phase 1 établit que le modèle fonctionne mais généralise mal. La Phase 2 valide
que les molécules générées sont chimiquement solides, malgré cette limitation.
La Phase 3 quantifie précisément *pourquoi* la généralisation est difficile (80 %
des nouvelles drogues sont hors domaine d'applicabilité) et *comment* signaler
cette limite à l'utilisateur final (alertes Tanimoto + MC Dropout).

---

## Références méthodologiques

| Méthode | Référence |
|---------|-----------|
| Gradient × Input | Simonyan et al. (2014) ; Kindermans et al. (2016) |
| MC Dropout | Gal & Ghahramani, *ICML* 2016 |
| Domaine d'applicabilité Tanimoto | Tropsha & Golbraikh, *J. Chem. Inf. Model.* 2007 |
| QED | Bickerton et al., *Nature Chemistry* 2012 |
| SA Score | Ertl & Schuffenhauer, *J. Cheminformatics* 2009 |
| PAINS | Baell & Holloway, *J. Med. Chem.* 2010 |
| Morgan FP / Tanimoto | Rogers & Hahn, *J. Chem. Inf. Model.* 2010 |
| CCLE dataset | Barretina et al., *Nature* 2012 |
