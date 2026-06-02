# Interprétation des figures — Projet Twin
## Guide expert : ce que montrent les figures 01–13

**Projet :** Twin — Prédicteur multimodal de réponse aux drogues (CCLE)  
**Auteur :** Zied Dorsane | **Superviseur :** M. Marouane  
**Dernière mise à jour :** 2 juin 2026  
**Audience :** expert en apprentissage automatique appliqué à la biologie computationnelle

---

## Vue d'ensemble du projet

Le modèle **Bi-Int** (Bipartite Interaction Transformer) prédit la concentration
inhibitrice à 50 % (IC50, log µM) d'une drogue sur une lignée cellulaire cancéreuse,
à partir de deux sources d'information :

1. **La structure chimique de la drogue** : encodée par un GNN (Graph Neural Network)
   appliqué au graphe moléculaire BRICS, pré-entraîné sur ChEMBL.
2. **Le profil omique de la lignée** : 978 gènes d'expression (GEx top-variance),
   426 gènes de variation du nombre de copies (CNA), 735 gènes mutés — fusionnés via
   un VAE quaternionique en un vecteur latent de dimension 128.

Les deux représentations sont fusionnées par des blocs d'**attention croisée
bipartite** (Bi-Int), puis un MLP prédit le log-IC50 final.

**Dataset :** CCLE Broad 2019 — 647 lignées × 201 drogues avec SMILES → 103 477
triplets valides (drogue, lignée, IC50), sous-échantillonnés à 20 000 pour les
contraintes mémoire.

**Splits d'évaluation :**
- **Random :** drogues et lignées partagées entre train et validation — mesure
  l'interpolation.
- **Leave-Drug-Out (LDO) :** drogues de validation jamais vues à l'entraînement —
  mesure la généralisation structurelle, c'est la métrique honnête.
- **Leave-Cell-Out (LCO) :** lignées de validation jamais vues à l'entraînement —
  mesure la généralisation omique.

---

## Bloc 1 — Performances prédictives (Figures 02–05)

### Figure 02 — Courbes d'entraînement QSAR (split random)
**`figures/02_training_curves.png`**

**Ce que montre la figure :** évolution de la RMSE (train et validation) et du
coefficient de Pearson r par epoch sur le split random, 4 epochs.

| Epoch | Train RMSE | Val RMSE | Pearson r |
|-------|-----------|----------|-----------|
| 1 | 0.959 | 0.854 | 0.506 |
| 2 | 0.767 | 0.899 | 0.631 |
| 3 | 0.669 | 0.594 | 0.791 |
| 4 | 0.606 | **0.588** | **0.811** |

**Interprétation :** La convergence est rapide et régulière sur le split random.
Le gradient norm diminue de moitié entre l'epoch 1 et 4 (26.2 → 9.4), signe d'une
optimisation stable. L'atteinte de r = 0.811 en 4 epochs témoigne d'une capacité
d'interpolation très élevée du modèle sur des drogues vues.

**Mise en garde critique :** Ce r = 0.811 est **optimiste par construction** — les
mêmes drogues apparaissent dans le train et la validation. En split LDO (drogues de
validation structurellement nouvelles), le même modèle descend à r = 0.316 dès
l'epoch 2 avant de sur-apprendre. Le split random mesure donc la mémorisation,
pas la généralisation.

---

### Figure 03 — Reward BRICS-DQN sur 5 000 épisodes
**`figures/03_dqn_reward.png`**

**Ce que montre la figure :** évolution de la récompense de l'agent DQN au fil des
épisodes, avec les movennes par blocs de 100 épisodes.

**Interprétation :** L'agent apprend progressivement à assembler des fragments BRICS
(sous-structures chimiques rétrosynthétiquement accessibles) pour maximiser un score
composite (QED + IC50 prédit + pénalités de valence). La progression du reward indique
un apprentissage effectif. La dispersion élevée reflète la nature stochastique de
l'assemblage par fragments — la même séquence de fragments peut produire des molécules
de qualité variable selon l'ordre d'assemblage.

**Limite :** Une récompense élevée ne garantit pas la validité chimique. Le taux de
validité (~60%) indique qu'environ 2 000 sur 5 000 molécules générées sont invalides
(valence incorrecte, cycles aromatiques mal formés). L'introduction de contraintes de
valence dans la fonction de récompense est la prochaine amélioration prévue.

---

### Figure 04 — Distribution QED et propriétés Lipinski (GraphGA top-10)
**`figures/04_qed_lipinski.png`**

**Ce que montre la figure :** pour les 10 candidats GraphGA, distribution du QED
(Quantitative Estimate of Drug-likeness), du poids moléculaire (MW) et du logP.

**Résultats :**
- QED moyen : **0.833** (0.710 – 0.926) — excellent pour une molécule générée
- MW moyen : **304 Da** (269 – 337) — bien dans la fenêtre Lipinski (< 500 Da)
- LogP moyen : **1.87** (0.65 – 3.33) — favorable pour la perméabilité membranaire

**Interprétation :** Les 10 candidats ont des propriétés physicochimiques comparables
à des médicaments de petite molécule en phase clinique précoce. Un QED > 0.7 est
généralement considéré comme "drug-like" dans la littérature médicinale computationnelle
(Bickerton et al., *Nature Chemistry* 2012). Aucun candidat ne viole les règles
de Lipinski, ce qui est un prérequis pour l'administration orale.

---

### Figure 05 — Dashboard synthétique
**`figures/05_dashboard.png`**

**Ce que montre la figure :** tableau de bord résumant en un seul panneau les
métriques clés : courbes d'entraînement, comparaison baselines, reward DQN, QED,
pré-entraînement ChEMBL.

**Interprétation :** Ce dashboard est conçu pour une lecture rapide par un encadrant
ou un lecteur externe. Il met en évidence les deux tensions centrales du projet :
(1) le modèle Bi-Int performe bien en random (r = 0.811) mais mal en LDO (r = 0.316) ;
(2) les molécules générées ont de bonnes propriétés drug-like (QED > 0.7) mais leurs
IC50 prédites sont hors distribution. Les deux informations doivent être lues ensemble
pour former un jugement juste sur l'état du projet.

---

## Bloc 2 — Génération et validation moléculaire (Figures 01, 06, 07, 08)

### Figure 01 — Structures 2D des candidats GraphGA
**`figures/01_molecular_structures.png`**

**Ce que montre la figure :** représentations RDKit 2D des 10 meilleures molécules
générées par l'algorithme génétique GraphGA, annotées avec leur QED et leur score
composite.

**Interprétation :** L'inspection visuelle des structures confirme la diversité
des scaffolds : motifs benzylaminopipérazine, tricarbamates, acétamides biphényle.
Ces structures ne sont pas des analogues proches de drogues anticancéreuses connues,
ce qui les positionne comme des hits potentiels pour des cibles non encore couvertes
dans l'espace CCLE. La présence systématique de groupes amino et carbonyle est
cohérente avec un ciblage de sites de liaison polaires (kinases, HDAC, récepteurs
nucléaires).

**Note :** Le candidat #1 (QED = 0.872, scaffold benzylaminopipérazine-cyclopropyl)
est le plus attractif pour une synthèse préliminaire en raison de sa complexité
modérée et de son profil électronique favorable.

---

### Figure 06 — Distribution Tanimoto des candidats vs drogues CCLE
**`figures/06_tanimoto_distribution.png`**

**Ce que montre la figure :** pour chaque candidat généré (GraphGA top-10), le
Tanimoto maximum calculé avec les 184 drogues CCLE ayant un SMILES valide (Morgan
FP, rayon = 2, 2048 bits). Histogramme de distribution.

**Résultats :**
- Tanimoto max médian : ~0.22
- Aucun candidat au-dessus de 0.30
- Candidat le plus proche d'une drogue CCLE : #1 (Tanimoto = 0.261 vs UNC1215)

**Interprétation :** Une similarité Tanimoto < 0.30 définit des composés
"structurellement nouveaux" selon les conventions de la chimie médicinale
computationnelle (Willett, 2011). Cette originalité structurelle présente deux
faces : elle maximise le potentiel de brevet et la couverture de l'espace chimique,
mais elle place les candidats dans une zone où le modèle Bi-Int extrapole
(voir Figure 13 — domaine d'applicabilité). Les IC50 prédites pour ces candidats
doivent donc être traitées comme des indicateurs ordinaux (classement relatif), non
comme des valeurs absolues fiables.

---

### Figure 07 — Ablation LDO : Bi-Int vs baselines
**`figures/07_ldo_ablation.png`**

**Ce que montre la figure :** comparaison des performances LDO (Pearson r) du
modèle Bi-Int face aux baselines classiques (XGBoost, RF, Ridge, MLP) et entre
différentes configurations du modèle (baseline, +early stopping, +régularisation,
+GNN freeze, +40k triplets).

**Résultats clés (valeurs réelles disponibles) :**

| Modèle | LDO r | LDO RMSE |
|--------|-------|----------|
| Bi-Int baseline (epoch 1) | 0.535 | 0.843 |
| XGBoost (cible) | 0.367 | 0.938 |
| Bi-Int epoch 2 (best run) | **0.316** | 0.983 |

**Interprétation :** La configuration ablation n'a retourné de résultats que pour la
configuration baseline (les autres configurations ont échoué lors de la réorganisation
`src/`). Le r = 0.535 à l'epoch 1 représente un point de départ avant overfitting. La
comparaison avec XGBoost (r = 0.367) confirme que le deep learning n'apporte pas encore
de gain en généralisation LDO à cette échelle de données. Ce résultat, loin d'être
décourageant, pointe vers les leviers d'amélioration prioritaires : early stopping
rigoureux, régularisation, augmentation du dataset.

---

### Figure 08 — Heatmap de diversité interne (60 candidats)
**`figures/08_internal_diversity.png`**

**Ce que montre la figure :** matrice symétrique de similarité Tanimoto entre les
60 candidats (10 GraphGA + 50 BRICS-DQN top-reward). Plus la couleur est chaude,
plus deux molécules sont similaires.

**Résultats :**
- Tanimoto moyen intra-bibliothèque : **0.10**
- Diversité (1 − Tanimoto moyen) : **0.90**

**Interprétation :** Une diversité de 0.90 sur 60 composés est exceptionnelle — à
titre de comparaison, une chimiothèque combinatoire autour d'un seul scaffold donne
typiquement 0.4–0.6. L'absence de clusters visibles dans la heatmap confirme que
GraphGA et BRICS-DQN explorent des régions complémentaires de l'espace chimique.
Ce résultat est un argument fort pour présenter un sous-ensemble de 10–15 composés
à un chimiste médicinal : la bibliothèque est représentative d'une large diversité
structurelle et maximise les chances de trouver un hit actif sur une cible donnée.

---

## Bloc 3 — Interprétabilité et fiabilité (Figures 09–13)

> Ces analyses ont été réalisées sur le checkpoint LDO (r = 0.210, epoch 1).
> **La faible performance de ce modèle implique que les attributions reflètent ce
> que ce modèle particulier a appris, pas nécessairement la biologie réelle.**
> Les biomarqueurs seront recalculés sur le checkpoint random (r = 0.811) pour
> comparaison et validation.

### Figure 09 — Importance des transcrits non-codants (ncRNA)
**`figures/09_ncrna_importance.png`**

**Méthode :** Gradient × Input sur le vecteur GEx (978 dimensions). Pour chaque
paire de validation (drogue *d*, lignée *c*), on calcule le gradient de la prédiction
IC50 par rapport aux 978 features d'expression génique, puis on multiplie par la
valeur d'entrée (Gradient × Input = attribution locale). L'importance d'un gène est
la moyenne des valeurs absolues sur 150 paires échantillonnées aléatoirement dans
le split de validation LDO. On restreint ensuite aux 76 transcrits non-codants
identifiés parmi les 978 features (patterns RP11-, AC-, AL-, LINC-, MIR-, RNU-,
SNORD-, SNHG-, ainsi que les ncRNA nommés connus : H19, GAS5, RMRP, SNHG5).

**Résultats :**

| Rang/76 | Gène | Rang/978 | Importance | Catégorie biologique |
|---------|------|----------|------------|----------------------|
| 1 | RNU1-28P | 34 | 0.001350 | Pseudogène snRNA U1 |
| 2 | RMRP | 57 | 0.001221 | ARN composant de la RNase MRP mitochondriale |
| 3–20 | RP11-*, AC*, Z*, VIM-AS1 | 69–271 | — | Loci génomiques non-annotés |
| 54 | SNHG5 | 715 | 0.000431 | Hôte snoRNA — oncogène (prolifération) |
| **61** | **H19** | **769** | **0.000323** | **lncRNA oncogène — résistance chimio** |
| **70** | **GAS5** | **916** | **0.000090** | **lncRNA suppresseur de tumeur** |

**Interprétation :** Le modèle pondère prioritairement des loci génomiques de
nomenclature RP11- (loci chromosomiques non-annotés dans les bases fonctionnelles
actuelles) et des pseudogènes snRNA (RNU1-28P). Ces transcrits ont une variance
d'expression élevée entre lignées cellulaires (raison de leur sélection dans le
top-978), mais leur rôle fonctionnel est non documenté. H19 et GAS5 — les deux
lncRNA à rôle oncologique bien établi — se trouvent en bas du classement (rangs
61 et 70 sur 76).

Ce résultat est **biologiquement cohérent avec la convergence partielle du modèle
(r = 0.210 en LDO)** : un modèle sous-optimal peut apprendre à exploiter des
corrélations artéfactuelles (variance cellulaire non liée à la pharmacorésistance)
plutôt que les mécanismes biologiques sous-jacents. La prochaine étape est de
recalculer ces attributions sur le checkpoint random (r = 0.811), où le modèle a
appris une représentation plus riche. Si H19 et GAS5 remontent dans le classement
avec le meilleur modèle, cela constituera une validation biologique de l'apprentissage.

**VIM-AS1** (rang 8) mérite une attention particulière : c'est le transcrit antisens
de VIM (Vimentine), une protéine du cytosquelette surexprimée dans la transition
épithélio-mésenchymateuse (EMT). Son importance relative dans ce classement pourrait
refléter un signal biologique réel lié à la plasticité phénotypique des lignées.

---

### Figure 10 — Heatmap importance ncRNA × drogues
**`figures/10_ncrna_vs_drugs.png`**

**Ce que montre la figure :** matrice (top-15 drogues) × (top-10 ncRNA) où chaque
cellule représente l'importance normalisée par colonne. Une couleur chaude indique
que ce ncRNA a une importance relative élevée pour cette drogue particulière.

**Interprétation :** Cette figure permet de détecter une éventuelle **spécificité
drug-ncRNA** : certaines drogues activent-elles préférentiellement certains ncRNA ?
Une heatmap uniforme (toutes les lignes similaires) indiquerait que le modèle ne
discrimine pas le profil ncRNA selon la drogue — cohérent avec un modèle peu
convergé. Une heatmap différenciée (blocs de couleur distinctifs) suggérerait une
spécificité apprise. Cette analyse sera plus informative après recalcul sur le
checkpoint random.

---

### Figure 11 — Importance des gènes codants (top-20)
**`figures/11_coding_biomarkers.png`**

**Méthode :** même approche Gradient × Input que Figure 09, restreinte aux 902 gènes
codants (978 − 76 ncRNA). Les biomarqueurs oncologiques connus sont surlignés en
rouge.

**Résultats :**

| Rang | Gène | Importance | Biomarqueur connu | Note biologique |
|------|------|------------|-------------------|-----------------|
| 1 | CCND3 | 0.001902 | Non | Cycline D3 — contrôle G1/S, prolifération |
| 4 | CTGF | 0.001826 | Non | CCN2 — fibrose, EMT, résistance aux thérapies ciblées |
| 5 | THY1 | 0.001783 | Non | CD90 — marqueur cellules souches tumorales |
| 8 | SQSTM1 | 0.001660 | Non | p62 — autophagie, résistance au stress oxydatif |
| 13 | AREG | 0.001551 | Non | Amphiréguline — ligand EGFR, résistance anti-EGFR |
| 18 | APP | 0.001498 | Non | Précurseur APP — signalisation Notch |
| 24 | CTNNB1 | 0.001411 | Non | β-caténine — voie Wnt, résistance aux taxanes |

**Biomarqueurs oncologiques canoniques (EGFR, KRAS, TP53, BRAF…) dans top-20 : 0/20.**

**Interprétation :** L'absence de marqueurs oncologiques classiques dans le top-20
est le résultat le plus instructif de cette analyse. Un modèle ayant appris la
pharmacorésistance devrait pondérer EGFR pour les inhibiteurs d'EGFR (Afatinib,
Erlotinib), BRAF pour les inhibiteurs BRAF (PLX-4720), ABL1 pour les inhibiteurs
BCR-ABL. Leur absence confirme que le checkpoint LDO (r = 0.210) n'a pas appris
les voies de signalisation pharmacologiquement pertinentes.

Cela dit, plusieurs gènes du top-20 ont une pertinence biologique indirecte :
CTGF est impliqué dans la résistance aux thérapies ciblées via le microenvironnement
tumoral ; AREG est un mécanisme de résistance aux cetuximab/erlotinib bien documenté ;
CTNNB1 est associé à la résistance aux taxanes dans le cancer du sein. Ces
cooccurrences ne sont pas aléatoires et suggèrent que le modèle capte partiellement
des signaux biologiques, sans avoir encore convergé vers les mécanismes clés.

---

### Figure 12 — Incertitude MC Dropout
**`figures/12_uncertainty_distribution.png`**

**Méthode :** pour 200 paires de validation (drogue, lignée) tirées aléatoirement,
N = 30 passages forward sont effectués avec `training=True` (dropout actif, taux
10 %). On calcule l'écart-type σ des 30 prédictions et l'intervalle de confiance
à 95 % (percentiles 2.5 et 97.5). Le seuil d'alerte est fixé à
médiane(σ) + écart-type(σ) des 200 paires.

**Résultats :**

| Métrique | Valeur |
|----------|--------|
| σ médian | ~0.13 |
| Seuil d'alerte | **0.1975** |
| Paires HIGH_UNCERTAINTY (σ > seuil) | **11 / 200 (5.5 %)** |
| Paires OK | 189 / 200 (94.5 %) |
| IC 95% moyen (amplitude) | ~0.68 unités z-score |

**Interprétation :** Un taux de 5.5 % de paires à haute incertitude est
**anormalement bas** pour un modèle à r = 0.210. Ce résultat illustre un phénomène
bien documenté en apprentissage profond : les réseaux de neurones sont
**calibrated pour être confiants**, même quand leurs prédictions sont fausses.
Le modèle produit des intervalles étroits (~0.68 σ) y compris sur des prédictions
incorrectes.

Deux mécanismes expliquent cette sous-estimation de l'incertitude : (1) le taux de
dropout de 10 % est insuffisant pour une estimation bayésienne robuste (Gal & Ghahramani,
2016 recommandent 20–50 % pour MC Dropout) ; (2) le modèle a convergé vers un
minimum local stable où les sous-réseaux activés par dropout donnent des prédictions
similaires. 

**Implication pratique :** l'incertitude MC Dropout seule ne suffit pas comme alerte
de fiabilité pour ce modèle. Elle doit être **combinée obligatoirement** avec
l'alerte Tanimoto (Figure 13) : un modèle peut être très confiant (σ faible) et
simultanément opérer hors de son domaine d'applicabilité (Tanimoto < 0.4). Les deux
métriques sont complémentaires et non substituables.

**Exemple emblématique dans les données :** la paire (NG-25, BT483_BREAST) a
ic50_true = 6.07 (log µM) mais ic50_mean = 1.94 — une erreur de ~4 unités z-score —
avec une alerte HIGH_UNCERTAINTY (σ = 0.205). Ce cas illustre que l'alerte MC Dropout
est utile mais tarde à se déclencher sur des prédictions profondément incorrectes.

---

### Figure 13 — Domaine d'applicabilité (Tanimoto)
**`figures/13_applicability_domain.png`**

**Méthode :** pour chaque drogue du dataset CCLE (train et validation), on calcule
la similarité Tanimoto maximale avec toutes les drogues d'entraînement (Morgan FP,
rayon 2, 2048 bits). Seuils : ≥ 0.6 = fiable, [0.4 – 0.6] = prudence, < 0.4 =
hors domaine. En split LDO, 161 drogues constituent le set d'entraînement et 40 le
set de validation.

**Résultats sur les 40 drogues de validation :**

| Niveau | Seuil Tanimoto | N drogues | % | Interprétation |
|--------|---------------|-----------|---|----------------|
| 🟢 RELIABLE | ≥ 0.6 | **4** | **10 %** | Analogue structurel d'une drogue d'entraînement |
| 🟡 CAUTION | 0.4 – 0.6 | **4** | **10 %** | Proximité partielle — prédiction à vérifier |
| 🔴 UNRELIABLE | < 0.4 | **32** | **80 %** | Drogue structurellement nouvelle — hors domaine |

**Drogues fiables (Tanimoto = 1.0) :** GSK269962A, JQ1, Refametinib — tous des
paires de variants (isoformes stéréochimiques ou formes prodrogue) d'une même molécule
présente dans le train.

**Interprétation :** Ce résultat est **attendu par définition du split LDO** : les
drogues de validation sont sélectionnées pour ne pas se chevaucher avec celles
d'entraînement. Le Tanimoto < 0.4 pour 80 % d'entre elles confirme qu'elles occupent
des régions distinctes de l'espace chimique. Ce résultat ne diagnostique pas un
problème — il le **quantifie**.

**Utilisation pratique immédiate :** cette alerte Tanimoto est utilisable en production
dès maintenant, indépendamment de toute amélioration du modèle prédictif. Pour toute
nouvelle drogue soumise au modèle :

```
si max_tanimoto(drogue_nouvelle, drogues_train) < 0.4 → afficher 🔴 HORS DOMAINE
si 0.4 ≤ max_tanimoto < 0.6                            → afficher 🟡 PRUDENCE
si max_tanimoto ≥ 0.6                                  → afficher 🟢 FIABLE
```

C'est la mesure de fiabilité la plus robuste du projet, car elle ne dépend pas de
la performance du modèle mais de la distance chimique entre la drogue cible et
l'espace d'entraînement.

---

## Synthèse globale — Ce que les 13 figures racontent ensemble

| Groupe | Message central | Figures |
|--------|----------------|---------|
| Performances | Le modèle interpole bien (r=0.811) mais généralise mal (r=0.316 LDO) | 02, 05, 07 |
| Génération | 38/60 candidats passent tous les filtres MedChem ; diversité = 0.90 | 01, 04, 08 |
| Originalité | Tous les candidats sont structurellement nouveaux vs CCLE (Tanimoto < 0.30) | 06 |
| Interprétabilité | Le checkpoint LDO (r=0.210) ne capture pas les biomarqueurs canoniques | 09, 10, 11 |
| Incertitude | Le modèle est trop confiant ; MC Dropout insuffisant seul | 12 |
| Fiabilité | 80 % des nouvelles drogues sont hors domaine — alerte Tanimoto opérationnelle | 13 |

**Conclusion pour l'expert :** Le projet Twin démontre la faisabilité technique d'un
prédicteur multimodal IC50 + générateur de novo sur données CCLE. Les performances en
généralisation LDO (r = 0.316) restent en dessous de XGBoost (r = 0.367), ce qui
indique que la complexité du modèle profond n'est pas encore justifiée à cette échelle
de données. Les analyses d'interprétabilité sur le checkpoint LDO confirment cette
limite : les biomarqueurs identifiés ne sont pas biologiquement cohérents. Les
prochaines étapes prioritaires sont : (1) relancer les biomarqueurs sur le checkpoint
random pour validation biologique, (2) améliorer la convergence LDO par régularisation
et augmentation de données, (3) déployer l'alerte Tanimoto en production.

---

## Références méthodologiques

- **Gradient × Input :** Simonyan et al. (2014), Kindermans et al. (2016)
- **MC Dropout comme approximation bayésienne :** Gal & Ghahramani (2016)
- **Domaine d'applicabilité Tanimoto :** Tropsha & Golbraikh (2007)
- **QED :** Bickerton et al., *Nature Chemistry* (2012)
- **SA Score :** Ertl & Schuffenhauer, *J. Cheminformatics* (2009)
- **PAINS :** Baell & Holloway, *J. Med. Chem.* (2010)
- **Tanimoto / Morgan FP :** Rogers & Hahn, *J. Chem. Inf. Model.* (2010)
- **Dataset CCLE :** Barretina et al., *Nature* (2012)
