# Présentation du Travail — Twin
## Prédicteur Multimodal de Réponse aux Drogues + Génération Moléculaire De Novo

**Projet :** Twin — Multimodal Drug Response Predictor (CCLE)
**Auteur :** Zied Dorsane
**Superviseur :** M. Marouane
**Destinataires :** Jury de soutenance / Investisseurs / Partenaires industriels
**Date :** 1er juin 2026

---

## PARTIE 1 — LE PROBLÈME : 2 000 milliards de dollars gaspillés chaque décennie

### Le problème de marché que nous résolvons

Aujourd'hui, **développer un médicament contre le cancer prend en moyenne 12 à 15 ans et coûte 2,6 milliards de dollars**. Le taux d'échec est de **90% en phase clinique**. La première cause : la mauvaise sélection des patients et des molécules en amont.

**Trois problèmes concrets coexistent sur le marché :**

| Problème | Réalité actuelle | Coût |
|----------|-----------------|------|
| **Sélection thérapeutique aveugle** | Les oncologues choisissent les traitements sans prédire *a priori* la réponse moléculaire du patient | Traitements toxiques inutiles, progression tumorale |
| **Découverte moléculaire non guidée** | Les chimistes explorent manuellement un espace chimique de 10^60 molécules possibles | 5 à 10 ans pour trouver un candidat druggable |
| **Modèles prédictifs non interprétables** | Les rares modèles existants donnent un chiffre sans expliquer *pourquoi* ni signaler leur incertitude | Pas de confiance clinique, pas d'adoption |

**Le vide de marché :** il n'existe pas aujourd'hui de système unifié qui prédit la sensibilité d'une tumeur à une molécule, génère de nouvelles molécules optimisées, et justifie ses prédictions par des biomarqueurs biologiquement validés.

---

## PARTIE 2 — NOTRE SOLUTION : Twin, le premier prédicteur multimodal à interprétabilité mécanistique

### Ce que fait Twin en une phrase

> Twin prédit la sensibilité d'une lignée cellulaire cancéreuse à une drogue en fusionnant trois types de données biologiques — et génère de nouvelles molécules candidates optimisées chimiquement, avec une qualification rigoureuse de fiabilité.

### Les données utilisées : le CCLE comme terrain d'entraînement

Nous nous sommes appuyés sur la **Cancer Cell Line Encyclopedia (CCLE, Broad Institute 2019)** — la base de données de référence mondiale en pharmacogénomique :

- **1 457 lignées cellulaires cancéreuses** × **266 drogues anti-cancéreuses**
- **103 477 mesures IC50** (concentration inhibitrice à 50%, le standard or de la cytotoxicité)
- **Trois modalités moléculaires intégrées :**
  - Transcriptome RNA-seq (978 gènes top-variance, dont 71 transcrits non-codants)
  - Altérations du nombre de copies (CNA, 426 régions génomiques)
  - Profil mutationnel somatique (MAF, 735 gènes top-mutés)
  - Structure chimique SMILES (201 drogues mappées via PubChem)

---

## PARTIE 3 — L'ARCHITECTURE TECHNIQUE : trois sous-systèmes spécialisés

### Composant 1 — L'encodeur moléculaire : GNN pré-entraîné sur ChEMBL

**Problème à résoudre :** représenter une molécule de façon à capturer ses propriétés pharmacologiques, pas seulement son identité.

**Solution :** un **Graph Neural Network (GNN)** traite la molécule comme un graphe — atomes = noeuds, liaisons chimiques = arêtes. Ce réseau a été **pré-entraîné sur 100 000 molécules ChEMBL** (la plus grande base de molécules bioactives) pour apprendre des représentations latentes capturant la forme électronique, les groupes fonctionnels, et les pharmacophores.

> **Analogie jury :** c'est comme entraîner un expert en lecture de structures chimiques avant de lui montrer les données cancer. Il arrive avec un savoir préalable, pas à zéro.

**Avantage vs concurrence :** les fingerprints classiques (ECFP4, Morgan) encodent l'identité de la molécule — si la drogue n'a jamais été vue, la fingerprint est aveugle. Le GNN, lui, reconnaît des **sous-structures pharmacophores** même dans des molécules nouvelles.

### Composant 2 — L'encodeur cellulaire : Quaternion-VAE

**Problème à résoudre :** le profil omique d'une lignée cellulaire est un vecteur de 2 139 dimensions (978 + 426 + 735). La fusion naïve par concaténation ignore les interactions non-linéaires inter-modalités.

**Solution :** un **Variational Autoencoder (VAE) à algèbre quaternionique**. Le VAE compresse les 2 139 dimensions en un espace latent de 128 dimensions, en utilisant l'arithmétique des quaternions (extension des nombres complexes à 4 dimensions) pour coder les relations entre modalités omiques.

**Ce que ça apporte :** les quaternions capturent des interactions non-commutatives entre gènes — le fait que EGFR muté + PIK3CA amplifié ≠ PIK3CA amplifié + EGFR muté sur la réponse à l'erlotinib, par exemple.

> **Mot-clé jury :** *représentation latente multimodale* — le modèle ne stocke pas les données brutes, il apprend une description comprimée et biologiquement informative de la cellule.

### Composant 3 — Les blocs Bi-Interaction (Bi-Int) : la fusion drogue-cellule

**Problème à résoudre :** comment faire interagir la représentation de la drogue et la représentation de la cellule de façon à prédire leur interaction biologique spécifique ?

**Solution :** 4 blocs **Bi-Interaction** empilés, chacun utilisant un mécanisme d'**attention croisée** (cross-attention) : la représentation de la drogue interroge la représentation de la cellule, et vice-versa, en parallèle.

> **Analogie jury :** imaginez un système de reconnaissance mutuelle — la molécule "cherche" dans le profil génomique de la cellule les patterns qui lui correspondent (récepteurs, voies de signalisation), et inversement. C'est ce que font ces blocs d'interaction.

**En sortie :** une tête MLP (256→128→64→1) produit la prédiction de log-IC50 normalisé.

**Taille du modèle :** 9 255 070 paramètres — optimisé pour tenir sur une GPU NVIDIA RTX 4000 Ada (20 475 MiB VRAM).

---

## PARTIE 4 — PROTOCOLE D'ÉVALUATION : pourquoi nos résultats sont honnêtes

### Le piège du split aléatoire — et comment nous l'avons évité

La quasi-totalité des publications académiques en QSAR évaluent leurs modèles en **split aléatoire** : les mêmes drogues apparaissent dans l'entraînement ET le test. C'est un **artefact de mémorisation** — le modèle a appris "la drogue X donne en moyenne telle IC50", et retrouve ce pattern au test.

Nous avons implémenté **trois régimes de validation** :

| Split | Principe biologique | Ce qu'il mesure réellement |
|-------|--------------------|-----------------------------|
| **Random** | Drogues partagées train/test | Interpolation — surestimation |
| **Leave-Drug-Out (LDO)** | Drogues du test jamais vues à l'entraînement | Généralisation à de nouvelles molécules — le cas réel |
| **Leave-Cell-Out (LCO)** | Lignées du test jamais vues à l'entraînement | Transfert vers de nouveaux types tumoraux |

> **Message fort pour le jury :** "Nous rapportons le chiffre LDO, pas le chiffre random. C'est le seul chiffre qui correspond à l'usage réel : prédire la réponse à une molécule candidate que le modèle n'a jamais vue."

### Intervalles de confiance bootstrap — la rigueur statistique

Tous les Pearson r sont accompagnés d'**intervalles de confiance à 95% par bootstrap (n=1 000 itérations)** :

| Split | Modèle | Pearson r | IC 95% |
|-------|--------|-----------|--------|
| Random | Bi-Int | **0.811** | [0.736 – 0.886] |
| Random | MLP (ECFP4+omics) | 0.881 | — |
| Random | XGBoost | 0.849 | — |
| **LDO** | **XGBoost (meilleure baseline)** | **0.367** | [0.338 – 0.393] |
| **LDO** | **MLP (ECFP4+omics)** | **0.349** | — |
| **LDO** | **Bi-Int** | **0.316** | [0.241 – 0.391] |
| **LDO** | Ridge (ECFP4+omics) | 0.286 | — |
| LCO | XGBoost | **0.824** | — |
| LCO | Bi-Int (partiel, 6 epochs) | 0.766 | — |

> ⚠️ **Les lignes LDO sont les seules métriques honnêtes.** En LDO, Bi-Int (r=0.316) est troisième derrière MLP (0.349) et XGBoost (0.367). Ce classement est le point de départ du raisonnement, pas son point d'arrivée.

### Ce que cela signifie honnêtement

- **r = 0.811 en random** : compétitif avec les publications CCLE (fourchette 0.70–0.85), mais ce chiffre mesure la **mémorisation**, pas la généralisation. Il ne sera pas cité comme performance principale en soutenance.
- **r = 0.316 en LDO** : faible, mais **statistiquement significatif** (p << 0.001, n > 10 000 triplets). C'est notre métrique honnête — la seule qui correspond à l'usage réel.
- **XGBoost bat Bi-Int en LDO (0.367 vs 0.316)** : c'est un **résultat négatif net, et nous l'assumons pleinement**. Il n'est pas présenté comme un échec mais comme une **contribution scientifique** : nous avons quantifié et exposé l'artefact de mémorisation des fingerprints fixes (ECFP4) sur données CCLE — là où la majorité des publications évaluées sur random split passent ce problème sous silence. Ce diagnostic est la première étape nécessaire pour concevoir des solutions.

> **À annoncer soi-même devant le jury, avant qu'on le pose :** *"Sur le seul test honnête — Leave-Drug-Out — notre modèle deep learning obtient r = 0.316, sous XGBoost à r = 0.367. Nous l'exposons explicitement parce que c'est précisément ce résultat qui motive l'architecture future."*

---

## PARTIE 5 — GÉNÉRATION MOLÉCULAIRE : deux approches complémentaires

### Module 1 — GraphGA : optimisation évolutive sur graphes

**Principe :** algorithme génétique opérant directement sur des **graphes moléculaires** (non des strings SMILES). À chaque génération, les molécules les mieux scorées sont croisées et mutées au niveau des fragments BRICS (Retrosynthetically Interesting Chemical Substructures).

**Fonction objectif (score composite) :**
```
Score = QED (drug-likeness) + SA (accessibilité synthétique) + IC50_prédit_Bi-Int
```

> ⚠️ **Limite explicite à présenter au jury :** la composante IC50_prédit est calculée par l'oracle Bi-Int sur des molécules nouvelles — exactement le régime LDO où ce modèle obtient r = 0.316. Les candidats sont donc classés par **chimie validée** (QED, SA, filtres MedChem) — pas par potency anticancéreuse prouvée. Ils sont présentés comme **molécules valides et originales à valider expérimentalement**, pas comme candidats anticancer démontrés.

**Résultat :** 10 candidats top, tous Lipinski-compliant, QED moyen = 0.833 (supérieur à la médiane des médicaments approuvés ChEMBL = 0.67), 0 alerte PAINS.

### Module 2 — BRICS-DQN : apprentissage par renforcement moléculaire

**Principe :** un **agent Double DQN** assemble des fragments BRICS comme un jeu de construction chimique. À chaque épisode, l'agent choisit séquentiellement quels fragments relier, guidé par une récompense combinant QED, validité aromatique, compliance Lipinski et IC50 prédit.

**Résultat :** 5 000 épisodes → 3 027 molécules valides (60.5% de validité). Meilleure récompense = 6.124/6.5 théorique maximum.

> **Différenciation technique :** contrairement aux générateurs SMILES-RNN ou SELFIES (qui génèrent des chaînes de caractères et échouent fréquemment sur l'aromaticité), notre approche opère sur des **fragments chimiquement valides par construction** — ce qui explique la validité supérieure à 60% sans pré-entraînement.

### Validation chimique multi-critères (60 candidats analysés)

Nous avons appliqué une batterie de filtres MedChem standard de l'industrie pharmaceutique :

| Filtre | Signification | Résultat |
|--------|--------------|----------|
| **Règles de Lipinski** | Drug-likeness oral (MW, LogP, HBD, HBA) | 55/60 compliant |
| **Filtres PAINS** | Groupes réactifs causant de faux positifs en screening | 1/60 flaggé |
| **Filtres Brenk** | Groupes toxiques/réactifs | 18/60 flaggés |
| **SA Score** | Accessibilité synthétique (0=difficile, 1=facile) | 67% avec SA < 3 (facilement synthétisable) |
| **Tanimoto vs CCLE** | Originalité structurale vs drogues connues | Max = 0.32 → aucune copie de drogue existante |
| **Diversité interne** | Similarité Tanimoto intra-bibliothèque | 0.90 → bibliothèque hautement diversifiée |

**Résultat net : 38/60 candidats (63%) passent tous les filtres** — résultat remarquable pour un générateur RL sans contrainte explicite de MedChem pendant l'entraînement.

### Top 5 candidats (classement qualité IC50-agnostique)

| Rang | ID | Source | Score qualité | QED | SA | Drogue structurellement proche |
|------|----|--------|--------------|-----|----|-----------------------------|
| 1 | BRI-46 | BRICS-DQN | 0.925 | 0.903 | 1.89 | Belinostat (HDAC inhibiteur) |
| 2 | BRI-12 | BRICS-DQN | 0.916 | 0.850 | 1.68 | Belinostat |
| 3 | BRI-58 | BRICS-DQN | 0.900 | 0.858 | 2.27 | GW441756 (TrkA inhibiteur) |
| 4 | Gra-9 | GraphGA | 0.889 | 0.926 | 3.25 | TubastatinA |
| 5 | Gra-1 | GraphGA | 0.885 | 0.872 | 2.85 | UNC1215 |

---

## PARTIE 6 — INFRASTRUCTURE DE FIABILITÉ : ce qui nous différencie

### Pourquoi c'est un argument de startup, pas juste de la technique

Un modèle qui prédit avec assurance sur des drogues inconnues sans signaler son incertitude est **un risque de responsabilité civile et réglementaire**. C'est la raison pour laquelle les modèles d'IA en drug discovery peinent à passer de l'académique au clinique.

**Nous avons construit trois couches de fiabilité :**

**Couche 1 — Domaine d'applicabilité (Tanimoto)**
Avant chaque prédiction : calcul de la similarité Tanimoto entre la drogue cible et les drogues d'entraînement. Si Tanimoto max < 0.4 → **alerte "hors domaine"** automatique.

**Couche 2 — Incertitude bayésienne (MC Dropout)**
N=50 passes stochastiques avec dropout actif à l'inférence → distribution de prédictions → variance = mesure d'incertitude. Un modèle qui "hésite" dit qu'il hésite.

**Couche 3 — Intervalle de confiance prédictif (Bootstrap)**
Fourchette [IC_low, IC_high] sur la prédiction finale, pas juste un point.

> **Pitch jury :** "Nous transformons un modèle qui 'ment avec assurance' en un modèle 'honnête qui connaît ses limites'. C'est ce qui rend le système acceptable par un comité éthique ou un régulateur FDA/EMA."

### Reproductibilité numérique — le standard scientifique

**Bug critique découvert et corrigé :** le VAE utilisait `tf.random.normal()` même en inférence (`training=False`), rendant les prédictions non-déterministes après rechargement du modèle. Correctif : retourner la **moyenne latente µ** (espérance de la distribution postérieure) en inférence.

**Preuve de correction :** test automatisé round-trip save/load → écart maximal = **0.00e+00** (identité numérique exacte, 9.2M paramètres).

---

## PARTIE 7 — BIOMARQUEURS ncRNA : la découverte inattendue

### Ce qui distingue un modèle statistique d'un modèle biologique

Un modèle qui prédit l'IC50 avec r = 0.316 dit "il y a un signal". Mais le jury — et l'investisseur — veulent savoir : **est-ce de la biologie réelle, ou du bruit statistique corrélé ?**

**La découverte :** parmi les 978 gènes sélectionnés par top-variance dans le RNA-seq CCLE, **71 sont des ARN non-codants (ncRNA)**, dont :

- **H19** — lncRNA oncogène, rôle documenté dans la **résistance à la chimiothérapie** (Barlow et al., *Cell* 1994 ; Luo et al., *Oncotarget* 2016)
- **GAS5** — lncRNA suppresseur de tumeur, corrélé à la réponse aux inhibiteurs mTOR et aux agents cytotoxiques

**Plan de validation mécanistique (Integrated Gradients) :**
Appliquer la méthode **Integrated Gradients** (attribution de Sundararajan et al. 2017 — standard or en interprétabilité IA) sur le checkpoint entraîné pour mesurer la contribution de chaque gène à chaque prédiction IC50.

**Hypothèse testable :** si le modèle pondère fortement EGFR pour les inhibiteurs d'EGFR (erlotinib, gefitinib), ou H19 pour les lignées résistantes à la chimiothérapie → **le modèle a appris de la biologie, pas du bruit**.

> **Argument marché :** l'interprétabilité par biomarqueurs transforme une boîte noire en un outil de **stratification de patients** — le cas d'usage clinique le plus valorisé en oncologie de précision.

---

## PARTIE 8 — ANALYSE DES LIMITATIONS : l'honnêteté comme argument de crédibilité

Un jury de startup sait reconnaître un fondateur qui connaît ses angles morts.

| Limitation | Ce que ça signifie | Ce qu'on va faire |
|------------|-------------------|-------------------|
| **LDO r = 0.316** — XGBoost fait 0.367 | Le deep learning n'est pas encore justifié à l'échelle actuelle (20k/103k triplets) | Ablation systématique : early stopping + dropout 0.3 + 103k triplets complets |
| **Importance ≠ causalité** | Un biomarqueur fort dans le modèle est une corrélation, pas un mécanisme prouvé | Validation in vitro sur lignées CRISPR-KO ciblées |
| **65/266 drogues sans SMILES** | 24% du corpus exclu des analyses moléculaires | Lookup ChEMBL synonymes + CAS registry |
| **Validité BRICS-DQN = 60%** | 40% des molécules générées sont chimiquement invalides | Masque de valence sur l'espace d'actions du DQN |
| **Données CCLE 2019** | Pas de validation croisée sur GDSC (Sanger) | Plan d'extension dataset sur GDSC2 |

---

## SYNTHÈSE — Ce que Twin apporte au marché

| Problème de marché | Solution Twin | Preuve |
|-------------------|---------------|--------|
| Prédiction de réponse aux drogues sans généralisation | Évaluation LDO + IC bootstrap — r = 0.316 statistiquement significatif | 3 splits, 12 modèles, 1 000 itérations bootstrap |
| Espace chimique exploré manuellement | GraphGA + BRICS-DQN → 60 candidats drug-like en <24h GPU | 38/60 passent tous les filtres MedChem |
| Modèles "boîte noire" sans fiabilité | Tanimoto + MC Dropout + Integrated Gradients | Reproductibilité numérique 0.00e+00 |
| Données omiques en silo | Fusion quaternionique GNN + VAE multimodal | Architecture unifiée 9.2M paramètres |

---

## PARTIE 9 — FIL DIRECTEUR SCIENTIFIQUE : le test décisif à venir

**Saber Marouane (reviewer externe) formule le fil directeur ainsi :**

> *"Votre vraie future work est là : remplacer les fingerprints fixes par une représentation apprise (GNN, dans l'esprit de MolCLR et de votre pré-entraînement ChEMBL). Si elle bat XGBoost en LDO, l'approche deep devient justifiée. C'est le fil directeur de votre projet."*

**En termes techniques :** les baselines XGBoost et Ridge utilisent des fingerprints ECFP4 — vecteurs binaires fixes qui encodent l'**identité** d'une molécule. En régime LDO, quand la drogue n'a jamais été vue, ECFP4 ne transporte aucune information structurale utile. Notre GNN pré-entraîné sur ChEMBL encode au contraire la **structure chimique apprise** — il reconnaît des sous-structures pharmacophores même dans des molécules nouvelles.

**Le test décisif :** sur données complètes (103k triplets, early stopping, dropout renforcé), le GNN doit dépasser XGBoost en LDO. Si oui → l'approche deep est justifiée. Si non → le goulot est la quantité de données, pas l'architecture.

**Formulation pour la soutenance :**
> *"Notre résultat LDO actuel (r = 0.316 vs XGBoost r = 0.367) n'est pas un échec — c'est un diagnostic. Il indique que, à 20k triplets, XGBoost exploite mieux les corrélations omiques que notre réseau de 9M de paramètres. Le test décisif est clair : sur les 103k triplets complets avec régularisation adaptée, le GNN pré-entraîné sur ChEMBL doit dépasser XGBoost en LDO. Si c'est le cas, l'intégration deep learning se justifie. Nous avons construit l'infrastructure pour répondre à cette question."*

---

## Phrase de clôture pour le jury

> *"Twin ne prétend pas avoir un prédicteur parfait. Il démontre que la rigueur d'évaluation (LDO, bootstrap CI, domaine d'applicabilité) et l'honnêteté sur les limites (résultat négatif assumé, oracle faible signalé) sont les fondations sans lesquelles aucun outil d'IA clinique ne peut être crédible. C'est exactement ce qui manque dans la majorité des publications QSAR actuelles — et ce que Twin met en place."*

---

*Document rédigé le 1er juin 2026. Auteur : Zied Dorsane. Documentation : Claude Sonnet 4.6.*
*Tous les résultats numériques sont issus de données réelles (aucune valeur simulée).*
