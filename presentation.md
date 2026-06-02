---
marp: true
theme: default
class: invert
paginate: true
backgroundColor: #0f1117
color: #e0e0e0
style: |
  section {
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    padding: 50px 60px;
  }
  h1 {
    color: #4dd0e1;
    font-size: 2.2em;
    border-bottom: 2px solid #4dd0e1;
    padding-bottom: 10px;
  }
  h2 {
    color: #80cbc4;
    font-size: 1.4em;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
  }
  th {
    background-color: #1e3a4a;
    color: #4dd0e1;
    padding: 8px 12px;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #2a2a3a;
  }
  blockquote {
    border-left: 4px solid #4dd0e1;
    padding-left: 16px;
    color: #b0bec5;
    font-style: italic;
  }
  ul li {
    margin-bottom: 6px;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2em;
  }
---

<!-- _class: invert -->
<!-- _paginate: false -->

# Twin

## Jumeau Numérique pour la Découverte de Médicaments Anti-Cancer

**Prédiction d'IC50 par apprentissage profond multimodal**
**+ Génération de candidats moléculaires de novo**

---
Projet de recherche — Juin 2026

---

# Problème & Motivation

## Pourquoi est-ce difficile ?

- Les cancers varient génétiquement d'un patient à l'autre
- Tester chaque médicament sur chaque tumeur en laboratoire est **coûteux et lent**
- Les modèles classiques ignorent les **données multi-omiques** des tumeurs

## Notre objectif

> Étant donné la **structure moléculaire d'un médicament** et le **profil génomique d'une tumeur** (expression génique, mutations, CNV), prédire son **efficacité (IC50)**

---

# Architecture : Bi-Int Transformer

## 3 encodeurs en parallèle

| Encodeur | Entrée | Méthode |
|----------|--------|---------|
| **Drug** | Structure SMILES | Fragments BRICS → GNN (ChEMBL 1.9M molécules) |
| **Omics** | 978 gènes + 426 CNV + 735 mutations | Quaternion VAE |
| **Fusion** | Sorties des deux encodeurs | Blocs Bi-Int (attention croisée + mises à jour triangulaires) |

**Sortie →** MLP → log IC50 (μM)

---

# Résultats Honnêtes

## Attention au split de validation !

| Protocole | Pearson r | Interprétation |
|-----------|-----------|----------------|
| Random Split | **0.811** ⚠️ | Optimiste — fuite de données |
| Leave-Drug-Out — Deep Model | **0.316** `[0.287–0.344]` | Généralisation réelle |
| Leave-Drug-Out — XGBoost | **0.367** `[0.338–0.393]` | Baseline surpasse le deep learning |

> **Leçon clé :** Sur des données limitées (~16k triplets), XGBoost reste compétitif.
> L'honnêteté scientifique prime sur les métriques flatteuses.

---

# Génération Moléculaire de Novo

## Deux générateurs complémentaires

- **DQN (Deep Q-Network)** : assemble des fragments BRICS par renforcement
- **GraphGA** : algorithme génétique sur graphes moléculaires

## Résultats

| Critère | Valeur |
|---------|--------|
| Candidats générés | 60 |
| Filtres MedChem passés | **38/60 (63%)** |
| Diversité interne | **0.90** (très élevée) |
| Similarité Tanimoto vs training | **< 0.30** — nouveauté structurelle confirmée |

---

# Fiabilité & Quantification d'Incertitude

## Le modèle sait quand il ne sait pas

**Domaine d'applicabilité (Tanimoto fingerprints) :**

- 🔴 **UNRELIABLE** — 80% des nouveaux médicaments
- 🟡 **CAUTION** — 10%
- 🟢 **RELIABLE** — 10%

**MC Dropout (30 passes stochastiques) :**
- 5.5% de prédictions à haute incertitude (σ > 0.198)

**Biomarqueurs identifiés (Gradient×Input) :**
- Gènes codants et ARN non-codants les plus influents sur les prédictions

> Essentiel pour un usage clinique responsable

---

# Conclusions & Perspectives

## Ce que Twin apporte

- ✅ Architecture Bi-Int originale fusionnant chimie + multi-omique
- ✅ Évaluation honnête Leave-Drug-Out avec IC 95% bootstrap
- ✅ 38 candidats MedChem-valides, structurellement nouveaux
- ✅ Cadre de fiabilité avec alertes d'applicabilité + incertitude MC

## Limites & Prochaines étapes

- ⚠️ Deep learning sous-performant vs baselines sur données limitées
- ⚠️ 80% des nouveaux médicaments hors domaine d'applicabilité

**→** Validation expérimentale · Augmentation du dataset · Publication open-source
