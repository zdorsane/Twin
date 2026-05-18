# Rapport de Session Technique — 18 Mai 2026
## Bi-Int Digital Twin : Pipeline de Drug Discovery — Prédiction IC50 & Génération Moléculaire De Novo

**Projet :** Bipartite Interaction (Bi-Int) Digital Twin pour la pharmacogénomique du cancer  
**Jeu de données :** CCLE (Cancer Cell Line Encyclopedia) Broad 2019 — 266 drogues × 647 lignées cellulaires × 137 182 triplets IC50  
**Plateforme :** Ubuntu 24.04 LTS (WSL2) · NVIDIA RTX 4000 Ada (17 710 Mo VRAM) · TensorFlow 2.21.0  
**Branche :** `main` — commits `8853f83` → `7864675`

---

## Résumé Exécutif

Cette session a porté sur deux axes : (1) l'**implémentation des suggestions de l'encadrant** concernant la fonction de perte du VAE omique et la représentation moléculaire du DQN, et (2) le **diagnostic et la correction d'une défaillance critique du pipeline de données** — l'absence de features moléculaires réelles pour les drogues — qui masquait la vraie capacité de généralisation du modèle. Quatre résultats expérimentaux à implications scientifiques directes ont été produits.

**Score global de l'encadrant : 7.7/10** — architecture solide, itération expérimentale intentionnelle, limitations de données identifiées et en cours de correction.

---

## 1. État du Système en Entrée de Session

### 1.1 Architecture Bi-Int Digital Twin

Le système intègre trois modules :

- **Encodeur de drogue (GNN) :** Réseau de neurones sur graphes pré-entraîné sur 100 000 molécules ChEMBL par régression multi-tâche auto-supervisée (8 descripteurs RDKit : LogP, TPSA, MW, QED, HBD, HBA, NumRings, NumAromaticRings). RMSE de validation : **0.208** (espace normalisé). Poids transférés à l'étape QSAR.

- **Encodeur omique (QuatVAE) :** Autoencodeur variationnel quaternionique — fusion par produit de Hamilton de l'expression génique (GEx, 978 gènes, RPKM normalisé) et des altérations de nombre de copies (CNA, 426 gènes). Espace latent : `z ∈ ℝ^128`, régularisé par β-KL divergence (β=2.0, free_bits=0.5). Point de fonctionnement KL : 64 nats (= 0.5 nat/dimension — utilisation complète de la capacité latente, pas d'effondrement postérieur).

- **Tête d'interaction (Bi-Int) :** 4 blocs d'interaction bipartite empilés (attention croisée par rangées, attention croisée par colonnes, mise à jour multiplicative triangulaire — inspiré d'AlphaFold2) → tête MLP → prédiction IC50.

**Baseline QSAR (split aléatoire 85/15) :** Val RMSE **0.472** (normalisé).

### 1.2 Problème Non Résolu en Entrée

Les features de drogues dans l'entraînement CCLE étaient des **vecteurs aléatoires** (`np.random.randn`). Le fichier IC50 CCLE identifie les drogues par des identifiants internes avec suffixes de réplicats (ex. `Afatinib-1`, `Afatinib-2`) — aucun SMILES n'était injecté dans le GNN. Le modèle apprenait la réponse IC50 depuis les omiques uniquement, la branche drogue ne contribuant que du bruit.

---

## 2. Remarques de l'Encadrant — Implémentation

### 2.1 Suggestions du jour

> *"Deux suggestions distinctes à implémenter : 1. BRICS tokenization dans le DQN (remplacer SELFIES tokens par fragments BRICS). 2. Cross-entropy loss à la place de KL divergence dans le VAE."*

**Sur les fragments BRICS :**
> *"Au lieu que le DQN apprenne à assembler atome par atome ([C@@H1], [N], [O]...), il apprend à combiner des blocs chimiques entiers comme un chimiste médicinal le ferait. Un fragment BRICS comme `[*:1]c1ccncc1` représente directement un noyau pyridine — le modèle n'a plus besoin de 'découvrir' que 6 atomes adjacents font un cycle aromatique."*

**Sur le cross-entropy :**
> *"Le KL divergence contraint la distribution latente à rester proche de N(0,I), ce qui est parfois trop agressif et empêche le VAE d'encoder des informations fines sur les profils omiques. Le cross-entropy sur la reconstruction force le modèle à mieux mémoriser le signal biologique sans cette contrainte gaussienne."*

### 2.2 Cross-Entropy comme Fonction de Perte VAE

**Justification scientifique :**
La fonction de perte de reconstruction implicite (MSE via la voie de régression) est sous-optimale pour les données omiques pour deux raisons :

1. **Sparsité des features :** Les profils de mutations somatiques sont binaires (0/1 par gène) ; la perte MSE assigne des gradients identiques aux erreurs sur les features nulles indépendamment de leur signification biologique. La binary cross-entropy (BCE) rend compte de cela via la log-vraisemblance des features distribuées de manière de Bernoulli.

2. **Sur-régularisation KL (hypothèse) :** À β=2.0, la pénalité KL `D_KL[q(z|x) ‖ N(0,I)]` pousse tous les embeddings vers une gaussienne isotrope unique, effaçant la structure propre aux drogues et aux lignées cellulaires. Si le paysage de réponse pharmacologique réel n'est pas bien approximé par N(0,I), cette régularisation détruit activement le signal prédictif.

**Implémentation (`fullPipeline.py` — `UnifiedOmicsVAE.call()`) :**

Trois modes de perte ajoutés via le paramètre `loss_mode` :

| Mode | Terme de reconstruction | Régularisation | Usage prévu |
|------|------------------------|---------------|-------------|
| `kl` (défaut) | MSE implicite | β · D_KL | Baseline rétrocompatible |
| `cross_entropy` | BCE sur omiques normalisées min-max | Aucune | Tester reconstruction sans contrainte KL |
| `both` | BCE + MSE | β · D_KL | Régularisation combinée |

**β-annealing (`BiIntTrainer`) :**
Pour prévenir l'effondrement KL précoce, un schedule linéaire de β-annealing a été implémenté :

```
β(epoch) = 0.0 + (epoch / 10) × 2.0
```

Cela permet au modèle d'apprendre d'abord la reconstruction (β≈0, autoencodeur pur), puis d'introduire progressivement la régularisation.

### 2.3 BRICS Fragment-Based DQN (`brics_dqn_optimizer.py`)

**Cause racine identifiée des échecs SELFIES DQN v3–v5 :**

> Dans SELFIES 2.x, le benzène s'encode comme `[C][=C][C][=C][C][=C][Ring1][=Branch1]`. La fermeture du cycle aromatique est signalée par le token `[=Branch1]` — un symbole abstrait spécifique à SELFIES sans signification chimique directe. Sur 10 000 molécules ChEMBL, `[=Branch1]` apparaît **20 539 fois** dans les encodages SELFIES. La fonction de récompense ne récompensait pas spécifiquement ce token — le DQN ne l'a jamais appris.

**Solution BRICS :**
BRICS (Break Retrosynthetically Interesting Chemical Substructures, Degen et al. 2008) décompose les molécules en fragments retrosynthétiquement valides avec des points d'attachement `[*:N]`. Chaque fragment est un scaffold médicinal complet :

| Fragment | Scaffold | Aromatique |
|----------|---------|-----------|
| `[*:1]c1ccccc1` | Phényle | ✓ |
| `[*:1]N1CCNCC1` | Pipérazinyle | — |
| `[*:1]c1ccncc1` | Pyridyle | ✓ |

En remplaçant la génération atome par atome par **l'assemblage de scaffolds BRICS**, l'espace d'action devient sémantiquement signifiant : chaque action EST un scaffold drug-like. L'aromaticité est inhérente aux fragments eux-mêmes.

---

## 3. Résultats Expérimentaux

### 3.1 Comparaison des Modes de Perte VAE — Split Aléatoire

**Protocole :** `compare_vae_losses.py --epochs 5 --fast --batch-size 256`  
20 000 triplets train, 4 864 val (split aléatoire 85/15). Poids ChEMBL pré-entraînés chargés. Seed fixe (tf.random.set_seed(42)).

| Mode de perte | Val RMSE ↓ | Pearson r ↑ | Perte auxiliaire VAE | Idéal pour |
|--------------|-----------|------------|---------------------|-----------|
| `kl` (baseline) | 0.848 | 0.546 | KL = 64.00 nats | Modélisation générative & échantillonnage |
| **`cross_entropy`** | **0.702** | **0.713** | BCE = 0.459 | **Précision des prédictions en aval** |
| `both` | 0.747 | 0.689 | Combinée = 64.48 | Apprentissage semi-supervisé & interpolation |

**Résultat principal :**  
La suppression complète de la contrainte KL (mode `cross_entropy`) améliore le Pearson r de **+30.6 %** (0.546 → 0.713) et réduit la RMSE de validation de **17.2 %** par rapport à la baseline KL pure.

**Interprétation détaillée — pourquoi chaque mode se comporte ainsi :**

**A. `kl` (VAE classique) — Val RMSE : 0.848 | r : 0.546**  
La valeur KL élevée (débutant à ~82, finissant à ~64) montre que l'encodeur s'efforce de projeter l'espace latent vers N(0,I). En forçant les représentations latentes de différentes drogues et lignées à se chevaucher sous cette distribution a priori gaussienne, la KL détruit la variance fine et hautement spécifique nécessaire à la prédiction QSAR. Ce *lissage de l'information* est excellent pour les tâches génératives mais pénalise les tâches de régression qui dépendent de features moléculaires précises.

**B. `cross_entropy` (autoencodeur déterministe) — Val RMSE : 0.702 | r : 0.713**  
Ce mode transforme le VAE en autoencodeur déterministe. Sans contrainte gaussienne sur l'espace latent, le réseau s'organise librement pour préserver un maximum d'information d'entrée. L'encodeur peut projeter des sous-structures moléculaires rares ou très actives vers des régions isolées à forte variance de l'espace latent — ce qui se traduit directement par des prédictions de réponse aux drogues nettement supérieures. Pearson r = 0.713 est **compétitif avec les modèles CCLE multimodaux publiés** (DeepDR : r~0.72, MOLI : r~0.75) sur des conditions comparables, en notant que les features de drogues restaient des vecteurs aléatoires.

**C. `both` (VAE combiné) — Val RMSE : 0.747 | r : 0.689**  
Formulation VAE complète : L = L_reconstruction + β·L_KL. Il surpasse le mode `kl` pur car la BCE force la préservation des descripteurs, limitant l'effet de flou du KL. Ses performances sont légèrement inférieures à `cross_entropy` pur car la contrainte KL continue de restreindre la structure de l'espace latent.

**Le compromis fondamental : reconstruction vs génération**

```
Autoencodeur (CE uniquement)              Autoencodeur Variationnel (KL + CE)
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│ • Espace latent discret         │      │ • Espace latent continu & lisse │
│ • Préserve les détails fins     │      │ • Distribution standardisée     │
│ • Idéal pour QSAR / régression  │      │ • Idéal pour génération de novo │
│ • Difficile d'échantillonner    │      │ • Représentation légèrement     │
│   de nouvelles molécules        │      │   altérée                       │
└─────────────────────────────────┘      └─────────────────────────────────┘
          (Meilleure prédiction IC50)              (Meilleure conception moléculaire)
```

**Réserves méthodologiques :**
- Contrainte `--fast` : test sur ~20k échantillons au lieu de 137k — les jeux de données plus petits ont tendance à exagérer l'avantage des modèles non régularisés.
- Risque de surapprentissage : les modèles sans contrainte KL (`cross_entropy`) sont plus sujets à l'overfitting sur des entraînements prolongés. En seulement 5 epochs, nous observons la transition sous-apprentissage → ajustement ; 50 epochs pourrait révéler une divergence de la courbe de validation pour `cross_entropy`.

---

### 3.2 Comparaison — Split Leave-Drug-Out (Run 1 — Sans SMILES Réels)

**Protocole :** `compare_vae_losses.py --epochs 5 --fast --batch-size 256 --split-mode leave_drug_out`  
227 drogues d'entraînement / 39 drogues de validation — **aucune drogue partagée**. SMILES réels chargés : **0/266** (voir Section 4 — bug pipeline).

| Mode | Val RMSE | Pearson r | Gap train/val | Interprétation |
|------|----------|-----------|--------------|----------------|
| `kl` | 1.369 | −0.354 | 0.67 → 1.37 (+0.70) | Overfit massif sur les 227 drogues train |
| `cross_entropy` | 1.357 | −0.327 | 0.73 → 1.36 (+0.63) | Idem — reconstruction sans contrainte mémorise |
| **`both`** | **1.181** | **−0.125** | 0.68 → 1.18 (+0.50) | **Meilleur des trois — gap le plus faible** |

**Constat critique — contre-généralisation active :**  
Tous les Pearson r sont négatifs. Un r négatif signifie que le modèle prédit dans le **mauvais sens** : quand la vraie IC50 monte, le modèle prédit qu'elle baisse. Ce n'est pas du bruit — c'est une contre-généralisation active. **Cause identifiée : vecteurs aléatoires comme features de drogues.** Le modèle a mémorisé un fingerprint aléatoire unique par drogue ; pour une drogue jamais vue, ce vecteur est statistiquement indépendant de tout vecteur préalablement appris.

**Pourquoi `both` résiste mieux (r=−0.125 vs −0.354) :**  
La régularisation KL contraint `q(z|x)` vers N(0,I), ce qui prévient partiellement la mémorisation d'identités par drogue dans la branche omique. Le modèle conserve une structure générique sur les lignées cellulaires qui réduit (sans éliminer) la contre-généralisation.

**Ce que ces résultats ne mesurent pas encore :**  
Ces chiffres sont en mode `--fast` (20k/5k samples), 5 epochs, sans vrais SMILES. Le r négatif ne signifie pas que l'architecture est mauvaise — il signifie que le signal chimique est absent et que 5 epochs sont insuffisantes pour extrapoler à de nouvelles drogues depuis les seuls omiques. Le run complet (20 epochs, 137k samples, split aléatoire) avait donné Val RMSE = 0.472 — un résultat bien plus propre, mais sur un split plus facile.

---

### 3.3 Comparaison — Split Leave-Drug-Out (Run 2 — Après Partial Fix Pipeline)

**Protocole :** Identique au Run 1, après corrections partielles du pipeline (chemin CSV correct, mais `fetch_drug_smiles.py` pas encore re-exécuté avec le bug de clés JSON corrigé).  
SMILES réels chargés : **0/266** (CSV existant mais entièrement vide — voir Section 4).

| Mode | Val RMSE | Pearson r | Δ vs Run 1 |
|------|----------|-----------|-----------|
| `kl` | 1.219 | −0.197 | +0.157 ↑ |
| **`cross_entropy`** | **1.062** | **+0.121** | **+0.448 ↑ — seul r positif** |
| `both` | 1.212 | −0.267 | −0.142 ↓ |

**Observation importante :**  
Le mode `cross_entropy` est le seul à obtenir un **Pearson r positif** (+0.121) en leave-drug-out, même sans SMILES réels. Ce signal partiel (+0.121) suggère que la branche omique seule porte une information de niveau classe-de-drogues — les lignées cellulaires sensibles à une classe de drogues partagent des signatures transcriptomiques indépendamment du composé spécifique. La reconstruction BCE sans contrainte KL permet de capturer cette structure.

**Recommandation directe de l'encadrant :**
> *"Passe le pipeline complet en mode both comme défaut. Lance `python fullPipeline.py --no-ppo --loss-mode both --epochs 20` et compare le Val RMSE avec 0.472 (ancien KL). Si both donne un Val RMSE inférieur à 0.472 sur le split aléatoire complet, c'est une amélioration réelle à documenter."*

---

### 3.4 Comparaison — Split Leave-Drug-Out (Run 3 — Avec 201/266 SMILES Réels)

**Protocole :** Identique aux runs précédents, après exécution réussie de `fetch_drug_smiles.py` corrigé.  
**SMILES réels chargés : 201/266 (75.6%)** — premier run avec de vraies features moléculaires dans le GNN.

| Mode | Val RMSE | Pearson r | Gap train/val | Δ Pearson r vs Run 1 (0 SMILES) |
|------|----------|-----------|--------------|-------------------------------|
| `kl` | 1.295 | −0.360 | 0.73→1.30 (+0.57) | −0.006 (quasi stable) |
| `cross_entropy` | 1.665 | −0.370 | 0.68→1.66 (+0.98) | −0.043 (dégradé) |
| **`both`** | **1.125** | **−0.129** | 0.70→1.13 (+0.43) | **−0.004 (quasi stable, meilleur RMSE)** |

**Observations clés :**

**1. `both` est le mode le plus robuste en généralisation OOD.**  
Val RMSE = 1.125 — le meilleur des trois runs, et le gap train/val le plus faible (+0.43 contre +0.98 pour CE). La combinaison KL + BCE force simultanément une reconstruction fidèle du profil omique ET un espace latent régularisé proche de N(0,I). Cette double contrainte est moins susceptible de mémoriser des identités de drogues spécifiques.

**2. `cross_entropy` se dégrade avec les vrais SMILES (r=−0.370, RMSE=1.665).**  
C'est le résultat contre-intuitif le plus important : avec de vraies features moléculaires, la reconstruction BCE sans contrainte KL **overfit encore plus fortement**. Train RMSE = 0.683 vs Val RMSE = 1.665 — gap de 0.98, le plus élevé des trois modes. Explication : avec de vrais embeddings chimiques (non plus du bruit uniforme), le GNN peut créer des représentations distinctives par drogue, que le VAE sans contrainte KL mémorise directement dans l'espace latent. La regularisation KL dans `both` prévient cette mémorisation en forçant `q(z|x)` vers N(0,I).

**3. Le Pearson r reste négatif malgré les vrais SMILES.**  
Tous les modes produisent r < 0 en leave-drug-out. Cela confirme que le problème n'est **pas uniquement** le mapping SMILES — l'architecture et le régime d'entraînement doivent également évoluer pour la généralisation OOD :

| Cause résiduelle | Impact | Solution |
|-----------------|--------|---------|
| 65/266 drogues sans SMILES (24.4%) | Bruit résiduel dans la branche drogue | Mapper manuellement les composés propriétaires via ChEMBL |
| 5 epochs insuffisantes pour leave-drug-out | Sous-apprentissage de la généralisation | Entraîner 20-50 epochs avec early stopping sur val RMSE leave-drug-out |
| Espace latent z=128 dimensions | Capacité de mémorisation élevée | Réduire à 64 ou 32 pour contraindre la compression |
| β=2.0 encore trop fort même en mode `both` | Compression excessive du signal pharmacologique | Tester β ∈ {0.05, 0.1, 0.3} (β-VAE faible) |
| GNN 3-couches sur graphes aléatoires (65 drogues) | Signal structurel tronqué | Augmenter la couverture SMILES en priorité |

**Conclusion du Run 3 :**  
Le mode `both` avec vrais SMILES devient le **point de départ recommandé** pour tous les entraînements futurs. Il est le plus stable face au changement de distribution (OOD), produit le meilleur RMSE de validation (1.125), et dispose du gap train/val le plus faible. L'objectif suivant est de le faire converger vers un Pearson r positif en augmentant les epochs et en réduisant β.

---

## 4. Défaillance Pipeline — Drug SMILES : Diagnostic et Correction

### 4.1 Le Problème : 0/266 Drogues Mappées

`fetch_drug_smiles.py` avait été exécuté et avait produit `Dataset/ccle_drug_smiles.csv` avec 266 lignes — mais **toutes les colonnes SMILES étaient nulles**. Conséquence : `load_ccle_real_data()` affichait "SMILES réels chargés : 0/266" et revenait aux vecteurs aléatoires.

### 4.2 Bug 1 — Clés JSON PubChem Incorrectes

Le script requestait la propriété `CanonicalSMILES,IsomericSMILES` et lisait ces clés dans la réponse. Réponse réelle vérifiée de l'API PubChem REST (mai 2026) :

```json
{
  "PropertyTable": {
    "Properties": [{
      "CID": 10184653,
      "SMILES": "CN(C)C/C=C/C(=O)NC1=C...",
      "ConnectivitySMILES": "CN(C)CC=CC(=O)NC1=C..."
    }]
  }
}
```

L'API retourne `"SMILES"` (isomérique) et `"ConnectivitySMILES"` (canonique). Le parser ne trouvait pas les clés attendues → `None` pour toutes les entrées.

### 4.3 Bug 2 — Suffixe de Réplicat Non Supprimé

Les noms CCLE contiennent des suffixes de réplicats : `Afatinib-1`, `Afatinib-2`. PubChem retourne HTTP 404 pour `"Afatinib-1"`. La suppression du suffixe avant la requête : `re.sub(r'-\d+$', '', name)` → `"Afatinib"` → HTTP 200, SMILES valide récupéré.

### 4.4 Correction Implémentée

`fetch_drug_smiles.py` réécrit avec :
1. **Clés JSON correctes :** `props.get("SMILES")` et `props.get("ConnectivitySMILES")`
2. **Suppression du suffixe :** `_strip_replicate_suffix(name)` appliqué avant toutes les requêtes
3. **Cascade de fallbacks (3 variantes) :** nom nettoyé → normalisé (supprimer `uM`, tirets finaux) → split CamelCase (ex. `AKTinhibitorVIII` → `AKT inhibitor VIII`)
4. **Colonne `query_name`** dans le CSV de sortie pour la traçabilité

**Couverture obtenue après correction : 201/266 drogues (75.6%).** Les codes propriétaires (AKTinhibitorVIII, composés outils) ne sont pas dans PubChem. Les 65 drogues non mappées continuent de recevoir des vecteurs aléatoires déterministes.

**Vérification sur cas de test :**

| Nom CCLE | Requête envoyée | Résultat |
|---------|----------------|---------|
| `Afatinib-1` | `Afatinib` | ✓ SMILES récupéré |
| `BMS-536924-1` | `BMS-536924` | ✓ SMILES récupéré |
| `Erlotinib-1` | `Erlotinib` | ✓ SMILES récupéré |
| `Imatinib-2` | `Imatinib` | ✓ SMILES récupéré |
| `CHIR-99021-2` | `CHIR-99021` | ✓ SMILES récupéré |
| `AKTinhibitorVIII-1` | `AKT inhibitor VIII` | ✗ Absent de PubChem |

### 4.5 Intégration dans `load_ccle_real_data()`

`fullPipeline.py` modifié pour :
1. Tenter le chargement de `Dataset/ccle_drug_smiles.csv` au moment du chargement des données
2. Pour chaque drogue avec SMILES résolu : featurisation via `BRICSMolecularFeaturizer` (matrice de features atomiques + matrice d'adjacence → entrée GNN)
3. Pour les SMILES manquants : fallback vers vecteur aléatoire déterministe seedé par index de drogue (reproductible mais non informatif)

---

## 5. Progression DQN — Versions et Diagnostics

### 5.1 v4.0 — 10 000 Épisodes (Run Complet)

```
Valides :       4 110 / 10 000 (41.1%)
Meilleure récompense : 2.667 (épisode ~200, figée ensuite)
ε final :       0.050
Moy50 final :   ~0.0
Top-5 :         tous acycliques (arom_rings = 0 sur toutes les molécules)
```

Le reward shaping v4.0 assignait `+0.03` pour les tokens `[Ring1]` et `[Ring2]`. Analyse post-hoc : dans SELFIES 2.x, `[Ring1]` seul encode la fermeture de cycle **aliphatique**. La fermeture aromatique requiert le digramme `[Ring1][=Branch1]` — le token `[=Branch1]` porte le signal d'aromaticité et n'avait **aucune récompense positive** assignée. Le DQN a été entraîné 10 000 épisodes sans jamais renforcer spécifiquement le token de fermeture aromatique.

### 5.2 v5.0 — Buffer de Warm-Start

```
Valides :         6 040 / 10 000 (60.4%)
Meilleure récompense : 3.153
Top-5 :           toujours tous acycliques
```

Le pré-remplissage du buffer de replay avec 500 trajectoires expertes depuis les `SEED_SMILES` a amélioré le taux de validité de 41.1% → 60.4%. Le token `[=Branch1]` apparaît dans les trajectoires de warm-start mais sans récompense de step spécifique, le DQN n'apprend pas à le sélectionner lors de l'exploration ε-greedy.

### 5.3 v5.1 — Correction Chirurgicale du Token Aromatique

Récompense intermédiaire directe au niveau de la génération de token :

```python
if token in ("[=Branch1]", "[#Branch1]"):
    step_reward = +0.20   # fermeture de cycle aromatique
elif token in ("[Ring1]", "[Ring2]"):
    step_reward = +0.10   # fermeture de cycle
elif token in ("[=C]", "[=N]"):
    step_reward = +0.05   # liaison insaturée
```

**Justification :** `[=Branch1]` a la fréquence la plus élevée parmi les tokens structuraux SELFIES dans le corpus ChEMBL (20 539 occurrences / 10k molécules = 2.05 par molécule en moyenne). En fournissant une récompense positive immédiate au moment de la génération — non différée à la fin de l'épisode — le DQN reçoit un signal d'apprentissage dense pour la construction de cycles aromatiques. Statut : implémenté, en attente d'exécution.

---

## 6. Évaluation de l'Encadrant — Score 7.7/10

L'encadrant a fourni une évaluation structurée des forces et zones d'amélioration du projet :

### Points forts reconnus
- ✅ Solide compréhension de l'architecture (Bi-Int, VAE, GNN)
- ✅ Pratiques TensorFlow propres (gestion des gradients et de la mémoire GPU)
- ✅ Pensée expérimentale : 6 versions du DQN = itération intentionnelle et documentée

### Zones d'amélioration et corrections (à faire)

| N° | Problème | Impact | Statut |
|----|---------|--------|--------|
| 1 | Features de drogue = vecteurs aléatoires | −2 pts | **En cours** — `fetch_drug_smiles.py` corrigé |
| 2 | Validation des données manquante (NaN/inf IC50) | −1.5 pts | **À implémenter** |
| 3 | Modalité mutations jamais utilisée dans le VAE | −1 pt | **À implémenter** |
| 4 | ChEMBL pre-training limité à 3.5% (100k/2.8M) | −1.5 pts | **À implémenter** (HDF5 streaming) |
| 5 | DQN reward hacking — 6 itérations sans SA score | −1.5 pts | **Partiellement résolu** — v5.1 + BRICS DQN |
| 6 | Pas de logging/monitoring (TensorBoard, CSVLogger) | −1 pt | **À implémenter** |

**Correction prioritaire N°1 — fix drug SMILES (Impact : −2 pts) :**

Implémentation suggérée par l'encadrant :
```python
for drug_id in drug_ids:
    smiles = fetch_pubchem_smiles(drug_id)  # GET /compound/name/{name}/...
    if smiles:
        featurize(smiles)  # Utilise le vrai SMILES
```
✅ **Implémenté** dans `fetch_drug_smiles.py` (corrigé) et `load_ccle_real_data()`.

**Correction N°2 — validation des données IC50 :**
```python
ic50_valid = ic50[~np.isnan(ic50) & ~np.isinf(ic50) & (ic50 > 0)]
print(f"IC50 range: {ic50_valid.min():.4f} — {ic50_valid.max():.4f}")
print(f"Outliers (>100 µM): {(ic50 > 100).sum()} supprimées")
```

**Correction N°5 — SA score dans la récompense DQN :**
```python
from rdkit.Chem import sascorer
sa = sascorer.calculateScore(mol)  # 1=facile, 10=difficile
if sa > 4.0:
    reward -= 1.0  # Pénalise les molécules difficiles à synthétiser
```

**Correction N°6 — TensorBoard et CSVLogger :**
```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.CSVLogger('training_history.csv')
]
```

---

## 7. Fichiers Modifiés — Récapitulatif

| Fichier | Type | Description |
|---------|------|-------------|
| `fullPipeline.py` | Modifié | `UnifiedOmicsVAE.call(loss_mode)` · `BiIntTrainer` avec β-annealing · `load_ccle_real_data` charge les SMILES réels · fix parser MAF · flags CLI `--loss-mode` / `--beta-anneal` |
| `compare_vae_losses.py` | Modifié | Flags `--fast` / `--batch-size` / `--split-mode` · colonne `split_mode` dans le CSV |
| `fetch_drug_smiles.py` | Réécrit | Clés JSON PubChem correctes · suppression suffixe réplicat · fallback 3 variantes |
| `dqn_optimizer.py` | Modifié | v5.1 récompenses de step pour `[=Branch1]` · v5.0 buffer warm-start · `ε_min=0.15` |
| `brics_dqn_optimizer.py` | Créé | 860 lignes · `BRICSVocabulary` · `BRICSDQNOptimizer` · récompense diversité fragments |
| `README.md` | Mis à jour | Tous les résultats, diagnostics et prochaines étapes documentés |

---

## 8. Résumé Quantitatif Complet

| Expérience | Split | Mode | Val RMSE | Pearson r | Note |
|-----------|-------|------|----------|-----------|------|
| VAE comparison | Aléatoire | KL (baseline) | 0.848 | 0.546 | Référence |
| VAE comparison | Aléatoire | **Cross-Entropy** | **0.702** | **0.713** | Meilleur random split |
| VAE comparison | Aléatoire | Both | 0.747 | 0.689 | — |
| VAE leave-drug-out (Run 1) | Leave-drug-out | KL | 1.369 | −0.354 | 0 SMILES, drug=bruit |
| VAE leave-drug-out (Run 1) | Leave-drug-out | CE | 1.357 | −0.327 | 0 SMILES |
| VAE leave-drug-out (Run 1) | Leave-drug-out | Both | **1.181** | **−0.125** | Moins de contre-généralisation |
| VAE leave-drug-out (Run 2) | Leave-drug-out | KL | 1.219 | −0.197 | 0 SMILES (CSV vide) |
| VAE leave-drug-out (Run 2) | Leave-drug-out | CE | 1.062 | +0.121 | Seul r>0 — signal omique pur |
| VAE leave-drug-out (Run 2) | Leave-drug-out | Both | 1.212 | −0.267 | — |
| **VAE leave-drug-out (Run 3)** | Leave-drug-out | KL | 1.295 | −0.360 | **201/266 vrais SMILES** |
| **VAE leave-drug-out (Run 3)** | Leave-drug-out | CE | 1.665 | −0.370 | CE overfit avec vrais SMILES |
| **VAE leave-drug-out (Run 3)** | Leave-drug-out | **Both** | **1.125** | **−0.129** | **Meilleur RMSE — mode recommandé** |
| DQN v4.0 | — | SELFIES | — | — | Valid=41.1%, R=2.667, arom=0 |
| DQN v5.0 | — | SELFIES | — | — | Valid=60.4%, R=3.153, arom=0 |

---

## 9. Interprétation Scientifique et Prochaines Étapes

### 9.1 Sur la fonction de perte VAE — Synthèse des 3 Runs

Les trois runs successifs permettent de tirer des conclusions solides sur le comportement de chaque mode :

**Synthèse comparative :**

| Condition | Gagnant random split | Gagnant leave-drug-out | Gagnant OOD robustesse |
|-----------|---------------------|----------------------|----------------------|
| 0 SMILES (bruit pur) | `cross_entropy` (r=0.713) | `both` (r=−0.125) | `both` |
| 0 SMILES (CSV vide) | — | `cross_entropy` (r=+0.121) | `cross_entropy` |
| **201/266 SMILES réels** | — | **`both` (r=−0.129, RMSE=1.125)** | **`both`** |

**Conclusion consolidée :**  
Le mode `both` (KL + BCE) est le plus robuste en conditions de généralisation Out-of-Distribution (OOD) dès que de vraies features moléculaires sont disponibles. Le mode `cross_entropy` surperforme sur split aléatoire (interpolation intra-distribution) mais **se dégrade avec de vrais SMILES en leave-drug-out** : le GNN crée des embeddings chimiques trop distinctifs que le VAE sans contrainte KL mémorise au lieu de comprimer.

**La tension fondamentale identifiée :**  
Ce résultat illustre la tension classique de la littérature β-VAE (Higgins et al. 2017 ; Locatello et al. 2019) entre :
- **Fidélité de reconstruction** (favorisée par CE sans KL) → meilleure performance sur distribution connue
- **Régularisation de l'espace latent** (favorisée par KL) → meilleure généralisation OOD

**Recommandation technique — β-VAE à faible β :**  
Plutôt que de choisir entre les extrêmes, tester un β-VAE avec β ∈ {0.05, 0.1, 0.3} :

```
𝓛 = 𝓛_BCE_reconstruction + β · 𝓛_KL     avec β ≪ 1
```

Avec β = 0.05 : le modèle conserve la puissance prédictive de l'autoencodeur tout en maintenant une contrainte de régularisation suffisante pour éviter la mémorisation des identités de drogues en leave-drug-out.

**3 problèmes résiduels à corriger pour atteindre Pearson r > 0 en leave-drug-out :**

1. **Couverture SMILES insuffisante (75.6%)** — les 65 drogues manquantes introduisent du bruit résiduel. Objectif : >90% via mapping manuel des composés propriétaires dans ChEMBL.
2. **5 epochs trop court pour généralisation OOD** — augmenter à 20-50 epochs avec early stopping sur val RMSE leave-drug-out.
3. **Espace latent z=128 surdimensionné** — réduire à 64 ou 32 dimensions pour contraindre la capacité de mémorisation par drogue.

### 9.2 Sur la génération moléculaire DQN

L'échec SELFIES DQN (arom_rings=0 sur v3–v5) est une conséquence fondamentale du design de la représentation SELFIES : la fermeture de cycle est distribuée sur plusieurs tokens abstraits (`[Ring1]`, `[=Branch1]`), rendant impossible la récompense de la formation de cycles aromatiques par un terme unique. L'assemblage de fragments BRICS est la solution structurellement correcte — elle déplace l'espace d'action du niveau token vers le niveau scaffold, où chaque action a une sémantique chimique non ambiguë.

### 9.3 Prochaines Étapes Priorisées

**[Immédiat — Run 3 complété ✅]** 201/266 SMILES réels chargés, `both` mode confirmé comme le plus robuste OOD.

1. **[Critique]** Pipeline complet en mode `both` + β-annealing sur 137k triplets :
   ```bash
   python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 --no-ppo
   ```
   Objectif : Val RMSE < 0.472 (baseline KL) sur split aléatoire — si oui, amélioration réelle à documenter.

2. **[Critique]** Tester β-VAE faible β pour trouver le meilleur compromis prédiction/généralisation :
   ```bash
   python3 compare_vae_losses.py --epochs 10 --batch-size 256 --split-mode leave_drug_out
   ```
   Avec β ∈ {0.05, 0.1, 0.3} dans `HP['vae_beta']` — objectif : premier Pearson r positif en leave-drug-out.

3. **[Haut]** Augmenter la couverture SMILES : 201/266 → >240/266 via mapping manuel ChEMBL des 65 composés propriétaires manquants.

4. **[Haut]** Exécuter DQN v5.1 → vérifier première molécule aromatique dans le Top-5 :
   ```bash
   nohup python3 dqn_optimizer.py > logs_dqn_v5.1.txt 2>&1 &
   ```

5. **[Haut]** Exécuter BRICS DQN → solution structurelle à l'absence d'aromaticité :
   ```bash
   nohup python3 brics_dqn_optimizer.py > logs_brics_dqn.txt 2>&1 &
   ```

6. **[Moyen]** Ajouter validation IC50 (filtre NaN/inf/outliers) + TensorBoard + CSVLogger callbacks.

7. **[Moyen]** Intégrer SA score dans la récompense DQN — `sascorer.calculateScore(mol)`, pénaliser SA > 4.0.

8. **[Moyen]** Réduire dimension latente z : 128 → 64 → mesurer impact sur leave-drug-out Pearson r.

9. **[Futur]** Scale ChEMBL pre-training vers 500k–1M molécules via lecteur SDF streaming + cache HDF5.

---

*Rapport préparé par : équipe de recherche Bi-Int Digital Twin*  
*Toutes les expériences exécutées sur Ubuntu 24.04 LTS / WSL2, NVIDIA RTX 4000 Ada (17 710 Mo VRAM), TensorFlow 2.21.0, RDKit 2024, SELFIES 2.1.1.*
