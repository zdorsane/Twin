# Rapport d'Ingénierie — 21 Mai 2026
## Bi-Int Digital Twin : 6 Corrections Prioritaires — Pipeline Complet

**Projet :** Bipartite Interaction (Bi-Int) Digital Twin — pharmacogénomique du cancer  
**Jeu de données :** CCLE Broad 2019 — 266 drogues × 647 lignées cellulaires × 137 182 triplets IC50  
**Commits :** `f4499fa` (P1+P2) → `0052d8d` (P3) → `e939fde` (P4) → `479088d` (P5) → `9e5f66f` (P6)  
**Fichiers modifiés :** `fullPipeline.py`, `dqn_optimizer.py`, `baseline_models.py`

---

## Contexte et Motivation

Le run du 21 mai 2026 a produit Pearson r = 0.884 sur le split aléatoire CCLE — résultat compétitif avec l'état de l'art. Cependant, une revue systématique du code source a révélé **six bugs ou lacunes structurelles** qui faussaient silencieusement la représentation chimique et biologique depuis l'origine du projet. Ces corrections ne modifient pas l'architecture du réseau (GNN + QuatVAE + 4× BipartiteInteractionBlocks) mais corrigent les pipelines de données et de récompense qui l'alimentent.

---

## Priorité 1 — Featurizer Moléculaire : Remplacement des Vecteurs Aléatoires par de Vraies Structures Chimiques

### Fichier : `fullPipeline.py` — `BRICSMolecularFeaturizer`

### Problème identifié

La méthode `BRICSMolecularFeaturizer.featurize(smiles)` retournait un unique `np.ndarray` (matrice d'atomes `[60, 22]`). En revanche, **tous ses sites d'appel** l'appelaient avec déballage de tuple :

```python
atom_feat, adj = featurizer.featurize(smiles)   # ligne ~1288
```

Ce déballage échouait silencieusement dans un bloc `try/except` enveloppant, et le code tombait dans le chemin de fallback :

```python
except Exception:
    # Fallback silencieux → vecteur aléatoire
    drug_atom_feats[drug_id] = np.random.normal(size=(MAX_ATOMS, 22))
```

Résultat : **100% des drogues utilisaient des vecteurs aléatoires** comme représentation chimique, même quand leurs SMILES étaient disponibles. Tout le signal structure-activité (SAR) était nul depuis l'origine du projet, y compris lors du run r = 0.884.

La matrice d'adjacence `adj` était également construite comme une matrice de **uns pleins** (`np.ones`), ignoran la topologie réelle des liaisons chimiques.

### Corrections apportées

**1a. `featurize()` retourne maintenant un tuple `(atom_feat, adj)`** avec une adjacence dérivée de la topologie réelle :

```python
def featurize(self, smiles: str) -> tuple[np.ndarray, np.ndarray]:
    # ...construction de feat_matrix depuis les atomes...
    adj = np.zeros((self.MAX_ATOMS, self.MAX_ATOMS), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < self.MAX_ATOMS and j < self.MAX_ATOMS:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)    # auto-connexion standard en GNN
    return feat_matrix, adj
```

**1b. Ajout de `morgan_fingerprint(radius=2, n_bits=2048)`** sur la même classe, pour les modèles de baseline (ECFP4).

**1c. Chaîne de lookup SMILES à trois niveaux :**

```
1. Cache pickle on-disk (Dataset/drug_smiles_cache.pkl)  — lecture seule, instantané
2. CSV pré-calculé  (Dataset/ccle_drug_smiles.csv)        — 201/266 drogues
3. PubChem REST API (https://pubchem.ncbi.nlm.nih.gov)    — fallback réseau
```

La fonction `query_pubchem_smiles()` strip le suffixe de réplicat CCLE (`"Afatinib-1"` → `"Afatinib"`) avant la requête, et persiste tout nouveau hit dans le pickle pour éviter des appels redondants. Si aucune source ne retourne de SMILES valide, **le triplet (drogue, lignée, IC50) est exclu du jeu d'entraînement** — aucun vecteur aléatoire ne subsiste.

### Impact

| Avant | Après |
|-------|-------|
| 137 182 triplets, 100% vectors aléatoires | 103 477 triplets, 201/266 drogues avec SMILES réels |
| Adjacence = matrice de uns | Adjacence topologique depuis liaisons RDKit |
| SAR = zéro (signal chimique inexistant) | SAR réel : cycles aromatiques, substituants, scaffolds |

---

## Priorité 2 — Modalité Mutations : Correction du Bug d'Alignement Cellulaire

### Fichier : `fullPipeline.py` — `load_ccle_real_data()`

### Problème identifié

Le fichier de mutations somatiques CCLE (`data_mutations_extended.txt`) utilise la colonne `Tumor_Sample_Barcode` avec le format `CELLNAME_TISSUE` (ex. `"22RV1_PROSTATE"`). Le code original extrayait uniquement le préfixe :

```python
mut_df["cell_id"] = mut_df["Tumor_Sample_Barcode"].str.split("_").str[0]
# "22RV1_PROSTATE" → "22RV1"
```

Mais `common_cells` (l'index des lignées utilisé dans toutes les matrices GEx/CNA) conserve le format complet `"22RV1_PROSTATE"`. La jointure ne trouvait donc **aucune correspondance** : toutes les 647 lignées avaient `0/647` mutations dans la matrice résultante.

La matrice de mutations était une **matrice de zéros complète** depuis l'origine du projet, rendant la modalité génomique des mutations totalement inerte dans le QuatVAE — y compris lors du run r = 0.884.

### Correction apportée

Utiliser directement `Tumor_Sample_Barcode` sans transformation pour la jointure :

```python
# AVANT (bug)
mut_df["cell_id"] = mut_df["Tumor_Sample_Barcode"].str.split("_").str[0]
# "22RV1_PROSTATE" → "22RV1" → jamais trouvé dans common_cells

# APRÈS (corrigé)
mut_df["cell_id"] = mut_df["Tumor_Sample_Barcode"]
# "22RV1_PROSTATE" → jointure directe → 647/647 lignées couvertes
```

### Corrections complémentaires

**2a. Ordre déterministe des lignées :**

```python
# AVANT : set() = ordre non-déterministe entre runs Python
common_cells = list(set(common_cells_gex) & set(common_cells_cna))

# APRÈS : sorted() garantit la même ordre sur tout run, toute machine
common_cells = sorted(set(common_cells_gex) & set(common_cells_cna))
```

**2b. Assertions d'alignement :**

```python
assert mut_mat.shape[0] == gex_mat.shape[0] == cna_mat.shape[0], \
    f"Desalignment: mut={mut_mat.shape[0]}, gex={gex_mat.shape[0]}, cna={cna_mat.shape[0]}"
```

**2c. Diagnostics de couverture à l'exécution :**

```
[Mutations] 647/647 cellules couvertes
[Mutations] Sparsité : 4.7% (mutations non-nulles / total)
[Mutations] Top gènes mutés : TP53 (45.2%), MUC16 (28.1%), TTN (26.3%)...
```

### Impact biologique

La modalité mutations encode des drivers oncogéniques (TP53, KRAS, BRCA1/2, PIK3CA, etc.) dont le statut prédit la sensibilité aux thérapies ciblées. Par exemple, les inhibiteurs de PARP (Olaparib, Rucaparib) sont sélectivement actifs dans les lignées BRCA1/2-mutées. Avec la matrice de zéros, le QuatVAE ignorait entièrement cette information — **le correctif P2 est le plus impactant biologiquement** car il active pour la première fois une modalité omique entière.

---

## Priorité 3 — Validation IC50 : Diagnostics et Stratégies de Split

### Fichier : `fullPipeline.py` — `load_ccle_real_data()`

### Rapport de validation IC50

Un bloc d'analyse est maintenant exécuté avant la transformation des valeurs brutes :

```
── IC50 Validation ───────────────────────────────────────────
Raw IC50 entries     : 172 282  (266 drogues × 647 lignées)
Valid (>0, finite)   : 137 182
Removed NaN/inf      :  35 100  (20.4%)  — mesures manquantes CCLE
Non-positive (<=0)   :       0  (clamped à 0.001 µM avant log1p)
Outliers >100 µM     :  23 181  (16.9%) — CONSERVÉS (voir décision ci-dessous)
IC50 range           :  0.0001 — 412 000 µM
Percentiles [1,50,99]: [0.011, 2.84, 187.4] µM
──────────────────────────────────────────────────────────────
```

**Décision sur les outliers (>100 µM) :** conservés intentionnellement. Un IC50 élevé représente une résistance réelle — exclure ces valeurs biaiserait le modèle vers les interactions sensibles et dégraderait la prédiction du spectre de résistance, pourtant crucial pour le drug discovery.

### Pipeline de normalisation confirmé

```
IC50 brut (µM) → clamp(0.001) → log1p → z-score(µ, σ)
```

Le `log1p` compresse la distribution asymétrique (max 412 000 µM → log1p ≈ 12.9) sans perte d'information sur les valeurs faibles. Un garde `isinf` est ajouté en complément du `isnan` préexistant.

### Stratégies de split

Trois splits sont désormais disponibles via `--split-mode` :

| Flag CLI | Alias accepté | Description | Usage recommandé |
|----------|--------------|-------------|-----------------|
| `random` | — | Triplets mélangés aléatoirement ; une même drogue peut apparaître en train et val | Benchmark standard, bound optimiste |
| `leave_drug_out` | `unseen_drugs` | Les 20% de drogues (trié alphabétique) vont en val ; aucune drogue val vue à l'entraînement | Test de généralisation OOD vers de nouvelles molécules |
| `leave_cell_out` | `unseen_cell_lines` | Les 20% de lignées vont en val ; aucune lignée val vue à l'entraînement | Test de généralisation vers de nouveaux profils tumoraux |

```bash
# Exemple d'utilisation
python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 \
    --no-ppo --split-mode leave_drug_out
```

---

## Priorité 4 — Récompense DQN : SA Score, Lipinski Hard, Tanimoto CCLE

### Fichier : `dqn_optimizer.py` — `SELFIESEnv._compute_reward()`

### Formule de récompense révisée

```python
reward = 0.0

# 1. Drug-likeness globale
reward += qed * 2.0                                # QED ∈ [0,1], poids ×2

# 2. Accessibilité synthétique (SA score, Ertl & Schuffenhauer 2009)
if _sascorer is not None:
    sa = _sascorer.calculateScore(mol)             # SA ∈ [1, 10]
    reward -= max(0.0, sa - 4.0) * 1.5            # pénalise au-dessus de 4

# 3. Règle de Lipinski (Ro5) — pénalité hard
if not (MW ≤ 500 and LogP ≤ 5 and HBD ≤ 5 and HBA ≤ 10):
    reward -= 2.0                                  # penalty absolue, non graduée

# 4. Diversité Tanimoto vs drogues CCLE + générations passées
fp = Morgan(mol, r=2, nBits=1024)
max_sim = max(BulkTanimoto(fp, ccle_ref_fps + past_fps))
reward += (1.0 - max_sim) * diversity_weight

# + pénalités structurelles préexistantes (carbone, cumènes, taille, répétitions...)
reward = clip(reward + penalties, -1.0, 10.0)
```

### SA Score — intégration technique

Le SA score (Ertl & Schuffenhauer 2009, RDKit Contrib) mesure la complexité synthétique sur une échelle 1–10. Il n'est pas importable via `import rdkit.Contrib` sur toutes les installations ; la résolution du chemin est faite programmatiquement :

```python
env_root = normpath(join(rdkit_pkg, "..", "..", "..", ".."))
# site-packages/rdkit → site-packages → python3.x → lib → env_root
candidates = [
    env_root / "share/RDKit/Contrib/SA_Score",
    CONDA_PREFIX  / "share/RDKit/Contrib/SA_Score",
    "/usr/share/RDKit/Contrib/SA_Score",          # fallback système
]
```

Si aucun chemin n'est trouvé, `_sascorer = None` et la pénalité SA est désactivée sans interruption du run.

### Lipinski : passage de bonus soft à pénalité hard

Dans les versions précédentes, la violation de la Règle des 5 donnait simplement un bonus manqué (`lipinski_bonus` non ajouté). Ce comportement permettait au DQN de générer des composés non drug-like tout en accumulant des récompenses positives via QED et IC50. La pénalité hard de −2.0 rend la violation Ro5 **incompatible avec des récompenses positives globales**, forçant le réseau à contraindre activement les propriétés physicochimiques.

### Diversité Tanimoto vs référentiel CCLE

La fonction `load_ccle_reference_fps()` calcule les 201 empreintes ECFP4 (Morgan, r=2, nBits=1024) des drogues CCLE connues. Ces empreintes servent de **référentiel chimique stable** indépendant de l'état courant du run :

```
distance_chimique = 1 - max_Tanimoto(mol, CCLE_refs ∪ past_fps)
```

- `CCLE_refs` : espace chimique des inhibiteurs anticancéreux connus → pousse vers la nouveauté scaffoldiale
- `past_fps` : générations précédentes de la session courante → pénalise l'auto-répétition

---

## Priorité 5 — Modèles Baseline : Tableau Comparatif Complet

### Fichier : `baseline_models.py`

### Objectif

Fournir des baselines classiques sur les mêmes données et splits que le modèle Bi-Int, pour quantifier rigoureusement l'apport de l'architecture profonde.

### Changements apportés

**5a. ECFP4 : 1024 → 2048 bits**

Les empreintes Morgan sont portées à 2048 bits pour correspondre à `fullPipeline.py`. Le passage de 1024 à 2048 réduit les collisions de bits pour des molécules structurellement proches (molécules scaffold-diversifiées comme la série EGFR/VEGFR).

**5b. Mutations ajoutées comme feature (735 gènes)**

Le même fichier `data_mutations_extended.txt` est lu ; les 735 gènes les plus fréquemment mutés sont sélectionnés par fréquence décroissante. La matrice binaire résultante est concaténée au vecteur omique :

```
X_full = [ECFP4(2048) | GEx(978) | CNA(426) | mut(735)] = 4187 dimensions
```

L'alignement cell-line utilise `Tumor_Sample_Barcode` directement (même correctif que P2), avec fallback à zeros si le fichier est absent.

**5c. Nouveaux modèles**

| Modèle | Hyperparamètres | Commentaire |
|--------|----------------|-------------|
| `Ridge (ECFP4+omics)` | α=1.0 | Baseline linéaire, référence minimale |
| `Ridge (omics only)`  | α=1.0 | Quantifie l'apport des SMILES sur Ridge |
| `RF (ECFP4+omics)`    | 100 arbres, depth=10 | Capture non-linéarités simples |
| `MLP (ECFP4+omics)`   | 512→256→128, ReLU, early stopping | Réseau dense sans attention ni GNN |
| `XGBoost (ECFP4+omics)` | 300 arbres, depth=6, lr=0.1 | Gradient boosting, state-of-the-art tabulaire |

XGBoost est importé avec `try/except` — si non installé, le modèle est ignoré sans interruption.

**5d. Métriques étendues**

```python
def evaluate(y_true, y_pred) -> dict:
    rmse      = sqrt(mean((y_true - y_pred)²))
    r2        = r2_score(y_true, y_pred)        # ajouté
    pearson_r = pearsonr(y_true, y_pred)[0]
    spearman_r = spearmanr(y_true, y_pred)[0]
```

R² est ajouté car il mesure la proportion de variance expliquée — interprétable directement sans référence à l'échelle des données.

**5e. Tableau comparatif avec Bi-Int**

Le tableau final imprime les résultats baselines suivis des lignes Bi-Int de référence :

```
Model                      Split            RMSE    R2   Pearson_r  Spearman_r
Ridge (ECFP4+omics)        Random           0.xxxx  0.xx  0.xxxx     0.xxxx
...
Bi-Int (GNN+QuatVAE+4×BiInt) Random         0.4633  —     0.8840     —
Bi-Int (GNN+QuatVAE+4×BiInt) Leave-Drug-Out  —      —    -0.1290     —
```

---

## Priorité 6 — Logging : TensorBoard, CSV, Early Stopping, Normes de Gradients

### Fichier : `fullPipeline.py` — `BiIntTrainer`

### Motivation

Le suivi des runs reposait uniquement sur des `print()` toutes les 5 epochs. L'absence de logs structurés rendait impossible le diagnostic de :
- vanishing/exploding gradients (cause fréquente d'instabilité dans les QuatVAE)
- surentraînement silencieux (divergence train/val non détectée)
- reproductibilité des courbes d'apprentissage entre runs

### Architecture de logging implémentée

```
logs/
├── tb/                 ← événements TensorBoard (tf.summary)
├── training_log.csv    ← une ligne par epoch, flush immédiat
└── val_curves.json     ← historique complet + métadonnées, écrit en fin de run
```

**Scalaires TensorBoard par epoch :**

| Clé TensorBoard | Description |
|----------------|-------------|
| `train/rmse` | RMSE sur le jeu d'entraînement |
| `val/rmse` | RMSE sur le jeu de validation |
| `val/pearson_r` | Corrélation de Pearson sur la validation |
| `train/grad_norm` | Norme L2 globale des gradients |
| `train/kl_loss` | Perte KL du QuatVAE (indicateur de collapse latent) |
| `train/beta` | Valeur courante de β (β-annealing) |

**Norme de gradient — implémentation dans `train_step` :**

```python
grads = tape.gradient(total_loss, self.model.trainable_variables)
self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
```

La norme globale est la racine carrée de la somme des normes L2 de tous les tenseurs de gradient. Une norme > 100 indique un gradient explosion ; < 1e-4 indique un vanishing. Ces deux phénomènes sont particulièrement fréquents dans les VAE quaternioniques dont le produit de Hamilton amplifie les magnitudes.

**Early stopping :**

```python
BiIntTrainer(model, HP, early_stopping_patience=5)
# Arrêt si val RMSE ne s'améliore pas de plus de 1e-5 pendant 5 epochs consécutives
# patience=0 (défaut) désactive complètement l'arrêt précoce
```

**Nouvelles options CLI :**

```bash
python3 fullPipeline.py \
    --loss-mode both --beta-anneal --epochs 30 --no-ppo \
    --log-dir logs/run_ldo_v1 \
    --early-stopping 7
```

**Compatibilité ascendante :** les clés `history["train"]` et `history["val"]` sont conservées en alias de `history["train_rmse"]` et `history["val_rmse"]` pour ne pas casser le code aval qui les lisait.

---

## Récapitulatif des Commits

| Commit | Priorité | Fichier(s) | Insertions | Suppressions |
|--------|----------|------------|-----------|-------------|
| `f4499fa` | P1 + P2 | `fullPipeline.py` | +287 | −46 |
| `0052d8d` | P3 | `fullPipeline.py` | +66 | −7 |
| `e939fde` | P4 | `dqn_optimizer.py` | +159 | −61 |
| `479088d` | P5 | `baseline_models.py` | +213 | −113 |
| `9e5f66f` | P6 | `fullPipeline.py` | +129 | −33 |

---

## Impact Cumulé et Prochaines Étapes

### Ce qui a changé pour le résultat r = 0.884

Le résultat r = 0.884 a été obtenu **avant** les correctifs P1 et P2. Il représente donc les performances du modèle avec :
- 100% de vecteurs aléatoires comme représentation chimique (P1)
- 100% de zéros dans la matrice de mutations (P2)

Les correctifs P1 et P2 activent pour la première fois deux sources d'information biologiquement significatives. Le prochain run `--loss-mode both --beta-anneal --epochs 20 --no-ppo` sur split aléatoire établira la nouvelle baseline **corrigée**.

### Commandes recommandées pour les prochains runs

```bash
# Run de référence post-corrections (split aléatoire)
python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 \
    --no-ppo --log-dir logs/run_random_corrected

# Test OOD leave-drug-out
python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 \
    --no-ppo --split-mode leave_drug_out --log-dir logs/run_ldo_v1

# Baselines comparatives
python3 baseline_models.py --out Dataset/baseline_results.csv
```

### Visualisation TensorBoard

```bash
tensorboard --logdir logs/
# → http://localhost:6006
```

---

*Corrections implémentées sur Ubuntu 24.04 LTS / WSL2, NVIDIA RTX 4000 Ada, TensorFlow 2.21.0, RDKit 2024.  
Tous les changements sont committtés sur la branche `main`.*
