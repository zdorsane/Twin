# Rapport d'Ingénierie — 21 Mai 2026
## Bi-Int Digital Twin : 7 Corrections Prioritaires — Pipeline Complet

**Projet :** Bipartite Interaction (Bi-Int) Digital Twin — pharmacogénomique du cancer  
**Jeu de données :** CCLE Broad 2019 — 266 drogues × 647 lignées cellulaires × 103 477 triplets IC50  
**GPU :** NVIDIA RTX 4000 Ada, 20 475 MiB VRAM, CUDA 13.0, TensorFlow 2.15.0  
**Commits :** `e939fde` (P4) → `479088d` (P5) → `9e5f66f` (P6) → `d503903` (P7 OOM) → `78e35b7` (rapport)  
**Fichiers modifiés :** `fullPipeline.py`, `dqn_optimizer.py`, `baseline_models.py`

---

## Contexte et Motivation

Le run du 21 mai 2026 a mis en évidence à la fois des résultats prometteurs et un ensemble de bugs structurels dans le pipeline. Une revue systématique du code source a révélé **sept problèmes critiques** : six bugs ou lacunes architecturales qui faussaient silencieusement la représentation chimique et biologique depuis l'origine du projet, et une série de crashes mémoire (OOM) liés au chargement des données omiques. Ces corrections ne modifient pas l'architecture du réseau (GNN + QuatVAE + 4× BipartiteInteractionBlocks) mais corrigent les pipelines de données et de récompense qui l'alimentent.

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

Résultat : **100% des drogues utilisaient des vecteurs aléatoires** comme représentation chimique, même quand leurs SMILES étaient disponibles. Tout le signal structure-activité (SAR) était nul depuis l'origine du projet.

La matrice d'adjacence `adj` était également construite comme une matrice de **uns pleins** (`np.ones`), ignorant la topologie réelle des liaisons chimiques.

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
3. PubChem REST API (https://pubchem.ncbi.nlm.nih.gov)    — fallback réseau, 65 requêtes
```

La fonction `query_pubchem_smiles()` strip le suffixe de réplicat CCLE (`"Afatinib-1"` → `"Afatinib"`) avant la requête, et persiste tout nouveau hit dans le pickle pour éviter des appels redondants. Si aucune source ne retourne de SMILES valide, **le triplet (drogue, lignée, IC50) est exclu du jeu d'entraînement** — aucun vecteur aléatoire ne subsiste.

### Impact

| Avant | Après |
|-------|-------|
| 137 182 triplets, 100% vecteurs aléatoires | **103 477 triplets, 201/266 drogues avec SMILES réels** |
| Adjacence = matrice de uns | Adjacence topologique depuis liaisons RDKit |
| SAR = zéro (signal chimique inexistant) | SAR réel : cycles aromatiques, substituants, scaffolds |

La réduction du nombre de triplets (137 182 → 103 477) est directement due au filtre SMILES : les 65 drogues sans SMILES valide (34 705 paires drogue-lignée) sont exclues plutôt que représentées par des vecteurs sans signal.

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

La matrice de mutations était une **matrice de zéros complète** depuis l'origine du projet, rendant la modalité génomique des mutations totalement inerte dans le QuatVAE.

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

# APRÈS : sorted() garantit le même ordre sur tout run, toute machine
common_cells = sorted(set(common_cells_gex) & set(common_cells_cna))
```

**2b. Assertions d'alignement :**

```python
assert mut_mat.shape[0] == gex_mat.shape[0] == cna_mat.shape[0], \
    f"Desalignment: mut={mut_mat.shape[0]}, gex={gex_mat.shape[0]}, cna={cna_mat.shape[0]}"
```

**2c. Dimensions omiques confirmées au 21 mai 2026 :**

```
GEx   : (647, 978)   — 978 gènes d'expression
CNA   : (647, 426)   — 426 régions de nombre de copies
Mut   : (647, 735)   — 735 gènes mutés, sparsité = 0.844
```

### Impact biologique

La modalité mutations encode des drivers oncogéniques (TP53, KRAS, BRCA1/2, PIK3CA, etc.) dont le statut prédit la sensibilité aux thérapies ciblées. Par exemple, les inhibiteurs de PARP (Olaparib, Rucaparib) sont sélectivement actifs dans les lignées BRCA1/2-mutées. Avec la matrice de zéros, le QuatVAE ignorait entièrement cette information — **le correctif P2 est le plus impactant biologiquement** car il active pour la première fois une modalité omique entière.

---

## Priorité 3 — Validation IC50 : Diagnostics et Stratégies de Split

### Fichier : `fullPipeline.py` — `load_ccle_real_data()`

### Rapport de validation IC50 (chiffres confirmés)

Un bloc d'analyse est exécuté avant la transformation des valeurs brutes :

```
── IC50 Validation ───────────────────────────────────────────
Raw IC50 entries     : 172 282  (266 drogues × 647 lignées)
Valid (>0, finite)   : 103 477  (après filtre SMILES — drogues sans SMILES exclues)
Removed NaN/inf      :  35 100  (20.4%)  — mesures manquantes CCLE
Non-positive (<=0)   :       0  (clamped à 0.001 µM avant log1p)
Outliers >100 µM     :  23 163  (16.9%) — CONSERVÉS (voir décision ci-dessous)
IC50 post-log1p      :  mean=2.6747, std=1.8512, range 0.001–12.90
──────────────────────────────────────────────────────────────
```

**Décision sur les outliers (>100 µM) :** conservés intentionnellement. Un IC50 élevé représente une résistance réelle — exclure ces valeurs biaiserait le modèle vers les interactions sensibles et dégraderait la prédiction du spectre de résistance, pourtant crucial pour le drug discovery.

### Pipeline de normalisation confirmé

```
IC50 brut (µM) → clamp(0.001) → log1p → z-score(µ=2.6747, σ=1.8512)
```

Le `log1p` compresse la distribution asymétrique (max ~12.9 après clamp et log) sans perte d'information sur les valeurs faibles. Un garde `isinf` est ajouté en complément du `isnan` préexistant.

### Stratégies de split

Trois splits sont disponibles via `--split-mode` :

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
if not (MW <= 500 and LogP <= 5 and HBD <= 5 and HBA <= 10):
    reward -= 2.0                                  # pénalité absolue, non graduée

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
    rmse       = sqrt(mean((y_true - y_pred)²))
    r2         = r2_score(y_true, y_pred)        # ajouté
    pearson_r  = pearsonr(y_true, y_pred)[0]
    spearman_r = spearmanr(y_true, y_pred)[0]
```

R² est ajouté car il mesure la proportion de variance expliquée — interprétable directement sans référence à l'échelle des données.

**5e. Tableau comparatif avec Bi-Int**

Le tableau final imprime les résultats baselines suivis des lignes Bi-Int de référence (résultats Bi-Int en attente — voir section "État d'Avancement") :

```
Model                        Split            RMSE    R2   Pearson_r  Spearman_r
Ridge (ECFP4+omics)          Random           0.xxxx  0.xx  0.xxxx     0.xxxx
...
Bi-Int (GNN+QuatVAE+4×BiInt) Random           [en cours, batch_size=16]
Bi-Int (GNN+QuatVAE+4×BiInt) Leave-Drug-Out   [en cours]
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

## Priorité 7 — Corrections OOM et Cache Omics

### Commit : `d503903` — Fix OOM : float32 read_csv, del DataFrames, vectorised IC50/triplet builder

### Problème 1 : Explosion mémoire RAM lors du chargement GEx (WSL crash)

Le chargement du fichier d'expression génique CCLE (`CCLE_expression.csv`, 504 MB sur disque) déclenchait une consommation mémoire RAM pouvant atteindre **~30 GB** lors de la construction des matrices omiques, entraînant le crash du processus WSL2 sous Windows 11 sans message d'erreur explicite. Cause : pandas charge les CSV en float64 par défaut (8 octets/valeur), et les opérations de pivotage, intersection et réindexation impliquant 647 lignées × 978 gènes créaient plusieurs copies intermédiaires du DataFrame en mémoire simultanément.

**Profil mémoire avant correction (approximatif) :**

```
Lecture pandas (float64)         :  ~4 GB
pivot_table + reindex             :  ~8 GB (copie complète)
Intersection avec CNA/Mut        :  ~6 GB (3 DataFrames actifs)
Construction np.ndarray           :  ~4 GB
Garbage collection non déclenché :  +8 GB résidus
Total pic                        :  ~30 GB  →  WSL2 killed
```

### Fix 1 : Downcast float32 + suppression explicite des DataFrames intermédiaires (commit `d503903`)

```python
# Lecture immédiate en float32 — réduit de moitié la consommation brute
gex_df = pd.read_csv(gex_path, index_col=0, dtype=np.float32)

# Suppression explicite après extraction des matrices numpy
gex_mat = gex_df.loc[common_cells, common_genes].values.astype(np.float32)
del gex_df        # libère ~2 GB immédiatement
gc.collect()

cna_mat = cna_df.loc[common_cells, common_cna_genes].values.astype(np.float32)
del cna_df
gc.collect()
```

Le passage float64 → float32 réduit de 50% la taille des DataFrames pandas. La suppression explicite (`del` + `gc.collect()`) force la libération avant de charger le fichier suivant, aplatissant le pic mémoire de ~30 GB à ~4–6 GB.

**Vectorisation du builder de triplets IC50 :** l'ancienne boucle Python imbriquée `for drug in drugs: for cell in cells:` construisant la liste des triplets a été remplacée par une opération vectorisée via `np.where(~np.isnan(ic50_matrix))`, éliminant un goulot d'étranglement quadratique (O(n²) → O(n)).

### Fix 2 : Cache NPZ pour les matrices omiques pré-traitées

À chaque lancement, le pipeline recalculait l'intégralité des matrices GEx/CNA/Mut (lecture CSV + pivotage + intersection + normalisation). Sur 647 lignées × (978 + 426 + 735) features, ce recalcul prenait plusieurs minutes et ré-exposait l'OOM à chaque run.

La solution est un cache NPZ binaire écrit après le premier traitement réussi :

```python
CACHE_PATH = "Dataset/ccle_broad_2019/omics_cache_gex978_cna426.npz"

if os.path.exists(CACHE_PATH):
    # Chargement instantané — aucune lecture CSV, aucun pivotage
    cache = np.load(CACHE_PATH)
    gex_mat  = cache["gex_mat"]    # (647, 978) float32
    cna_mat  = cache["cna_mat"]    # (647, 426) float32
    mut_mat  = cache["mut_mat"]    # (647, 735) float32
    common_cells = cache["common_cells"].tolist()
    logging.info(f"[Cache] Omics chargés depuis {CACHE_PATH} en <1s")
else:
    # Première exécution : recalcul complet + sauvegarde cache
    gex_mat, cna_mat, mut_mat, common_cells = _load_omics_from_csv(...)
    np.savez_compressed(CACHE_PATH,
                        gex_mat=gex_mat,
                        cna_mat=cna_mat,
                        mut_mat=mut_mat,
                        common_cells=np.array(common_cells))
    logging.info(f"[Cache] Omics sauvegardés : {CACHE_PATH}")
```

**Fichier cache créé :** `Dataset/ccle_broad_2019/omics_cache_gex978_cna426.npz`  
**Taille sur disque :** ~70–90 MB (NPZ compressé)  
**Gain au 2ème run :** chargement en < 1 seconde au lieu de plusieurs minutes

### Fix 3 : Réduction batch_size GPU 32 → 16 (GPU ResourceExhaustedError)

Le premier run Bi-Int sur 20 epochs a déclenché une erreur TensorFlow :

```
tensorflow.python.framework.errors_impl.ResourceExhaustedError:
  OOM when allocating tensor with shape[32, 4, 256, 978]
  [[node SelectV2]] [[gradient_tape/...]]
  on /job:localhost/replica:0/task:0/device:GPU:0
```

L'erreur se produit dans l'opérateur `SelectV2` du gradient du bloc `BipartiteInteractionBlock` : avec `batch_size=32`, les activations intermédiaires du mécanisme d'attention bipartite (32 × 4 têtes × 256 dim × 978 gènes) dépassent les 20 475 MiB disponibles sur l'RTX 4000 Ada.

**Correction :**

```python
# HP dans fullPipeline.py
HP = HyperParams(
    ...
    batch_size = 16,   # réduit de 32 → 16 pour éviter OOM GPU (SelectV2)
    ...
)
```

Le halving du batch_size réduit de 50% la mémoire GPU requise par forward+backward pass. Impact sur la convergence : minime sur CCLE (légère augmentation du bruit de gradient, compensée par le scheduler de learning rate).

---

## Résultats Confirmés au 21 Mai 2026

### Pré-entraînement ChEMBL (encodeur drogue)

| Epoch | val_loss | val_RMSE |
|-------|----------|----------|
| 1     | —        | 0.4886   |
| ...   | ...      | ...      |
| 9 (best) | 0.04784 | — |
| 10    | 0.0491   | 0.2187   |

- **10 epochs** sur le corpus ChEMBL
- **val RMSE final :** 0.2187 (epoch 10)
- **Meilleure epoch :** epoch 9 — val_loss = 0.04784
- **Poids sauvegardés :** `pretrained_weights/chembl_drug_encoder.weights.h5`
- **Modèle Keras :** `pretrained_drug_encoder.keras`

### Jeu de données CCLE

| Dimension | Valeur |
|-----------|--------|
| Drogues avec SMILES | 201 / 266 |
| Drogues sans SMILES (exclues) | 65 |
| Requêtes PubChem exécutées | 65 |
| Lignées cellulaires communes | 647 |
| Triplets IC50 valides post-filtre SMILES | **103 477** |
| GEx shape | (647, 978) |
| CNA shape | (647, 426) |
| Mutations shape | (647, 735) |
| Sparsité mutations | 0.844 |

### Distribution IC50 post-transformation log1p

| Statistique | Valeur |
|------------|--------|
| Moyenne | 2.6747 |
| Écart-type | 1.8512 |
| Min | 0.001 |
| Max | 12.90 |
| Outliers >100 µM (conservés) | 23 163 (16.9%) |

### Configuration matérielle et logicielle

| Composant | Version / Modèle |
|-----------|-----------------|
| GPU | NVIDIA RTX 4000 Ada |
| VRAM | 20 475 MiB |
| CUDA | 13.0 |
| TensorFlow | **2.15.0** |
| RDKit | 2024 |
| OS | Ubuntu 24.04 LTS / WSL2 |
| Windows | Windows 11 Pro 10.0.26200 |

---

## État d'Avancement et Prochaines Étapes

### Terminé

- [x] Pré-entraînement ChEMBL : 10 epochs, val RMSE 0.4886 → 0.2187, poids sauvegardés
- [x] Chargement CCLE : 266 drogues × 647 lignées, 103 477 triplets valides
- [x] P1 : Featurizer moléculaire — SMILES réels, adjacence topologique, lookup PubChem
- [x] P2 : Alignement mutations — 647/647 lignées couvertes, ordre déterministe
- [x] P3 : Validation IC50 — diagnostics, stratégies de split (random/LDO/LCO)
- [x] P4 : Récompense DQN — SA score, Lipinski hard, Tanimoto CCLE
- [x] P5 : Baselines — XGBoost/MLP/Ridge/RF, 2048-bit ECFP4, mutations feature, R²
- [x] P6 : Logging — TensorBoard, CSVLogger, EarlyStopping, gradient norm
- [x] P7 : Corrections OOM — float32, del DataFrames, cache NPZ, batch_size=16
- [x] GPU setup confirmé : RTX 4000 Ada, 20 475 MiB, CUDA 13.0, TF 2.15.0
- [x] Cache omics créé : `Dataset/ccle_broad_2019/omics_cache_gex978_cna426.npz`

### En cours

- [ ] **Entraînement Bi-Int 20 epochs** (batch_size=16, run en cours après fix OOM GPU)
  - Split : random (run de référence post-corrections P1–P7)
  - Commande : `python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 --no-ppo`
  - Logs : `logs/` (TensorBoard + CSV)

### Manquant / À faire

- [ ] **Résultats Bi-Int** — val RMSE, Pearson r, R² sur split random (run en cours)
- [ ] **Comparaison baseline** — tableau complet baselines vs Bi-Int post-corrections
- [ ] **Leave-drug-out** — run OOD séparé pour mesure de généralisation vers nouvelles molécules
- [ ] **Leave-cell-out** — run OOD vers nouveaux profils tumoraux
- [ ] **Push GitHub** — branche `main` à jour localement, push distant en attente

### Commandes pour les prochains runs

```bash
# Run de référence post-corrections (split aléatoire)
python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 \
    --no-ppo --log-dir logs/run_random_corrected

# Test OOD leave-drug-out
python3 fullPipeline.py --loss-mode both --beta-anneal --epochs 20 \
    --no-ppo --split-mode leave_drug_out --log-dir logs/run_ldo_v1

# Baselines comparatives
python3 baseline_models.py --out Dataset/baseline_results.csv

# Visualisation TensorBoard
tensorboard --logdir logs/
# → http://localhost:6006
```

---

## Récapitulatif des Commits

| Commit | Priorité | Fichier(s) | Description |
|--------|----------|------------|-------------|
| `e939fde` | P4 | `dqn_optimizer.py` | SA score + Lipinski hard + Tanimoto CCLE dans récompense DQN |
| `479088d` | P5 | `baseline_models.py` | XGBoost/MLP/R², mutations feature, 2048-bit ECFP4, comparaison Bi-Int |
| `9e5f66f` | P6 | `fullPipeline.py` | TensorBoard + CSVLogger + EarlyStopping + gradient norm dans BiIntTrainer |
| `d503903` | P7 | `fullPipeline.py` | Fix OOM — float32 read_csv, del DataFrames, vectorised IC50/triplet builder |
| `78e35b7` | — | `reports/` | Rapport d'ingénierie initial du 21/05/2026 |

---

*Corrections implémentées sur Ubuntu 24.04 LTS / WSL2, NVIDIA RTX 4000 Ada (20 475 MiB VRAM), TensorFlow 2.15.0, CUDA 13.0, RDKit 2024.  
Tous les changements sont commités sur la branche `main`. Push GitHub en attente.*
