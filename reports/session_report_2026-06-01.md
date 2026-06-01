# Rapport de session — 31 mai & 1er juin 2026
## Bi-Int : Restructuration du dépôt, vérification du pipeline, infrastructure de fiabilité & biomarqueurs thérapeutiques

**Projet :** Twin — Prédicteur multimodal de réponse aux drogues (CCLE)  
**Auteur :** Zied Dorsane  
**Superviseur :** M. Marouane  
**Commits couverts :** `8f270f8` → `d4a040d` (31 mai–1er juin 2026)  
**Matériel :** NVIDIA RTX 4000 Ada, 20 475 MiB VRAM — Ubuntu 24.04 LTS (WSL2)

---

## 1. Résumé exécutif

Ces deux journées ont couvert trois axes distincts. **Hier (31 mai)** : livraison des
livrables demandés par l'encadrant (scripts d'ablation LDO, validation chimique
rigoureuse, restructuration du dépôt pour lecteurs externes, rapport de session),
puis identification d'un diagnostic ncRNA — les données RNA-seq CCLE contiennent déjà
71 transcrits non-codants parmi les 978 gènes sélectionnés, dont H19 et GAS5.
**Aujourd'hui (1er juin)** : correction d'un problème d'infrastructure critique (le
modèle entraîné n'était jamais sauvegardé sur disque), correction d'un bug de
déterminisme VAE à l'inférence, et lancement d'un run LDO avec sauvegarde du
checkpoint — actuellement en cours. Les analyses de biomarqueurs et d'interprétabilité
sont planifiées dès que ce checkpoint sera disponible.

---

## 2. Travail du 31 mai — Livrables et restructuration

### 2.1 Vérification des fonctionnalités demandées par l'encadrant

| # | Fonctionnalité | État | Preuve |
|---|---------------|------|--------|
| 1 | Mapping SMILES via PubChem | ✅ Opérationnel | `Dataset/ccle_drug_smiles.csv` — 201/266 drogues mappées, 65 restantes sans SMILES |
| 2 | Chargement mutations (MAF) | ✅ Opérationnel | Parser MAF dans `src/fullPipeline.py` — 735 gènes top-mutés, données binaires |
| 3 | Splits Random / LDO / LCO | ✅ Opérationnel | Modes `random`, `leave_drug_out`, `leave_cell_out` — `--split-mode` CLI |
| 4 | Baselines RF / XGBoost / MLP + Ridge | ✅ Opérationnel | `src/baseline_models.py` + `Dataset/baseline_results_with_CI.csv` (12 lignes avec IC bootstrap) |
| 5 | BRICS-DQN | ✅ Opérationnel | `src/brics_dqn_optimizer.py` — 50 épisodes générés, `Dataset/brics_dqn_results.csv` |
| 6 | Terme « digital twin » prudent | ✅ Noté | README et rapports indiquent explicitement : prototype de recherche, pas un jumeau numérique clinique validé |

### 2.2 Scripts livrés (commits `8f270f8` à `ed9bf94`)

| Script | Fonction |
|--------|----------|
| `scripts/ldo_ablation.py` | Ablation systématique LDO : early stopping, régularisation, GNN freeze, data-size |
| `scripts/smiles_augmentation.py` | Enumération SMILES (augmentation de données) |
| `scripts/tanimoto_analysis.py` | Similarité Tanimoto GraphGA vs drogues CCLE |
| `scripts/bootstrap_ci.py` | IC bootstrap (n=1 000) sur toutes baselines × splits |
| `scripts/molecular_validation.py` | Validation chimique rigoureuse (SA, PAINS, Brenk, Lipinski, diversité) |

**Résultats ablation LDO :** les 4 configurations améliorées (early stopping, dropout, GNN freeze, 40k triplets) ont toutes échoué à l'exécution (code retour 2 — `fullPipeline.py` alors à la racine avant la réorganisation `src/`). Seule la configuration baseline a produit un résultat : r = 0.535, RMSE = 0.843 (epoch 1 uniquement). Les configurations améliorées seront relancées avec la nouvelle architecture `src/`.

### 2.3 Restructuration du dépôt (`940f19c`)

- Déplacement `fullPipeline.py`, `baseline_models.py`, `chembl_pretrain.py`, `brics_dqn_optimizer.py`, `graphga_biint_optimizer.py` → `src/`
- `README.md` réécrit (145 lignes, figures intégrées)
- `LICENSE` (MIT), `requirements.txt` (versions épinglées), `docs/DATA.md`, `docs/FIGURES_GUIDE.md` créés
- Notebook `notebooks/evaluation.ipynb` mis à jour (10 sections, fallbacks pour exécution sans logs)

### 2.4 Diagnostic ncRNA (`3b1d054`)

Aucun fichier miRNA/lncRNA dédié n'existe dans le jeu CCLE Broad 2019 disponible. Cependant, le RNA-seq (`data_mrna_seq_rpkm.txt`) contient un mélange de gènes codants et non-codants. Parmi les **978 gènes top-variance** sélectionnés par le modèle, **71 transcrits non-codants** sont présents, dont :

- **H19** (lncRNA oncogène — résistance à la chimiothérapie documentée)
- **GAS5** (lncRNA suppresseur de tumeur — corrélation avec réponse aux drogues)

**Conclusion :** l'analyse d'importance des biomarqueurs ncRNA est possible **sans téléchargement supplémentaire ni réentraînement**, en appliquant Integrated Gradients ou l'attention des blocs Bi-Int sur le modèle entraîné.

---

## 3. Travail du 1er juin — Infrastructure de fiabilité

### 3.1 Problème critique identifié : absence de sauvegarde du modèle

Le pipeline `src/fullPipeline.py` produisait uniquement des CSV de métriques
(`training_log.csv`) — le modèle Bi-Int IC50 entraîné n'était **jamais sauvegardé**.
Toute inférence post-entraînement, toute analyse d'interprétabilité et tout Integrated
Gradients nécessitaient de réentraîner depuis zéro.

**Correction apportée** dans `run_pipeline()` :

```python
# Priorité : save_weights (robuste pour modèles subclassés Keras)
w_path = os.path.join(log_dir, "biint_ic50_model.weights.h5")
model.save_weights(w_path)
# Tentative full save (peut échouer sur subclassed models)
model.save(os.path.join(log_dir, "biint_ic50_model.keras"))
# Snapshot HP pour reconstruction fidèle
json.dump(HP, open(os.path.join(log_dir, "hp_snapshot.json"), "w"))
```

Argument CLI ajouté : `--save-model` (activé par défaut) / `--no-save-model`.

### 3.2 Bug de déterminisme VAE corrigé

Lors des tests de rechargement, les prédictions post-`load_weights` différaient du
modèle original (écart maximal 1.24) malgré des poids identiques. Cause : la méthode
`reparameterize()` du VAE appelait `tf.random.normal()` même en `training=False`.

**Correction :**

```python
def reparameterize(self, mu, log_var, training=False):
    if not training:
        return mu  # inférence déterministe sur la moyenne
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps
```

**Validation :** `scripts/test_model_loading.py --self-test` → écart maximal après
save/load : **0.00e+00** (identité numérique exacte). Commit `d4a040d`.

### 3.3 Script de test de rechargement

`scripts/test_model_loading.py` — deux modes :

- `--self-test` : round-trip save/load sur poids aléatoires, vérifie l'identité numérique
- `--weights <path>` : charge un checkpoint existant, lance une forward pass, vérifie l'absence de NaN/Inf

Résultat du self-test : ✅ **PASSED** — 37,4 MB, 9 255 070 paramètres.

### 3.4 Run LDO avec checkpoint — EN COURS

```bash
python3 src/fullPipeline.py --mode pretrained --loss-mode cross_entropy \
  --split-mode leave_drug_out --epochs 15 --early-stopping 3 \
  --log-dir logs/ldo_checkpoint --no-ppo --save-model
```

**État actuel :** epoch 1 terminée (val_rmse = 1.017, pearson r = 0.210). Résultats
définitifs (best epoch, checkpoint final) : **à venir** — run en cours, durée estimée
1h30–2h.

---

## 4. Pourquoi les alertes de fiabilité et les biomarqueurs — rôle dans le modèle

### 4.1 Alertes de fiabilité (à implémenter)

En LDO, r = 0.316 : le modèle prédit avec assurance des valeurs IC50 sur des drogues
qu'il n'a jamais vues. C'est le cas le plus dangereux d'un modèle d'apprentissage
automatique — il génère un chiffre précis sans signaler son incertitude.

Trois niveaux d'alerte prévus :

| Alerte | Méthode | Signal |
|--------|---------|--------|
| Domaine d'applicabilité | Tanimoto max vs drogues d'entraînement | < 0.4 → hors domaine |
| Incertitude du modèle | MC Dropout (N=50 passes stochastiques) | Variance élevée → modèle hésite |
| Intervalle de confiance | Bootstrap sur les prédictions | Fourchette [IC_low, IC_high] |

Objectif : transformer un modèle qui « ment avec assurance » en un modèle « honnête
qui signale ses limites ». C'est ce qui rend le système crédible scientifiquement.

### 4.2 Biomarqueurs thérapeutiques (à implémenter)

Le modèle prédit l'IC50 mais n'explique pas pourquoi. L'analyse d'importance des gènes
permettra de répondre à deux questions :

1. **Interprétabilité :** quels gènes expliquent la prédiction pour un triplet
   (drogue, lignée) donné ?
2. **Validation biologique :** si le modèle pondère fortement EGFR sur des inhibiteurs
   d'EGFR, ou H19 sur des lignées résistantes à la chimio, cela démontre qu'il a appris
   de la biologie réelle — pas du bruit statistique.

Gènes cibles prioritaires parmi les 978 features :
- **Codants :** EGFR, KRAS, BRAF, TP53, PIK3CA, MYC (oncogènes/suppresseurs classiques)
- **Non-codants :** H19, GAS5 (lncRNA avec rôle établi dans la réponse aux drogues)

Méthode : Integrated Gradients sur le checkpoint LDO — nécessite le fichier de poids
actuellement en cours de génération.

---

## 5. Meilleures molécules générées (GraphGA)

Source : `graphga_top_candidates.csv` — résultats réels, 10 molécules Lipinski-compliant, 0 alerte PAINS.

| Rang | SMILES (abrégé) | QED | MW (Da) | LogP | Score composite |
|------|----------------|-----|---------|------|----------------|
| 1 | benzylaminopipérazine-cyclopropyl | **0.872** | 303.5 | 1.97 | **1.667** |
| 2 | tricarbamate-aniline | 0.784 | 310.3 | 1.02 | 1.656 |
| 3 | tricarbamate-acide benzoïque | 0.733 | 311.2 | 0.65 | 1.624 |
| 4 | O=C(COC…)Nc1ccccc1C(=O)O | 0.710 | 337.3 | 1.18 | 1.586 |
| 5 | acétamide-biphényle-COC=O | **0.849** | 269.3 | 2.99 | 1.578 |

Molécule n°1 complète : `CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1` — scaffold benzylaminopipérazine, MW 303 Da, LogP 1.97, QED 0.872, SA 0.794.

**Similarité Tanimoto vs CCLE :** max_tanimoto = 0.261 (vs UNC1215) — structurellement nouveau, aucune drogue CCLE > 0.4 de similarité.

> **AVERTISSEMENT :** les IC50 prédits associés à ces molécules sont issus d'un modèle
> dont la performance en généralisation hors distribution (LDO) est r = 0.316.
> Ces valeurs ne constituent pas une prédiction pharmacologique fiable. Une validation
> expérimentale (wet-lab) est indispensable avant toute conclusion clinique.

---

## 6. État d'avancement global

| Composant | État |
|-----------|------|
| Architecture Bi-Int (GNN + Quaternion-VAE + Bi-Int blocks) | ✅ Fonctionnel |
| Entraînement QSAR (random split) | ✅ r = 0.811 [0.736, 0.886] — epoch 4 |
| Entraînement QSAR (LDO) | ✅ r = 0.316 [0.241, 0.391] — epoch 2 |
| Entraînement QSAR (LCO) | 🔄 Run partiel (6 epochs, best r = 0.766 epoch 4) — non finalisé |
| Baselines avec IC bootstrap | ✅ 12 modèles × 3 splits — `Dataset/baseline_results_with_CI.csv` |
| Validation chimique GraphGA + BRICS-DQN | ✅ `Dataset/molecular_validation_report.csv` (60 molécules) |
| Ablation LDO (early stopping, régularisation) | ⚠️ Échoué lors de la réorganisation `src/` — à relancer |
| Sauvegarde checkpoint modèle | ✅ Implémenté + testé (self-test 0.00e+00) |
| Run LDO avec checkpoint | 🔄 En cours — epoch 1/≤15 |
| Inférence déterministe (fix VAE) | ✅ `reparameterize` retourne `mu` en `training=False` |
| Analyse biomarqueurs (Integrated Gradients) | ⏳ Attend le checkpoint LDO |
| Alertes de fiabilité (Tanimoto + MC Dropout) | ⏳ À implémenter |

---

## 7. Prochaines étapes

1. **Immédiat :** attendre la fin du run LDO checkpoint (~1h) → vérifier
   `logs/ldo_checkpoint/biint_ic50_model.weights.h5` (~37 MB)
2. **Analyse biomarqueurs :** Integrated Gradients sur le checkpoint → importance
   EGFR, KRAS, H19, GAS5 par drogue/lignée
3. **Alertes de fiabilité :** MC Dropout (N=50) + domaine d'applicabilité Tanimoto
4. **Relancer ablation LDO** depuis `src/` — early stopping patience=3, dropout 0.3,
   GNN freeze 3 epochs, 40k triplets
5. **Analyse LCO** : compléter le run LCO interrompu à epoch 6 pour la grille de
   comparaison complète

---

## 8. Limitations honnêtes

- **Généralisation LDO limitée :** r = 0.316 — le modèle actuel généralise mal sur des
  drogues non vues à l'entraînement. Les baselines (XGBoost r = 0.367) font
  légèrement mieux sur ce split.
- **Résultats biomarqueurs non disponibles :** l'analyse Integrated Gradients dépend
  du checkpoint LDO en cours de génération.
- **Importance ≠ causalité :** un poids élevé d'un gène dans le modèle indique une
  corrélation statistique, pas un mécanisme biologique.
- **65/266 drogues sans SMILES :** exclues de toutes les analyses moléculaires.
- **Ablation LDO incomplète :** 4/5 configurations ont échoué lors de la réorganisation
  `src/` — résultats partiels, à reconduire.
- **Run LCO non finalisé :** 6 epochs disponibles (best r = 0.766 epoch 4), run
  interrompu — pas encore de résultat LCO définitif pour Bi-Int.
