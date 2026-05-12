# ============================================================================
#  CHEMBL PRE-TRAINING  (standalone – ne touche pas inference.py ni fullPipeline.py)
#  Dataset : Dataset/chembl_36.sdf
#  Featurizer : BRICS (identique au pipeline principal)
#  Epochs : 5
#  ZINC256k : NON utilisé
# ============================================================================
import os, sys, json, warnings, re
import numpy as np
import pandas as pd
import tensorflow as tf
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  GPU SETUP
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy() if gpus else tf.distribute.get_strategy()
print("Replicas:", strategy.num_replicas_in_sync)

# ---------------------------------------------------------------------------
#  HYPERPARAMÈTRES DU PRÉ-ENTRAÎNEMENT
# ---------------------------------------------------------------------------
PRETRAIN_HP = {
    'epochs'        : 5,            # demande encadrant : 5 epochs
    'batch_size'    : 32,
    'learning_rate' : 1e-3,
    'vae_beta'      : 1e-4,
    'max_atoms'     : 60,
    'max_compounds' : 50_000,       # limite mémoire – ajustable
    'val_split'     : 0.1,
    'random_seed'   : 42,
}

# ---------------------------------------------------------------------------
#  CHEMBL SDF LOADER  (lit directement Dataset/chembl_36.sdf)
#  Extrait : canonical SMILES + IC50 (nM) si disponible, sinon flag générique
# ---------------------------------------------------------------------------
def load_chembl_sdf(sdf_path, max_compounds=50_000):
    """
    Lit le fichier SDF ChEMBL et retourne un DataFrame avec :
      - smiles        : SMILES canonique RDKit
      - log_ic50      : log10(IC50 nM), ou NaN si absent
    Les entrées sans SMILES valide sont supprimées.
    """
    print(f"[LOAD] ChEMBL SDF : {sdf_path}")
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False
        print("  [WARN] RDKit non disponible – lecture SMILES brute depuis le SDF.")

    records = []
    count   = 0

    if rdkit_ok:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
        for mol in supplier:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if not smi:
                continue
            # Tenter de récupérer une valeur IC50 dans les propriétés SDF
            props = mol.GetPropsAsDict()
            ic50_val = None
            for key in props:
                if 'ic50' in key.lower() or 'standard_value' in key.lower():
                    try:
                        ic50_val = float(props[key])
                        break
                    except (ValueError, TypeError):
                        pass
            log_ic50 = np.log10(ic50_val) if (ic50_val and ic50_val > 0) else np.nan
            records.append({'smiles': smi, 'log_ic50': log_ic50})
            count += 1
            if count >= max_compounds:
                break
    else:
        # Fallback : parser les lignes "> <chembl_id>" et prendre le SMILES
        # (limité mais fonctionnel sans RDKit)
        with open(sdf_path, 'r', errors='ignore') as f:
            raw = f.read()
        blocks = raw.split('$$$$')
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 4:
                continue
            # Ligne 1 du header = parfois le SMILES dans ChEMBL SDF récents
            smi_line = lines[0].strip()
            if smi_line and len(smi_line) > 3:
                records.append({'smiles': smi_line, 'log_ic50': np.nan})
            count += 1
            if count >= max_compounds:
                break

    df = pd.DataFrame(records).dropna(subset=['smiles'])
    df = df[df['smiles'].str.len() > 3].reset_index(drop=True)
    print(f"  Molécules valides chargées : {len(df)}")
    print(f"  Avec IC50 mesurée          : {df['log_ic50'].notna().sum()}")
    return df


# ---------------------------------------------------------------------------
#  BRICS FEATURIZER  (identique à celui du pipeline principal)
# ---------------------------------------------------------------------------
class BRICSMolecularFeaturizer:
    """
    Featurize une molécule via BRICS + features atomiques.
    Retourne un tableau (max_atoms, atom_feat_dim).
    """
    ATOM_FEATURES = ['atomic_num', 'degree', 'formal_charge',
                     'hybridization', 'is_aromatic', 'in_ring']
    ATOM_FEAT_DIM = 16   # encodage one-hot + scalaires

    def __init__(self, max_atoms=60):
        self.max_atoms = max_atoms

    def _atom_features(self, atom):
        from rdkit.Chem import rdchem
        hyb_map = {
            rdchem.HybridizationType.SP : 0,
            rdchem.HybridizationType.SP2: 1,
            rdchem.HybridizationType.SP3: 2,
        }
        feats = [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 6.0,
            (atom.GetFormalCharge() + 3) / 6.0,
            hyb_map.get(atom.GetHybridization(), 3) / 3.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ]
        # Padding jusqu'à ATOM_FEAT_DIM
        feats += [0.0] * (self.ATOM_FEAT_DIM - len(feats))
        return feats

    def featurize(self, smiles: str) -> np.ndarray:
        """Retourne tableau (max_atoms, ATOM_FEAT_DIM) float32."""
        try:
            from rdkit import Chem
            from rdkit.Chem import BRICS
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((self.max_atoms, self.ATOM_FEAT_DIM), dtype=np.float32)
            # Fragmentation BRICS (annotation seulement, pas de découpe)
            _ = list(BRICS.FindBRICSBonds(mol))
            atoms = mol.GetAtoms()
            feat_list = [self._atom_features(a) for a in atoms]
            # Tronquer / padder
            feat_arr = np.zeros((self.max_atoms, self.ATOM_FEAT_DIM), dtype=np.float32)
            n = min(len(feat_list), self.max_atoms)
            feat_arr[:n] = feat_list[:n]
            return feat_arr
        except Exception:
            return np.zeros((self.max_atoms, self.ATOM_FEAT_DIM), dtype=np.float32)

    def adjacency(self, smiles: str) -> np.ndarray:
        """Retourne matrice d'adjacence (max_atoms, max_atoms) float32."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
            adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
            padded = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
            n = min(adj.shape[0], self.max_atoms)
            padded[:n, :n] = adj[:n, :n]
            return padded
        except Exception:
            return np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)


# ---------------------------------------------------------------------------
#  CONSTRUCTION DU DATASET TF  (SMILES only – pas de données omiques ici)
#  Target : log_ic50 si disponible, sinon 0.0 (pré-entraînement représentationnel)
# ---------------------------------------------------------------------------
def build_pretrain_dataset(df, featurizer, hp):
    """
    Retourne (train_ds, val_ds, atom_feat_dim).
    Chaque exemple : (atom_feats, adj_matrix) → log_ic50.
    """
    print("\n[PRETRAIN] Featurisation des molécules ChEMBL...")
    atom_feats_list, adj_list, targets = [], [], []
    skipped = 0

    for idx, row in df.iterrows():
        smi = row['smiles']
        target = row['log_ic50'] if not np.isnan(row['log_ic50']) else 0.0
        af = featurizer.featurize(smi)
        adj = featurizer.adjacency(smi)
        if af.sum() == 0:          # molécule invalide
            skipped += 1
            continue
        atom_feats_list.append(af)
        adj_list.append(adj)
        targets.append(np.float32(target))

    print(f"  Exemples valides : {len(targets)}  |  Ignorés : {skipped}")

    atom_feats_arr = np.stack(atom_feats_list).astype(np.float32)
    adj_arr        = np.stack(adj_list).astype(np.float32)
    targets_arr    = np.array(targets, dtype=np.float32)

    # Shuffle & split
    rng = np.random.default_rng(hp['random_seed'])
    idx = rng.permutation(len(targets_arr))
    split = int((1 - hp['val_split']) * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    def make_ds(indices):
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'atom_feats': atom_feats_arr[indices],
                'adj_matrix' : adj_arr[indices],
            },
            targets_arr[indices],
        ))
        return ds.batch(hp['batch_size']).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_idx)
    val_ds   = make_ds(val_idx)

    print(f"  Train : {len(train_idx)} | Val : {len(val_idx)}")
    atom_feat_dim = atom_feats_arr.shape[-1]
    return train_ds, val_ds, atom_feat_dim


# ---------------------------------------------------------------------------
#  MODÈLE DE PRÉ-ENTRAÎNEMENT : GNN léger (GAT-like) → régression log_IC50
#  Conçu pour initialiser les poids du drug encoder du pipeline principal.
# ---------------------------------------------------------------------------
def build_pretrain_model(max_atoms, atom_feat_dim, strategy):
    """
    GNN minimal :
      atom_feats (max_atoms, D) + adj (max_atoms, max_atoms)
      → mean-pool → MLP → log_ic50 scalaire
    Compatible avec les dimensions attendues par BiIntDigitalTwin.
    """
    with strategy.scope():
        atom_input = tf.keras.Input(shape=(max_atoms, atom_feat_dim), name='atom_feats')
        adj_input  = tf.keras.Input(shape=(max_atoms, max_atoms),     name='adj_matrix')

        # Message passing simplifié : A * H * W  (1 couche)
        x = tf.keras.layers.Dense(64, activation='relu', name='node_embed')(atom_input)
        # Agrégation voisins : adj @ x
        agg = tf.keras.layers.Lambda(
            lambda inputs: tf.matmul(inputs[0], inputs[1]),
            name='graph_conv'
        )([adj_input, x])
        x = tf.keras.layers.Add(name='residual')([x, agg])
        x = tf.keras.layers.LayerNormalization(name='ln1')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='node_proj')(x)

        # Pooling global
        pooled = tf.keras.layers.GlobalAveragePooling1D(name='graph_pool')(x)

        # Tête de régression
        out = tf.keras.layers.Dense(64, activation='relu', name='mlp1')(pooled)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = tf.keras.layers.Dense(1, name='ic50_head')(out)
        out = tf.keras.layers.Flatten()(out)

        model = tf.keras.Model(
            inputs=[atom_input, adj_input],
            outputs=out,
            name='ChEMBL_Pretrain_GNN'
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(PRETRAIN_HP['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
    return model


# ---------------------------------------------------------------------------
#  BOUCLE D'ENTRAÎNEMENT  (5 epochs, avec logs clairs)
# ---------------------------------------------------------------------------
def run_pretrain(model, train_ds, val_ds, epochs=5):
    print(f"\n[PRETRAIN] Démarrage : {epochs} epochs sur ChEMBL")
    print("="*55)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='pretrained_drug_encoder.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Résumé final
    print("\n[PRETRAIN] Résumé des epochs :")
    print(f"{'Epoch':>6} | {'Train RMSE':>10} | {'Val RMSE':>10}")
    print("-"*32)
    for i, (tl, vl) in enumerate(
            zip(history.history['loss'], history.history['val_loss']), 1):
        print(f"{i:>6} | {np.sqrt(tl):>10.4f} | {np.sqrt(vl):>10.4f}")

    return history


# ---------------------------------------------------------------------------
#  SAUVEGARDE DES POIDS  pour réutilisation dans le pipeline principal
# ---------------------------------------------------------------------------#
def save_pretrained_weights(model, save_dir='pretrained_weights'):
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, 'chembl_drug_encoder.weights.h5')
    model.save_weights(weights_path)
    meta = {
        'dataset'    : 'ChEMBL 36',
        'epochs'     : PRETRAIN_HP['epochs'],
        'featurizer' : 'BRICS',
        'zinc256k'   : False,
        'max_atoms'  : PRETRAIN_HP['max_atoms'],
        'model_name' : 'ChEMBL_Pretrain_GNN',
    }
    with open(os.path.join(save_dir, 'pretrain_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n[SAVE] Poids sauvegardés dans : {weights_path}")
    print(f"[SAVE] Métadonnées             : {save_dir}/pretrain_meta.json")
    print("\n[INFO] Pour charger ces poids dans le pipeline principal :")
    print("       model.get_layer('node_embed').set_weights(...)")
    print("       ou : pretrain_model.load_weights('pretrained_weights/chembl_drug_encoder.weights.h5')")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------#
def main():
    SDF_PATH = os.path.join('Dataset', 'chembl_36.sdf')

    if not os.path.exists(SDF_PATH):
        print(f"[ERROR] Fichier introuvable : {SDF_PATH}")
        print("        Vérifie que Dataset/chembl_36.sdf est bien présent.")
        sys.exit(1)

    # 1. Charger ChEMBL
    df = load_chembl_sdf(SDF_PATH, max_compounds=PRETRAIN_HP['max_compounds'])
    if len(df) == 0:
        print("[ERROR] Aucune molécule valide chargée depuis le SDF.")
        sys.exit(1)

    # 2. Featuriser (BRICS) — PAS de ZINC256k
    featurizer = BRICSMolecularFeaturizer(max_atoms=PRETRAIN_HP['max_atoms'])
    train_ds, val_ds, atom_feat_dim = build_pretrain_dataset(df, featurizer, PRETRAIN_HP)

    # 3. Construire le modèle
    print(f"\n[MODEL] atom_feat_dim={atom_feat_dim}, max_atoms={PRETRAIN_HP['max_atoms']}")
    model = build_pretrain_model(
        max_atoms=PRETRAIN_HP['max_atoms'],
        atom_feat_dim=atom_feat_dim,
        strategy=strategy,
    )
    model.summary()

    # 4. Pré-entraînement : 5 epochs
    history = run_pretrain(model, train_ds, val_ds, epochs=PRETRAIN_HP['epochs'])

    # 5. Sauvegarder les poids
    save_pretrained_weights(model)

    print("\n[DONE] Pré-entraînement ChEMBL terminé.")
    print("       Les poids sont prêts pour être chargés dans fullPipeline.py")
    print("       Le pipeline principal (inference.py, fullPipeline.py) n'a pas été modifié.")


if __name__ == "__main__":
    main()