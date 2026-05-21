# ============================================================================
#  CHEMBL PRE-TRAINING (standalone) — VERSION CORRIGÉE
#  Dataset    : Dataset/chembl_36.sdf
#  Featurizer : BRICS-style atom features (identique au pipeline principal)
#  Cible      : descripteurs RDKit multi-tâches (LogP, TPSA, MW, HBD, HBA, QED, NumRings, NumAromaticRings)
#               → pretraining auto-supervisé, pas besoin de labels IC50
#  Epochs     : 5
#
#  Pourquoi ce changement ?
#    Le SDF structurel de ChEMBL ne contient PAS de valeurs IC50 (elles sont
#    dans la table 'activities' de la DB). L'ancien script entraînait donc le
#    modèle à prédire la constante 0, ce qui produisait des poids inutiles.
#    Ici on prédit 8 descripteurs calculés par RDKit → l'encodeur apprend
#    une représentation chimique réellement transférable.
# ============================================================================
import os, sys, json, warnings
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
#  HYPERPARAMÈTRES
# ---------------------------------------------------------------------------
PRETRAIN_HP = {
    'epochs'        : 10,
    'batch_size'    : 64,
    'learning_rate' : 1e-3,
    'max_atoms'     : 60,
    'max_compounds' : None,       # None = toutes les 2.8M molécules via streaming HDF5
    'chunk_size'    : 10_000,     # molécules traitées par chunk SDF (RAM ~500 MB max)
    'val_split'     : 0.1,
    'random_seed'   : 42,
    'hdf5_cache'    : 'Dataset/chembl_features.h5',   # cache featurisé sur disque
}

# Liste des descripteurs prédits (cibles multi-tâches)
DESCRIPTOR_NAMES = [
    'MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors',
    'QED', 'NumRings', 'NumAromaticRings'
]
N_TARGETS = len(DESCRIPTOR_NAMES)


# ---------------------------------------------------------------------------
#  CHEMBL SDF LOADER — extrait SMILES + calcule descripteurs RDKit
# ---------------------------------------------------------------------------
def compute_descriptors(mol):
    """Calcule les 8 descripteurs cibles pour une molécule RDKit."""
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    try:
        return np.array([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolWt(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            QED.qed(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
        ], dtype=np.float32)
    except Exception:
        return None


def stream_chembl_to_hdf5(sdf_path, hdf5_path, featurizer, max_atoms, chunk_size=10_000):
    """
    Lit le SDF ChEMBL par chunks de chunk_size molécules, featurise chaque chunk,
    et écrit immédiatement dans un fichier HDF5 (append mode).
    RAM peak = chunk_size × (max_atoms×16 + max_atoms×max_atoms + 8) float32
             ≈ 10k × (60×16 + 60×60 + 8) × 4 bytes ≈ 170 MB — stable quelle que soit la taille du SDF.

    Retourne (n_total, mean, std) où mean/std sont calculés en deux passes sur le HDF5.
    """
    import h5py
    from rdkit import Chem

    print(f"[STREAM] ChEMBL SDF → HDF5 : {sdf_path}")
    print(f"         Cache              : {hdf5_path}")
    print(f"         Chunk size         : {chunk_size:,} molécules")

    # Si le cache existe déjà, recalculer mean/std et retourner directement
    if os.path.exists(hdf5_path):
        print(f"[STREAM] Cache HDF5 existant détecté — recalcul mean/std...")
        with h5py.File(hdf5_path, 'r') as h5:
            n = h5['targets'].shape[0]
            # Calcul mean/std en une seule passe (tout le dataset tient en float32)
            targets = h5['targets'][:]
        mean, std = targets.mean(axis=0), targets.std(axis=0) + 1e-6
        print(f"  {n:,} molécules déjà featurisées. Ré-utilisation du cache.")
        return n, mean, std

    os.makedirs(os.path.dirname(hdf5_path) if os.path.dirname(hdf5_path) else '.', exist_ok=True)
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)

    total_valid = 0
    total_seen  = 0
    skipped     = 0

    # Buffers de chunk
    chunk_af, chunk_adj, chunk_desc = [], [], []

    def _flush_chunk(h5_handle):
        nonlocal total_valid
        if not chunk_af:
            return
        af_arr   = np.stack(chunk_af).astype(np.float32)
        adj_arr  = np.stack(chunk_adj).astype(np.float32)
        desc_arr = np.stack(chunk_desc).astype(np.float32)
        n = af_arr.shape[0]
        if 'atom_feats' not in h5_handle:
            h5_handle.create_dataset('atom_feats', data=af_arr,
                maxshape=(None, max_atoms, af_arr.shape[-1]), chunks=(256, max_atoms, af_arr.shape[-1]))
            h5_handle.create_dataset('adj_matrix', data=adj_arr,
                maxshape=(None, max_atoms, max_atoms), chunks=(256, max_atoms, max_atoms))
            h5_handle.create_dataset('targets', data=desc_arr,
                maxshape=(None, desc_arr.shape[-1]), chunks=(256, desc_arr.shape[-1]))
        else:
            for key, arr in [('atom_feats', af_arr), ('adj_matrix', adj_arr), ('targets', desc_arr)]:
                old = h5_handle[key].shape[0]
                h5_handle[key].resize(old + n, axis=0)
                h5_handle[key][old:old + n] = arr
        total_valid += n
        chunk_af.clear(); chunk_adj.clear(); chunk_desc.clear()

    with h5py.File(hdf5_path, 'w') as h5:
        for mol in supplier:
            total_seen += 1
            if total_seen % 100_000 == 0:
                print(f"  ... {total_seen:,} scannées | {total_valid:,} valides | {skipped:,} ignorées")

            if mol is None or mol.GetNumAtoms() > max_atoms:
                skipped += 1
                continue
            smi = Chem.MolToSmiles(mol)
            if not smi or len(smi) < 3:
                skipped += 1
                continue
            desc = compute_descriptors(mol)
            if desc is None or not np.all(np.isfinite(desc)):
                skipped += 1
                continue
            af, adj = featurizer.featurize(smi)
            if af is None:
                skipped += 1
                continue

            chunk_af.append(af); chunk_adj.append(adj); chunk_desc.append(desc)
            if len(chunk_af) >= chunk_size:
                _flush_chunk(h5)

        _flush_chunk(h5)  # dernier chunk partiel

        print(f"\n  Total scanné  : {total_seen:,}")
        print(f"  Valides       : {total_valid:,}")
        print(f"  Ignorées      : {skipped:,}")
        print(f"  HDF5 écrit    : {hdf5_path}  ({os.path.getsize(hdf5_path)/1e9:.2f} GB)")

        targets = h5['targets'][:]

    mean, std = targets.mean(axis=0), targets.std(axis=0) + 1e-6
    return total_valid, mean, std


def load_chembl_sdf(sdf_path, max_compounds=None):
    """Conservé pour compatibilité — utilisé uniquement si max_compounds est fixé (tests rapides)."""
    from rdkit import Chem
    print(f"[LOAD] ChEMBL SDF (mode limité) : max={max_compounds}")
    records, skipped, total_seen = [], 0, 0
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    for mol in supplier:
        total_seen += 1
        if mol is None: skipped += 1; continue
        smi = Chem.MolToSmiles(mol)
        if not smi or len(smi) < 3: skipped += 1; continue
        if mol.GetNumAtoms() > PRETRAIN_HP['max_atoms']: skipped += 1; continue
        desc = compute_descriptors(mol)
        if desc is None or not np.all(np.isfinite(desc)): skipped += 1; continue
        records.append({'smiles': smi, 'descriptors': desc})
        if max_compounds and len(records) >= max_compounds:
            break
    df = pd.DataFrame(records)
    print(f"  Valides : {len(df):,} | Ignorées : {skipped:,}")
    return df


# ---------------------------------------------------------------------------
#  NORMALISATION DES CIBLES  (essentiel pour multi-tâches)
#  Les descripteurs ont des échelles très différentes :
#    MolWt ~ 0-1000, TPSA ~ 0-200, QED ~ 0-1, NumRings ~ 0-10
#  Sans normalisation, MolWt dominerait la loss et écraserait les autres.
# ---------------------------------------------------------------------------
def fit_target_scaler(targets_arr):
    """Calcule mean/std par dimension."""
    mean = targets_arr.mean(axis=0)
    std  = targets_arr.std(axis=0) + 1e-6
    return mean, std


def normalize_targets(targets_arr, mean, std):
    return (targets_arr - mean) / std


# ---------------------------------------------------------------------------
#  BRICS FEATURIZER  (inchangé mais corrigé : adjacence normalisée)
# ---------------------------------------------------------------------------
class BRICSMolecularFeaturizer:
    ATOM_FEAT_DIM = 16

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
            atom.GetTotalNumHs() / 4.0,
            float(atom.GetChiralTag() != 0),
        ]
        feats += [0.0] * (self.ATOM_FEAT_DIM - len(feats))
        return feats

    def featurize(self, smiles: str):
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            atoms = mol.GetAtoms()
            n = min(len(atoms), self.max_atoms)
            feat_arr = np.zeros((self.max_atoms, self.ATOM_FEAT_DIM), dtype=np.float32)
            for i in range(n):
                feat_arr[i] = self._atom_features(atoms[i])

            # Adjacence + self-loops + normalisation symétrique D^-1/2 A D^-1/2
            adj_raw = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
            adj_raw = adj_raw[:n, :n]
            adj_sl = adj_raw + np.eye(n, dtype=np.float32)   # self-loops
            deg = adj_sl.sum(axis=1)
            d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-6)
            adj_norm = (d_inv_sqrt[:, None] * adj_sl) * d_inv_sqrt[None, :]

            adj_padded = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
            adj_padded[:n, :n] = adj_norm
            return feat_arr, adj_padded
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
#  CONSTRUCTION DU DATASET TF
# ---------------------------------------------------------------------------
def build_pretrain_dataset(df, featurizer, hp):
    """
    Featurise les molécules par chunks pour éviter une explosion mémoire
    sur 2M+ molécules, puis construit un tf.data.Dataset avec cache disque.
    """
    print(f"\n[PRETRAIN] Featurisation de {len(df):,} molécules ChEMBL...")
    chunk_size = hp.get('chunk_size', 100_000)

    atom_feats_list, adj_list, targets = [], [], []
    skipped = 0

    for chunk_start in range(0, len(df), chunk_size):
        chunk = df.iloc[chunk_start:chunk_start + chunk_size]
        for _, row in chunk.iterrows():
            af, adj = featurizer.featurize(row['smiles'])
            if af is None or af.sum() == 0:
                skipped += 1
                continue
            atom_feats_list.append(af)
            adj_list.append(adj)
            targets.append(row['descriptors'])
        done = min(chunk_start + chunk_size, len(df))
        print(f"  Chunk {done:,}/{len(df):,} — valides jusqu'ici : {len(targets):,}")

    print(f"  Exemples valides : {len(targets):,}  |  Ignorés : {skipped:,}")

    atom_feats_arr = np.stack(atom_feats_list).astype(np.float32)
    adj_arr        = np.stack(adj_list).astype(np.float32)
    targets_arr    = np.stack(targets).astype(np.float32)
    del atom_feats_list, adj_list, targets  # libérer la RAM

    # Normaliser les cibles (fit sur tout le set avant split)
    mean, std = fit_target_scaler(targets_arr)
    targets_norm = normalize_targets(targets_arr, mean, std)
    print(f"  Cibles (avant norm)  mean={mean.round(2)}")
    print(f"  Cibles (avant norm)  std ={std.round(2)}")

    # Shuffle & split
    rng = np.random.default_rng(hp['random_seed'])
    idx = rng.permutation(len(targets_arr))
    split = int((1 - hp['val_split']) * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    def make_ds(indices, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'atom_feats': atom_feats_arr[indices],
                'adj_matrix': adj_arr[indices],
            },
            targets_norm[indices],
        ))
        if shuffle:
            ds = ds.shuffle(min(len(indices), 50_000), seed=hp['random_seed'])
        return ds.batch(hp['batch_size']).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_idx, shuffle=True)
    val_ds   = make_ds(val_idx)

    print(f"  Train : {len(train_idx):,} | Val : {len(val_idx):,}")
    atom_feat_dim = atom_feats_arr.shape[-1]
    return train_ds, val_ds, atom_feat_dim, mean, std


# ---------------------------------------------------------------------------
#  MODÈLE : GNN multi-couches → tête multi-tâches (8 descripteurs)
# ---------------------------------------------------------------------------
def build_pretrain_model(max_atoms, atom_feat_dim, n_targets, strategy):
    with strategy.scope():
        atom_input = tf.keras.Input(shape=(max_atoms, atom_feat_dim), name='atom_feats')
        adj_input  = tf.keras.Input(shape=(max_atoms, max_atoms),     name='adj_matrix')

        # Couche 1 : embedding atomique
        h = tf.keras.layers.Dense(64, activation='relu', name='node_embed')(atom_input)

        # Couche 2 : message passing (adj normalisée @ h)
        agg1 = tf.keras.layers.Lambda(
            lambda inp: tf.matmul(inp[0], inp[1]),
            name='graph_conv_1'
        )([adj_input, h])
        h = tf.keras.layers.Dense(64, activation='relu', name='gcn_proj_1')(agg1)
        h = tf.keras.layers.LayerNormalization(name='ln1')(h)

        # Couche 3 : 2e passe de message passing
        agg2 = tf.keras.layers.Lambda(
            lambda inp: tf.matmul(inp[0], inp[1]),
            name='graph_conv_2'
        )([adj_input, h])
        h = tf.keras.layers.Dense(128, activation='relu', name='node_proj')(agg2)
        h = tf.keras.layers.LayerNormalization(name='ln2')(h)

        # Pooling (mean + max concaténés)
        mean_pool = tf.keras.layers.GlobalAveragePooling1D(name='mean_pool')(h)
        max_pool  = tf.keras.layers.GlobalMaxPooling1D(name='max_pool')(h)
        pooled = tf.keras.layers.Concatenate(name='graph_pool')([mean_pool, max_pool])

        # Tête multi-tâches
        x = tf.keras.layers.Dense(128, activation='relu', name='mlp1')(pooled)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation='relu', name='mlp2')(x)
        out = tf.keras.layers.Dense(n_targets, name='descriptor_head')(x)

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
#  ENTRAÎNEMENT
# ---------------------------------------------------------------------------
def run_pretrain(model, train_ds, val_ds, epochs=5):
    print(f"\n[PRETRAIN] Démarrage : {epochs} epochs sur ChEMBL")
    print("=" * 55)

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

    print("\n[PRETRAIN] Résumé des epochs (sur cibles normalisées) :")
    print(f"{'Epoch':>6} | {'Train RMSE':>10} | {'Val RMSE':>10} | {'Val MAE':>10}")
    print("-" * 50)
    for i, (tl, vl, vm) in enumerate(zip(
            history.history['loss'],
            history.history['val_loss'],
            history.history['val_mae']), 1):
        print(f"{i:>6} | {np.sqrt(tl):>10.4f} | {np.sqrt(vl):>10.4f} | {vm:>10.4f}")

    # Sanity check : val_loss devrait être nettement < 1.0 (variance des cibles normalisées = 1)
    final_val = history.history['val_loss'][-1]
    if final_val > 0.8:
        print(f"\n[WARN] val_loss final = {final_val:.4f} : le modèle apprend peu.")
    elif final_val < 0.01:
        print(f"\n[WARN] val_loss final = {final_val:.4f} : suspicieusement bas — vérifier les cibles.")
    else:
        print(f"\n[OK] val_loss final = {final_val:.4f} : apprentissage cohérent.")

    return history


# ---------------------------------------------------------------------------
#  SAUVEGARDE
# ---------------------------------------------------------------------------
def save_pretrained_weights(model, target_mean, target_std, save_dir='pretrained_weights'):
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, 'chembl_drug_encoder.weights.h5')
    model.save_weights(weights_path)
    meta = {
        'dataset'         : 'ChEMBL 36',
        'epochs'          : PRETRAIN_HP['epochs'],
        'featurizer'      : 'BRICS-atomic',
        'zinc256k'        : False,
        'max_atoms'       : PRETRAIN_HP['max_atoms'],
        'model_name'      : 'ChEMBL_Pretrain_GNN',
        'objective'       : 'multi-task RDKit descriptor regression',
        'descriptors'     : DESCRIPTOR_NAMES,
        'target_mean'     : target_mean.tolist(),
        'target_std'      : target_std.tolist(),
        'transfer_layers' : ['node_embed', 'gcn_proj_1', 'ln1',
                             'node_proj', 'ln2'],
    }
    with open(os.path.join(save_dir, 'pretrain_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n[SAVE] Poids sauvegardés dans : {weights_path}")
    print(f"[SAVE] Métadonnées             : {save_dir}/pretrain_meta.json")
    print("\n[INFO] Pour réutiliser dans fullPipeline.py :")
    print("       pretrain_model.load_weights('pretrained_weights/chembl_drug_encoder.weights.h5')")
    print("       Puis copier les poids des couches : node_embed, gcn_proj_1, ln1, node_proj, ln2")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
def main():
    SDF_PATH = os.path.join('Dataset', 'chembl_36.sdf')

    if not os.path.exists(SDF_PATH):
        print(f"[ERROR] Fichier introuvable : {SDF_PATH}")
        sys.exit(1)

    # 1. Charger ChEMBL + calculer descripteurs (toutes les molécules si max_compounds=None)
    df = load_chembl_sdf(SDF_PATH, max_compounds=PRETRAIN_HP['max_compounds'])
    if len(df) == 0:
        print("[ERROR] Aucune molécule valide chargée.")
        sys.exit(1)

    # 2. Featuriser
    featurizer = BRICSMolecularFeaturizer(max_atoms=PRETRAIN_HP['max_atoms'])
    train_ds, val_ds, atom_feat_dim, t_mean, t_std = build_pretrain_dataset(
        df, featurizer, PRETRAIN_HP)

    # 3. Modèle
    print(f"\n[MODEL] atom_feat_dim={atom_feat_dim}, max_atoms={PRETRAIN_HP['max_atoms']}, n_targets={N_TARGETS}")
    model = build_pretrain_model(
        max_atoms=PRETRAIN_HP['max_atoms'],
        atom_feat_dim=atom_feat_dim,
        n_targets=N_TARGETS,
        strategy=strategy,
    )
    model.summary()

    # 4. Pretraining
    history = run_pretrain(model, train_ds, val_ds, epochs=PRETRAIN_HP['epochs'])

    # 5. Sauvegarde
    save_pretrained_weights(model, t_mean, t_std)

    print("\n[DONE] Pré-entraînement ChEMBL terminé.")
    print("       Les poids sont prêts pour fullPipeline.py")


if __name__ == "__main__":
    main()