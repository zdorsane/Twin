# ============================================================================
#  CCLE PIPELINE TRAINER & DIGITAL TWIN INFERENCE  (multi-GPU, memory‑optimised)
# ============================================================================
import os, sys, json, warnings, re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")



# ---------------------------------------------------------------------------
#  GPU SETUP (MirroredStrategy, memory growth)
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)
if len(gpus) == 0:
    raise RuntimeError("No GPU detected")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
print("Replicas:", strategy.num_replicas_in_sync)

# ---------------------------------------------------------------------------
#  HELPER: feature selection by variance (to reduce model size)
# ---------------------------------------------------------------------------
def select_top_k_features(df, k):
    """Keep the k most variable features."""
    if df.shape[1] <= k:
        return df
    var = df.var(axis=0)
    top = var.nlargest(k).index
    return df[top]

# ---------------------------------------------------------------------------
#  CELL LINE / DRUG NAME NORMALISATION
# ---------------------------------------------------------------------------
TISSUE_SUFFIXES = [
    '_PROSTATE','_STOMACH','_URINARY_TRACT','_LUNG','_BREAST',
    '_SKIN','_OVARY','_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
    '_LARGE_INTESTINE','_OESOPHAGUS','_ENDOMETRIUM','_THYROID',
    '_PANCREAS','_BONE','_SOFT_TISSUE','_LIVER','_KIDNEY',
    '_BRAIN','_CENTRAL_NERVOUS_SYSTEM','_AUTONOMIC_GANGLIA',
    '_BILIARY_TRACT','_PLEURA','_PROSTATE','_UPPER_AERODIGESTIVE_TRACT',
    '_SALIVARY_GLAND','_TESTIS','_VULVA','_CERVIX','_FIBROBLAST',
    '_FIBROBLASTS','_LYMPHOBLASTOID','_MESOTHELIUM','_EYE'
]

def normalize_cell(name):
    name = str(name).upper().strip()
    for t in TISSUE_SUFFIXES:
        if name.endswith(t):
            name = name[:-len(t)]
            break
    return name

def normalize_drug_name(name):
    name = str(name).strip()
    name = re.sub(r'-\d+$', '', name)
    name = re.sub(r'-$', '', name)
    name = re.sub(r'\d+\s*uM$', '', name, flags=re.IGNORECASE)
    return name.strip()

def aggregate_duplicates(df, agg):
    if df.index.duplicated().any():
        print(f"  Aggregating {df.index.duplicated().sum()} duplicates via {agg}.")
        df = df.groupby(df.index).agg(agg)
    return df

# ---------------------------------------------------------------------------
#  DATA LOADERS (all genes, then feature selection)
# ---------------------------------------------------------------------------
def load_expression(filepath):
    print(f"[LOAD] Expression: {filepath}")
    df = pd.read_csv(filepath, sep="\t", comment='#', low_memory=False)
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if df.max().max() > 30:
        df = np.log2(df + 1)
    df = df.T
    df.index = df.index.map(normalize_cell)
    df = aggregate_duplicates(df, 'mean')
    df.index.name = "cell_line"
    print(f"  Raw shape: {df.shape}")
    return df

def load_mutations_maf(filepath):
    print(f"[LOAD] Mutations (MAF): {filepath}")
    maf = pd.read_csv(filepath, sep="\t", comment='#', low_memory=False)
    maf = maf[['Tumor_Sample_Barcode', 'Hugo_Symbol']].dropna()
    maf['Tumor_Sample_Barcode'] = maf['Tumor_Sample_Barcode'].apply(normalize_cell)
    maf['present'] = 1
    mat = maf.pivot_table(index='Tumor_Sample_Barcode',
                          columns='Hugo_Symbol',
                          values='present',
                          aggfunc='max', fill_value=0).astype(np.float32)
    mat.index.name = "cell_line"
    mat = aggregate_duplicates(mat, 'max')
    print(f"  Raw shape: {mat.shape}")
    return mat

def load_cna(filepath):
    print(f"[LOAD] CNA: {filepath}")
    df = pd.read_csv(filepath, sep="\t", comment='#', low_memory=False)
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    df = df.T
    df.index = df.index.map(normalize_cell)
    df = aggregate_duplicates(df, 'mean')
    df.index.name = "cell_line"
    print(f"  Raw shape: {df.shape}")
    return df

def load_ic50(filepath):
    print(f"[LOAD] IC50: {filepath}")
    df = pd.read_csv(filepath, sep="\t", comment='#', low_memory=False)
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.T
    df.index = df.index.map(normalize_cell)
    df = aggregate_duplicates(df, 'mean')
    df.index.name = "cell_line"
    return df

# ---------------------------------------------------------------------------
#  SMILES MAP (with cache cleaning)
# ---------------------------------------------------------------------------
def build_drug_smiles_map(drug_names, cache_file="CCLE_drug_smiles.csv"):
    if os.path.exists(cache_file):
        print(f"[SMILES] Loading cached SMILES from {cache_file}")
        smi_df = pd.read_csv(cache_file, index_col=0)
        def clean_val(val):
            if isinstance(val, str) and len(val) > 3:
                return val
            return None
        return {d: clean_val(smi_df.at[d, smi_df.columns[0]]) if d in smi_df.index else None
                for d in drug_names}
    try:
        import pubchempy as pcp
        print("[SMILES] Querying PubChem...")
        mapping = {}
        for drug in drug_names:
            try:
                clean = normalize_drug_name(drug)
                comps = pcp.get_compounds(clean, 'name')
                mapping[drug] = comps[0].isomeric_smiles if comps else None
            except:
                mapping[drug] = None
        pd.Series(mapping).to_csv(cache_file)
        return mapping
    except ImportError:
        print("[SMILES] pubchempy not installed – tiny fallback.")
        fallback = {
            "17-AAG": "CC(C)C(=O)NC1=CC(=O)C(=C(C1)OC)NC(=O)C2=CC=CC=C2",
            "AEW541": "COC1=CC=C(C=C1)NC2=NC=CC(=N2)N3CCN(CC3)C4=CC=CC=C4",
            "AZD6244": "CN1C=NC2=C1C=C(C=C2)NC3=NC=CC(=N3)C4=CC=CC=C4",
            "Erlotinib": "COC1=CC2=C(C=C1OC)NC(=O)C2=CC3=CC=CC=N3",
            "Imatinib": "CC1=CC=C(C=C1)NC2=NC=CC(=N2)N3CCN(CC3)C4=CC=CC=C4",
            "Lapatinib": "COC1=CC2=C(C=C1OC)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl",
            "Nilotinib": "CC1=CC=C(C=C1)NC(=O)C2=NC(=CC=C2)C3=CN=CC=C3",
            "Paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1O)O)OC(=O)C)OC(=O)C5=CC=CC=C5)O)C)OC(=O)C",
            "Sorafenib": "CNC(=O)C1=CC=CC=C1OC2=CC=C(C=C2)NC3=NC=CC(=N3)C4=CC=CC=C4",
            "Vemurafenib": "C1=CC(=CC=C1CNC(=O)C2=CC=C(C=C2)F)C3=NC=C(C=N3)C4=CC=CC=C4",
        }
        return {d: fallback.get(d.upper().strip()) for d in drug_names}

# ---------------------------------------------------------------------------
#  STREAMING DATASET BUILDER (memory friendly)
# ---------------------------------------------------------------------------
def build_ccle_data(gex_df, mut_df, cna_df, ic50_df, smiles_map,
                    batch_size=32, max_genes=6000):
    """Returns train_ds, val_ds, info. Also applies variance‑based feature trimming."""
    common = sorted(set(gex_df.index) & set(mut_df.index) & set(cna_df.index) & set(ic50_df.index))
    print(f"\nMatched cell lines: {len(common)}")

    # Trim to most variable genes to fit GPU memory
    gex_df = select_top_k_features(gex_df, max_genes)
    mut_df = select_top_k_features(mut_df, max_genes)
    cna_df = select_top_k_features(cna_df, max_genes)
    print(f"  Expression genes kept: {gex_df.shape[1]}")
    print(f"  Mutation genes kept:  {mut_df.shape[1]}")
    print(f"  CNA genes kept:       {cna_df.shape[1]}")

    # Align (already trimmed)
    gex_arr = gex_df.loc[common].values.astype(np.float32)
    mut_arr = mut_df.loc[common].values.astype(np.float32)
    cna_arr = cna_df.loc[common].values.astype(np.float32)
    ic50_mat = ic50_df.loc[common]

    # Build drug features
    featurizer = BRICSMolecularFeaturizer()
    drug_atom_dict, drug_adj_dict = {}, {}
    valid_drugs = []
    for drug in ic50_mat.columns:
        smi = smiles_map.get(drug)
        if not smi:
            continue
        try:
            atom_feat = featurizer.featurize(smi)
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                max_at = BASE_HP['max_atoms']
                if mol:
                    adj_mat = Chem.GetAdjacencyMatrix(mol)
                    padded = np.zeros((max_at, max_at), dtype=np.float32)
                    n = min(adj_mat.shape[0], max_at)
                    padded[:n, :n] = adj_mat[:n, :n]
                    adj = padded
                else:
                    adj = np.ones((max_at, max_at), dtype=np.float32)
            except:
                adj = np.ones((BASE_HP['max_atoms'], BASE_HP['max_atoms']), dtype=np.float32)
            drug_atom_dict[drug] = atom_feat.astype(np.float32)
            drug_adj_dict[drug] = adj.astype(np.float32)
            valid_drugs.append(drug)
        except Exception as e:
            print(f"  Skipping drug {drug}: {e}")

    print(f"  Valid drugs: {len(valid_drugs)}")

    # Pair list
    pairs = []
    for i, cell in enumerate(common):
        for drug in valid_drugs:
            val = ic50_mat.at[cell, drug]
            if not pd.isna(val):
                pairs.append((i, drug))
    print(f"  Total pairs: {len(pairs)}")

    # Shuffle & split
    np.random.seed(42)
    np.random.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]

    # Generator
    def gen(p_list):
        for i, drug in p_list:
            yield (drug_atom_dict[drug],
                   drug_adj_dict[drug],
                   gex_arr[i],
                   mut_arr[i],
                   cna_arr[i],
                   ic50_mat.at[common[i], drug])

    atom_feat_dim = next(iter(drug_atom_dict.values())).shape[-1]
    sig = (
        tf.TensorSpec([BASE_HP['max_atoms'], atom_feat_dim], tf.float32),
        tf.TensorSpec([BASE_HP['max_atoms'], BASE_HP['max_atoms']], tf.float32),
        tf.TensorSpec([gex_arr.shape[1]], tf.float32),
        tf.TensorSpec([mut_arr.shape[1]], tf.float32),
        tf.TensorSpec([cna_arr.shape[1]], tf.float32),
        tf.TensorSpec([], tf.float32),
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: gen(train_pairs), output_signature=sig
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: gen(val_pairs), output_signature=sig
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    info = {
        'gex_dim': gex_arr.shape[1],
        'mut_dim': mut_arr.shape[1],
        'cnv_dim': cna_arr.shape[1],
        'valid_drugs': valid_drugs,
        'common_cells': common,
        'num_train': len(train_pairs),
        'num_val': len(val_pairs),
    }
    return train_ds, val_ds, info

# ---------------------------------------------------------------------------
#  DISTRIBUTED TRAINING LOOP
# ---------------------------------------------------------------------------
class DistributedTrainer:
    def __init__(self, model, hp):
        self.model = model
        self.hp    = hp
        self.opt   = tf.keras.optimizers.AdamW(hp['learning_rate'])
        self.mse   = tf.keras.losses.MeanSquaredError()

    def train_step(self, inputs):
        """Called inside strategy.run, so the tape/gradients are per‑replica."""
        drug_atoms, adj_mask, gex, mut, cnv, ic50_true = inputs
        with tf.GradientTape() as tape:
            ic50_pred, kl_loss = self.model(
                (drug_atoms, adj_mask, gex, mut, cnv), training=True)
            reg_loss = tf.reduce_mean(self.mse(ic50_true, ic50_pred))
            total_loss = reg_loss + self.hp['vae_beta'] * kl_loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return reg_loss, kl_loss

    @tf.function
    def distributed_train_step(self, inputs):
        per_replica_losses = strategy.run(self.train_step, args=(inputs,))
        reduced_reg = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                      per_replica_losses[0], axis=None)
        reduced_kl  = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                      per_replica_losses[1], axis=None)
        return reduced_reg, reduced_kl

    def val_step(self, inputs):
        drug_atoms, adj_mask, gex, mut, cnv, ic50_true = inputs
        ic50_pred, kl_loss = self.model(
            (drug_atoms, adj_mask, gex, mut, cnv), training=False)
        reg_loss = tf.reduce_mean(self.mse(ic50_true, ic50_pred))
        return reg_loss

    @tf.function
    def distributed_val_step(self, inputs):
        per_replica_loss = strategy.run(self.val_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    def fit(self, train_ds, val_ds, epochs=30):
        for epoch in range(1, epochs+1):
            train_regs, train_kls = [], []
            for batch in train_ds:
                reg, kl = self.distributed_train_step(batch)
                train_regs.append(reg.numpy())
                train_kls.append(kl.numpy())

            val_losses = []
            for batch in val_ds:
                loss = self.distributed_val_step(batch)
                val_losses.append(loss.numpy())

            t_rmse = np.sqrt(np.mean(train_regs))
            v_rmse = np.sqrt(np.mean(val_losses))
            print(f"Epoch {epoch:3d} | Train RMSE: {t_rmse:.4f} | "
                  f"Val RMSE: {v_rmse:.4f} | KL: {np.mean(train_kls):.4f}")

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
def main():
    CCLE_DIR = "CCLE"
    EXPR_FILE = os.path.join(CCLE_DIR, "data_mrna_seq_rpkm.txt")
    MUT_FILE  = os.path.join(CCLE_DIR, "data_mutations.txt")
    CNA_FILE  = os.path.join(CCLE_DIR, "data_cna.txt")
    IC50_FILE = os.path.join(CCLE_DIR, "data_drug_treatment_ic50.txt")

    # 1. Load raw omics
    gex_raw   = load_expression(EXPR_FILE)
    mut_raw   = load_mutations_maf(MUT_FILE)
    cna_raw   = load_cna(CNA_FILE)
    ic50_raw  = load_ic50(IC50_FILE)

    # 2. Drug SMILES
    drugs = ic50_raw.columns.tolist()
    smiles_map = build_drug_smiles_map(drugs)

    # 3. Build streaming dataset (with feature trimming inside)
    train_ds, val_ds, info = build_ccle_data(
        gex_raw, mut_raw, cna_raw, ic50_raw, smiles_map,
        batch_size=BASE_HP['batch_size'], max_genes=6000  # tune as needed
    )

    # 4. HP adjustment
    custom_hp = BASE_HP.copy()
    custom_hp.update({
        'gex_dim': info['gex_dim'],
        'mut_dim': info['mut_dim'],
        'cnv_dim': info['cnv_dim'],
    })
    print("\n[HP] Updated:")
    for k, v in custom_hp.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")

    # 5. Build model inside strategy scope
    with strategy.scope():
        model = BiIntDigitalTwin(custom_hp)
        trainer = DistributedTrainer(model, custom_hp)

    print(f"\nTrain pairs: {info['num_train']}, Val pairs: {info['num_val']}")
    print("\n[Training] Starting...")
    history = trainer.fit(train_ds, val_ds, epochs=30)

    # 6. Save
    SAVE_DIR = "trained_model"
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(os.path.join(SAVE_DIR, "biint_digital_twin.keras"))
    meta = {'gex_dim': info['gex_dim'], 'mut_dim': info['mut_dim'],
            'cnv_dim': info['cnv_dim'], 'hp': custom_hp}
    with open(os.path.join(SAVE_DIR, "metadata.json"), 'w') as f:
        json.dump(meta, f)
    print(f"\n[Save] Model saved to {SAVE_DIR}")

    # 7. Inference demo
    print("\n[Inference] Demo:")
    twin = DigitalTwinInference(model, BRICSMolecularFeaturizer())
    cell = info['common_cells'][0]
    drug = info['valid_drugs'][0]
    smi = smiles_map[drug]
    gex_vec = gex_raw.loc[cell].values.astype(np.float32)
    mut_vec = mut_raw.loc[cell].values.astype(np.float32) if cell in mut_raw.index else np.zeros(info['mut_dim'], dtype=np.float32)
    cna_vec = cna_raw.loc[cell].values.astype(np.float32) if cell in cna_raw.index else np.zeros(info['cnv_dim'], dtype=np.float32)
    print(f"  Cell: {cell}\n  Drug: {drug} ({smi[:40]}...)")
    ic50_pred = twin.predict_ic50(smi, gex_vec, mut_vec, cna_vec)
    print(f"  Predicted IC50: {ic50_pred:.3f} log µM")
    print("\n[Done]")

if __name__ == "__main__":
    main()