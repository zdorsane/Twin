"""
Shared CCLE data loader for interpretability scripts.
Loads from the NPZ cache (exact same matrices as used during training).
"""
import os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ccle_cached(hp_path=None):
    """
    Returns:
        gex_mat     : (647, 978)  float32  z-scored GEx
        cna_mat     : (647, 426)  float32  z-scored CNA
        mut_mat     : (647, 735)  float32  binary mutations
        common_cells: list[str]   647 cell names
        top_genes   : list[str]   978 GEx gene names (same order as gex_mat columns)
        HP          : dict        hyperparameters
        smiles_map  : dict        drug_name → SMILES
        ic50_df     : DataFrame   drugs × cells IC50 values
        drugs_w_smi : list[str]   drug names that have SMILES
    """
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from fullPipeline import HP as HP_DEFAULT
    HP = dict(HP_DEFAULT)

    if hp_path and os.path.exists(hp_path):
        with open(hp_path) as f:
            HP.update(json.load(f))

    ccle_dir  = os.path.join(ROOT, "Dataset/ccle_broad_2019")
    cache_path = os.path.join(ccle_dir, f"omics_cache_gex{HP['gex_dim']}_cna{HP['cnv_dim']}.npz")
    cache = np.load(cache_path, allow_pickle=True)
    common_cells = list(cache["common_cells"])
    gex_mat = cache["gex_mat"]   # (647, 978)
    cna_mat = cache["cna_mat"]   # (647, 426)

    # Gene names — from pre-saved list
    gene_list_path = os.path.join(ROOT, "Dataset/top978_gex_genes.txt")
    with open(gene_list_path) as f:
        top_genes = [l.strip() for l in f if l.strip()]
    assert len(top_genes) == gex_mat.shape[1], \
        f"Gene list length {len(top_genes)} != gex_mat cols {gex_mat.shape[1]}"

    # Mutations
    mut_mat = np.zeros((len(common_cells), HP["mut_dim"]), dtype=np.float32)
    mut_path = os.path.join(ccle_dir, "data_mutations.txt")
    if os.path.exists(mut_path):
        try:
            mut_df_raw = pd.read_csv(mut_path, sep="\t", low_memory=False,
                                     comment="#", on_bad_lines="skip")
            if "Tumor_Sample_Barcode" in mut_df_raw.columns and "Hugo_Symbol" in mut_df_raw.columns:
                c_set = set(common_cells)
                mut_df_raw = mut_df_raw[mut_df_raw["Tumor_Sample_Barcode"].isin(c_set)]
                top_mut = mut_df_raw["Hugo_Symbol"].value_counts().head(HP["mut_dim"]).index.tolist()
                c2i = {c: i for i, c in enumerate(common_cells)}
                for gi, gene in enumerate(top_mut):
                    cells_w_mut = mut_df_raw[mut_df_raw["Hugo_Symbol"] == gene]["Tumor_Sample_Barcode"].unique()
                    for c in cells_w_mut:
                        if c in c2i:
                            mut_mat[c2i[c], gi] = 1.0
        except Exception as e:
            print(f"[WARN] Mutations load failed: {e}")

    # SMILES
    smi_path = os.path.join(ROOT, "Dataset/ccle_drug_smiles.csv")
    smiles_map = {}
    if os.path.exists(smi_path):
        df_s = pd.read_csv(smi_path)
        for _, r in df_s.iterrows():
            if pd.notna(r.get("smiles")) and r["smiles"]:
                smiles_map[r["drug_name"]] = r["smiles"]

    # IC50
    ic50_path = os.path.join(ccle_dir, "data_drug_treatment_ic50.txt")
    ic50_df = pd.read_csv(ic50_path, sep="\t", index_col=0)
    meta_cols = [c for c in ic50_df.columns if ic50_df[c].dtype == object]
    ic50_df = ic50_df.drop(columns=meta_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")
    drugs_w_smi = [d for d in ic50_df.index if d in smiles_map]

    print(f"[Loader] cells={len(common_cells)}, genes={len(top_genes)}, "
          f"drugs_w_smi={len(drugs_w_smi)}, gex={gex_mat.shape}, cna={cna_mat.shape}, mut={mut_mat.shape}")
    return gex_mat, cna_mat, mut_mat, common_cells, top_genes, HP, smiles_map, ic50_df, drugs_w_smi


def sample_pairs(gex_mat, cna_mat, mut_mat, common_cells, smiles_map, ic50_df,
                 drugs_w_smi, n_pairs=150, seed=42):
    """Sample (drug, cell) pairs with valid IC50 and SMILES. Returns list of tensors."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from fullPipeline import BRICSMolecularFeaturizer
    featurizer = BRICSMolecularFeaturizer()
    c2i = {c: i for i, c in enumerate(common_cells)}
    rng = np.random.default_rng(seed)

    pairs, labels = [], []
    attempts = 0
    while len(pairs) < n_pairs and attempts < 10000:
        attempts += 1
        drug = rng.choice(drugs_w_smi)
        cell = rng.choice(common_cells)
        if cell not in ic50_df.columns:
            continue
        val = ic50_df.loc[drug, cell]
        if pd.isna(val):
            continue
        feats = featurizer.featurize(smiles_map[drug])
        if feats is None:
            continue
        atoms, adj = feats
        ci = c2i[cell]
        pairs.append((atoms, adj,
                      gex_mat[ci].copy(),
                      mut_mat[ci].copy(),
                      cna_mat[ci].copy()))
        labels.append((drug, cell, float(np.log1p(max(val, 0.001)))))

    print(f"[Loader] Sampled {len(pairs)} pairs ({attempts} attempts)")
    return pairs, labels
