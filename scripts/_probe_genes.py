"""Quick probe: extract top-978 gene names from CCLE GEx data."""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEX_PATH = os.path.join(ROOT, "Dataset/ccle_broad_2019/data_mrna_seq_rpkm.txt")
IC50_PATH = os.path.join(ROOT, "Dataset/ccle_broad_2019/data_drug_treatment_ic50.txt")
CNA_PATH  = os.path.join(ROOT, "Dataset/ccle_broad_2019/data_cna.txt")

gex_df = pd.read_csv(GEX_PATH, sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
ic50_df = pd.read_csv(IC50_PATH, sep="\t", index_col=0)
cna_df  = pd.read_csv(CNA_PATH,  sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)

meta_cols = [c for c in ic50_df.columns if ic50_df[c].dtype == object]
ic50_df = ic50_df.drop(columns=meta_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")

common_cells = sorted(set(ic50_df.columns) & set(gex_df.columns) & set(cna_df.columns))
print(f"Common cells: {len(common_cells)}")

gex_sub = gex_df[common_cells].T
top_gex  = gex_sub.var(axis=0).sort_values(ascending=False).index[:978].tolist()
print(f"Top 978 GEx genes selected: {len(top_gex)}")
print("First 10:", top_gex[:10])
print("Last 10:", top_gex[-10:])

# Save gene list
out = os.path.join(ROOT, "Dataset", "top978_gex_genes.txt")
with open(out, "w") as f:
    for g in top_gex:
        f.write(g + "\n")
print(f"Saved to {out}")
