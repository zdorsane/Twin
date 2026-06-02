"""
Verify gene list and save — use cache gex_mat directly for correct normalization.
Also saves common_cells order for downstream scripts.
"""
import os, numpy as np, pandas as pd

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "Dataset/ccle_broad_2019/omics_cache_gex978_cna426.npz")
GEX   = os.path.join(ROOT, "Dataset/ccle_broad_2019/data_mrna_seq_rpkm.txt")
CNA   = os.path.join(ROOT, "Dataset/ccle_broad_2019/data_cna.txt")

cache = np.load(CACHE, allow_pickle=True)
common_cells = list(cache["common_cells"])
gex_mat_ref  = cache["gex_mat"]   # (647, 978)

print(f"Cache: gex_mat {gex_mat_ref.shape}, {len(common_cells)} cells")

# Load GEx, dedup, find top-978
gex_df = pd.read_csv(GEX, sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
gex_df = gex_df[~gex_df.index.duplicated(keep="first")]
gex_sub = gex_df[common_cells].T.astype("float32")
top_genes = gex_sub.var(axis=0).sort_values(ascending=False).index[:978].tolist()
print(f"top_genes (dedup): {len(top_genes)} genes")

# Verify top gene names match regardless of normalization
# Use correlation to check gene ordering is correct
raw_vals = gex_sub[top_genes].values.astype("float32")
# Pearson correlation of each column between raw (unnorm) and cache
corr_per_gene = np.array([
    np.corrcoef(raw_vals[:, i], gex_mat_ref[:, i])[0, 1]
    for i in range(10)  # spot-check first 10
])
print(f"Correlation first 10 genes (raw vs cache): min={corr_per_gene.min():.4f} max={corr_per_gene.max():.4f}")
if corr_per_gene.min() > 0.95:
    print("✅ Gene ordering confirmed — same genes, different normalization scale")
else:
    print("⚠️  Low correlation — gene ordering may differ")

# Save gene list
out_genes = os.path.join(ROOT, "Dataset", "top978_gex_genes.txt")
with open(out_genes, "w") as f:
    for g in top_genes:
        f.write(g + "\n")

# Save common_cells order
out_cells = os.path.join(ROOT, "Dataset", "ccle_common_cells.txt")
with open(out_cells, "w") as f:
    for c in common_cells:
        f.write(c + "\n")

# Also save CNA top-426
cna_df = pd.read_csv(CNA, sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
cna_df = cna_df[~cna_df.index.duplicated(keep="first")]
cna_sub = cna_df[common_cells].T.astype("float32")
top_cna = cna_sub.var(axis=0).sort_values(ascending=False).index[:426].tolist()
out_cna = os.path.join(ROOT, "Dataset", "top426_cna_genes.txt")
with open(out_cna, "w") as f:
    for g in top_cna:
        f.write(g + "\n")

print(f"Saved: {out_genes}, {out_cells}, {out_cna}")
print(f"Sample genes: {top_genes[:5]} ... {top_genes[-3:]}")
