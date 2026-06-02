"""Identify ncRNA among the top-978 GEx genes."""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENE_FILE = os.path.join(ROOT, "Dataset", "top978_gex_genes.txt")

with open(GENE_FILE) as f:
    genes = [l.strip() for l in f if l.strip()]

# Known ncRNA names (exact)
KNOWN_NCRNA = {
    "MALAT1","NEAT1","HOTAIR","XIST","H19","MEG3","KCNQ1OT1","TUG1","GAS5",
    "PVT1","FENDRR","FIRRE","NORAD","DANCR","SNHG1","SNHG3","SNHG5","SNHG6",
    "SNHG7","SNHG12","SNHG15","SNHG16","SNHG17","LINC00152","LINC00261",
    "LINC00355","LINC00472","LINC00473","LINC00657","LINC00839","LINC01116",
    "LINC01234","LINC01268","LINC01296","LINC01558","LINC01600","LINC01781",
    "LINC02454","MIAT","HULC","HOXA11-AS","SNHG4","SNHG8","SNHG9","SNHG10",
    "SNHG11","SNHG14","MIR155HG","CASC2","PCAT1","PCAT2","PCAT6",
    "RMRP","RPPH1","RNU1-1","TERC","SCARNA","SNORA",
}

# Pattern-based ncRNA detection
NCRNA_PATTERNS = [
    r'^LINC\d',        # LINC lncRNA
    r'^MIR\d',         # microRNA host genes
    r'^SNHG\d',        # small nucleolar RNA host genes
    r'^SNORD\d',       # snoRNA
    r'^SNORA\d',       # snoRNA
    r'^SCARNA\d',      # small Cajal body RNA
    r'^RNU\d',         # snRNA
    r'^RP\d+[-_]',     # read-through / antisense loci (RP11- etc)
    r'^AC\d{6}\.',     # Ensembl AC loci
    r'^AL\d{6}\.',     # Ensembl AL loci
    r'^AP\d{6}\.',     # Ensembl AP loci
    r'^Z\d{5}\.',      # Ensembl Z loci
    r'[-]AS\d*$',      # antisense transcripts (GENE-AS1)
    r'[-]IT\d*$',      # intronic transcripts
    r'[-]OT\d*$',      # overlapping transcripts
    r'^PURPL$',r'^CASC\d',r'^PCAT\d',r'^MIAT$',r'^HULC$',r'^DANCR$',
]

ncrna_found = []
coding_genes = []

for g in genes:
    is_nc = False
    reason = ""
    if g in KNOWN_NCRNA:
        is_nc = True
        reason = "known_ncrna"
    else:
        for pat in NCRNA_PATTERNS:
            if re.search(pat, g, re.IGNORECASE):
                is_nc = True
                reason = pat
                break
    if is_nc:
        ncrna_found.append((g, reason))
    else:
        coding_genes.append(g)

print(f"\nncRNA found: {len(ncrna_found)}/{len(genes)}")
print(f"Coding genes: {len(coding_genes)}/{len(genes)}")
print("\ncRNA list:")
for name, reason in sorted(ncrna_found):
    print(f"  {name:30s}  [{reason}]")

# Check specific targets
targets = ["H19","GAS5","MALAT1","NEAT1","HOTAIR","MEG3","XIST","TUG1","PVT1","NORAD"]
print("\nKey ncRNA targets present:")
present = [g for g,_ in ncrna_found]
for t in targets:
    print(f"  {t}: {'✓ FOUND at index ' + str(genes.index(t)) if t in genes else '✗ not in top-978'}")

# Save
import pandas as pd
df = pd.DataFrame(ncrna_found, columns=["gene","detection_reason"])
df["index_in_978"] = df["gene"].apply(lambda g: genes.index(g))
df.to_csv(os.path.join(ROOT, "Dataset", "ncrna_in_top978.csv"), index=False)
print(f"\nSaved to Dataset/ncrna_in_top978.csv")
