"""Test which SMILES work with RDKit — run this on Streamlit Cloud via st.write"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

smiles_list = [
    ("Benzene (simple)",         "c1ccccc1"),
    ("Aspirine",                  "CC(=O)Oc1ccccc1C(=O)O"),
    ("Imatinib",                  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("BRI-46",                    "O=S(=O)(c1ccc2ccccc2c1)N1CCNCC1"),
    ("BRI-12",                    "NS(=O)(=O)c1ccc(-c2cccc(O)c2)cc1"),
    ("Gra-1",                     "CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1"),
    ("Erlotinib (sans stereo)",   "COCCOc1cc2ncnc(Nc3cccc(Cl)c3)c2cc1OCCOC"),
    ("Gefitinib",                 "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    ("Afatinib (sans stereo)",    "CN(C)CCCOc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OC"),
]

results = []
for name, smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    results.append((name, smi, mol is not None))
    print(f"{'OK' if mol else 'FAIL'} | {name} | {smi}")

print("\nValides:", sum(1 for _,_,ok in results if ok), "/", len(results))
