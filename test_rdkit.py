#!/usr/bin/env python3
from rdkit import Chem
import numpy as np

print("NumPy version:", np.__version__)
mol = Chem.MolFromSmiles("CC")
print("RDKit works:", mol is not None)

# Test SMILES validation
test_smiles = ["CC", "c1ccccc1", "invalid", "?N[=P#IF+Cl+07+C15C@0343(=5"]
for s in test_smiles:
    mol = Chem.MolFromSmiles(s)
    print(f"'{s}': {'valid' if mol else 'invalid'}")