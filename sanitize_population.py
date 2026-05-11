#!/usr/bin/env python3
"""
Post-process GraphGA results to sanitize and validate SMILES.
Handles kekulization, valence, aromaticity, and ring closure errors.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import pandas as pd
from rdkit import Chem, RDLogger

# Disable RDKit logging completely
RDLogger.DisableLog('rdApp.*')

def safe_mol_from_smiles(smiles: str) -> str | None:
    """
    Robust SMILES parsing with comprehensive error handling.
    Returns canonical SMILES or None if invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        # Parse without sanitization first
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        
        # Sanitize with error catching enabled
        Chem.SanitizeMol(mol, catchErrors=True)
        
        # Double-check via canonical SMILES
        canonical_smi = Chem.MolToSmiles(mol)
        if not canonical_smi:
            return None
        
        # Re-parse canonical form to ensure it's truly valid
        mol2 = Chem.MolFromSmiles(canonical_smi)
        if mol2 is None:
            return None
        
        return Chem.MolToSmiles(mol2)
    except Exception:
        return None


def sanitize_csv(input_file: str, output_file: str = None):
    """
    Load GraphGA CSV results and sanitize SMILES column.
    """
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    if output_file is None:
        output_file = input_file.replace(".csv", "_sanitized.csv")
    
    print(f"📖 Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    if "smiles" not in df.columns:
        print("❌ No 'smiles' column found in CSV")
        return
    
    print(f"🔧 Sanitizing {len(df)} SMILES...")
    df["smiles_canonical"] = df["smiles"].apply(safe_mol_from_smiles)
    df["valid"] = df["smiles_canonical"].notna()
    
    valid_count = df["valid"].sum()
    invalid_count = len(df) - valid_count
    
    print(f"✅ Valid   : {valid_count:3d} ({100*valid_count/len(df):.1f}%)")
    print(f"❌ Invalid : {invalid_count:3d} ({100*invalid_count/len(df):.1f}%)")
    
    # Save full results
    df.to_csv(output_file, index=False)
    print(f"💾 Saved full results: {output_file}")
    
    # Save only valid molecules
    df_valid = df[df["valid"]].copy()
    valid_output = output_file.replace(".csv", "_valid_only.csv")
    df_valid.to_csv(valid_output, index=False)
    print(f"💾 Saved valid only: {valid_output}")
    
    # Show sample of invalid molecules (if any)
    if invalid_count > 0:
        df_invalid = df[~df["valid"]]
        print(f"\n🔴 Sample invalid SMILES:")
        for idx, row in df_invalid.head(5).iterrows():
            print(f"   {row['smiles']}")


if __name__ == "__main__":
    input_file = "graphga_ranked_population.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    sanitize_csv(input_file)
