# smiles_sanitizer.py

from rdkit import Chem
from rdkit.Chem import SanitizeMol, SanitizeFlags
import logging

# Supprimer les warnings RDKit dans le terminal
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def sanitize_smiles(smiles: str) -> str | None:
    """
    Tente de parser et sanitizer un SMILES.
    Retourne le SMILES canonique si valide, None sinon.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # Filtre rapide — rejeter les SMILES clairement corrompus
    if smiles.startswith('?'):
        return None
    if len(set(smiles)) < 2:          # ex: "OOOOOOOO" ou "11111111"
        return None
    if smiles.count('(') != smiles.count(')'):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # Sanitization partielle — continue même si aromaticité échoue
        try:
            SanitizeMol(mol)
        except Exception:
            # Tentative de sanitization partielle
            try:
                SanitizeMol(
                    mol,
                    SanitizeFlags.SANITIZE_ALL ^
                    SanitizeFlags.SANITIZE_PROPERTIES
                )
            except Exception:
                return None

        # Retourner le SMILES canonique propre
        canonical = Chem.MolToSmiles(mol)
        return canonical if canonical else None

    except Exception:
        return None


def filter_smiles_list(smiles_list: list) -> list:
    """
    Filtre une liste de SMILES — garde uniquement les valides.
    Affiche un résumé du filtrage.
    """
    valid = []
    invalid_count = 0

    for smi in smiles_list:
        clean = sanitize_smiles(smi)
        if clean:
            valid.append(clean)
        else:
            invalid_count += 1

    print(f"✅ Valides   : {len(valid)}")
    print(f"❌ Rejetés   : {invalid_count}")
    print(f"📊 Taux valid: {len(valid)/len(smiles_list)*100:.1f}%")
    return valid