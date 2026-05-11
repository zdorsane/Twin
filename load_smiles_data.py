"""
Load valid SMILES from public sources for pre-training.
Run this once to create smiles_data.txt
"""

# Option 1: Manually curated drug-like SMILES (tested valid)
VALID_SMILES = [
    # Aromatic cores
    "c1ccccc1",  # benzene
    "c1cccnc1",  # pyridine
    "c1ccsc1",  # thiophene
    "c1ccoc1",  # furan
    
    # Common drugs
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC1=CC=C(C=C1)NC2=NC=CC(=N2)N3CCN(CC3)C4=CC=CC=C4",  # imatinib
    "COC1=CC2=C(C=C1OC)NC(=O)C2=CC3=CC=CC=N3",  # gefitinib
    
    # Simple molecules
    "CCO",  # ethanol
    "CC(C)O",  # isopropanol
    "C1CCCCC1",  # cyclohexane
    "CC(C)CC1=CC(=C(C=C1)O)C(C)C",  # carvacrol
    "C1=CC=C2C(=C1)C=CC(=C2)O",  # naphthol
    
    # Heterocycles
    "C1CCNCC1",  # piperidine
    "C1CCOCC1",  # tetrahydropyran
    "c1ccc2c(c1)cccc2",  # naphthalene
    "C1=CC=C(C=C1)N",  # aniline
    "C1=CC=C(C=C1)O",  # phenol
    
    # Esters & amides
    "CC(=O)Nc1ccc(cc1)O",  # paracetamol
    "CC(=O)Nc1ccccc1",  # acetanilide
    "CC(=O)OCC(=O)Nc1ccccc1",  # aspirin analog
    
    # Carboxylic acids
    "O=C(O)c1ccccc1",  # benzoic acid
    "CC(C)c1ccc(cc1)C(=O)O",  # isobutyric benzoate
    
    # Amines
    "NC(=O)c1ccccc1",  # benzamide
    "CCN(CC)CC",  # triethylamine
    "c1ccc(cc1)CCN",  # phenethylamine
    
    # Complex aromatics
    "C1=CC=C(C=C1)C2=CC=CC=C2",  # biphenyl
    "c1cc(ccc1c2ccccc2)C",  # methylbiphenyl
    "c1ccc2c(c1)ccc3c2cccc3",  # anthracene
    
    # Halogenated
    "C1=CC=C(C=C1)Cl",  # chlorobenzene
    "C1=CC=C(C=C1)F",  # fluorobenzene
    "C1=CC=C(C=C1)Br",  # bromobenzene
    
    # Sulfur compounds
    "C1=CC=C(C=C1)S(=O)(=O)N",  # sulfanilamide
    "c1ccc(s1)C",  # methylthiophene
    
    # Nitrogen compounds
    "C1=CC=C(C=C1)N=O",  # nitrosobenzene
    "c1ccc(cc1)[N+](=O)[O-]",  # nitrobenzene
    
    # Ketones & aldehydes
    "CC(=O)c1ccccc1",  # acetophenone
    "O=Cc1ccccc1",  # benzaldehyde
    
    # Ethers
    "CCOc1ccccc1",  # phenetole
    "COc1ccccc1O",  # catechol dimethyl ether
    
    # Vitamins & cofactors (simplified)
    "CC(C)=CCCC(C)=CC(=CC(=CC(=CC(=CC(=CC(=CC(=C(C)C)C)C)C)C)C)C)O",  # beta-carotene-like
    
    # More pharmaceuticals
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # naproxen
    "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",  # diclofenac
    "C1CN(CCN1CCCC(=O)Nc2ccc(Cl)cc2)c3ccc(cc3)S(=O)(=O)N",  # complex sulfonamide
    
    # Additional aromatics
    "c1ccc(cc1)c2ccccc2c3ccccc3",  # triphenylmethane-like
    "C1=CC=C(C=C1)C(=C(c2ccccc2)c3ccccc3)c4ccccc4",  # complex conjugate
]

def save_smiles_data():
    """Save SMILES to file for easy loading."""
    with open("smiles_data.txt", "w") as f:
        for smiles in VALID_SMILES:
            f.write(smiles + "\n")
    print(f"Saved {len(VALID_SMILES)} valid SMILES to smiles_data.txt")

def load_smiles_data(filepath="smiles_data.txt"):
    """Load SMILES from file."""
    try:
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Returning default SMILES list.")
        return VALID_SMILES

if __name__ == "__main__":
    save_smiles_data()
