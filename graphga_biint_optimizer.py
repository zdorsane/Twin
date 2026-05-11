import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false"
import random
import sys
import io
import logging
from contextlib import redirect_stderr
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

try:
    import rdkit_sascorer
    RDKitSAScorer_AVAILABLE = True
except Exception:
    rdkit_sascorer = None
    RDKitSAScorer_AVAILABLE = False

sys.path.insert(0, "/home/crbt/Twin")
from fullPipeline import BiIntDigitalTwin, BRICSMolecularFeaturizer, DigitalTwinInference, HP

INITIAL_POP_FILE = "smiles_data.txt"
MAX_POPULATION = 40
GENERATIONS = 50
OFFSPRING_PER_GEN = 40
MUTATION_RATE = 0.6
MIN_QED = 0.7
FRAGMENTS = [
    "CC",
    "CCC",
    "CO",
    "NCC",
    "C(=O)O",
    "c1ccccc1",
    "C1CC1",
    "N(C)C",
    "C#N",
]


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Robust SMILES parsing with comprehensive error handling.
    Catches kekulization, valence, aromaticity, and ring closure errors.
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
        
        return mol2
    except Exception:
        return None


def validate_smiles_syntax(smiles: str) -> bool:
    """Quick syntax check: balanced parentheses and rings."""
    if not smiles:
        return False
    # Check balanced parentheses
    if smiles.count('(') != smiles.count(')'):
        return False
    # Check balanced ring markers (simplified: %10, %11, etc. and single digits)
    ring_nums = {}
    i = 0
    while i < len(smiles):
        if smiles[i].isdigit():
            if i + 1 < len(smiles) and smiles[i + 1].isdigit():
                ring_num = int(smiles[i:i+2])
                ring_nums[ring_num] = ring_nums.get(ring_num, 0) + 1
                i += 2
            else:
                ring_num = int(smiles[i])
                ring_nums[ring_num] = ring_nums.get(ring_num, 0) + 1
                i += 1
        else:
            i += 1
    # Each ring number should appear exactly 0 or 2 times
    for count in ring_nums.values():
        if count % 2 != 0:
            return False
    return True


def is_valid_smiles(smiles: str) -> bool:
    return validate_smiles_syntax(smiles) and safe_mol_from_smiles(smiles) is not None


def canonical_smiles(smiles: str) -> Optional[str]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_initial_population(max_size: int = MAX_POPULATION) -> List[str]:
    if os.path.exists(INITIAL_POP_FILE):
        with open(INITIAL_POP_FILE, "r") as f:
            smiles = [line.strip() for line in f if line.strip()]
        valid = []
        for s in smiles:
            c = canonical_smiles(s)
            if c:
                valid.append(c)
                if len(valid) >= max_size:
                    break
        return valid
    return [
        "CC1=CC=CC=C1",
        "C1=CC=CC=C1O",
        "CCO",
        "CCC",
        "CCN",
        "COC",
        "CC(=O)O",
        "C1CCCCC1",
        "c1ccccc1",
        "C1=CN=CN=C1",
        "CCN(CC)CC",
        "CCOc1ccccc1",
    ]


def mutate_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    if random.random() < 0.5:
        i = random.randrange(len(smiles))
        fragment = random.choice(FRAGMENTS)
        candidate = smiles[:i] + fragment + smiles[i + 1 :]
    else:
        candidate = smiles + random.choice(FRAGMENTS)
    # Validate syntax before attempting canonical SMILES
    if not validate_smiles_syntax(candidate):
        return None
    return canonical_smiles(candidate)


def crossover_smiles(a: str, b: str) -> Optional[str]:
    if len(a) < 4 or len(b) < 4:
        candidate = random.choice([a, b])
    else:
        split_a = random.randrange(1, len(a) - 1)
        split_b = random.randrange(1, len(b) - 1)
        candidate = a[:split_a] + b[split_b:]
    # Validate syntax before attempting canonical SMILES
    if not validate_smiles_syntax(candidate):
        return None
    return canonical_smiles(candidate)


def synthetic_accessibility_score(mol) -> float:
    if RDKitSAScorer_AVAILABLE:
        try:
            return float(rdkit_sascorer.calculateScore(mol))
        except Exception:
            pass
    heavy = Descriptors.HeavyAtomCount(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    rings = mol.GetRingInfo().NumRings()
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    score = 1.0 - min(1.0, 0.08 * heavy + 0.08 * rot + 0.05 * max(0, rings - 3) + 0.03 * hetero_atoms)
    return float(max(0.0, min(1.0, score)))


def logp_penalty(logp: float) -> float:
    return max(0.0, abs(logp) - 2.0) * 0.15


def chem_quality(mol) -> tuple[float, float, float, float]:
    qed = float(QED.qed(mol))
    sa = synthetic_accessibility_score(mol)
    logp = float(Crippen.MolLogP(mol))
    penalty = logp_penalty(logp)
    composite = qed + sa - penalty
    return qed, sa, logp, composite


class GraphGABiIntOptimizer:
    def __init__(self):
        print("[GraphGA] Loading Bi-Int Digital Twin model...")
        self.model = BiIntDigitalTwin(HP)
        self.featurizer = BRICSMolecularFeaturizer()
        self.inference = DigitalTwinInference(self.model, self.featurizer)
        print("[GraphGA] Model loaded.")

    def fitness(self, smiles: str, gex: np.ndarray, mut: np.ndarray, cnv: np.ndarray) -> float:
        ic50 = float(self.inference.predict_ic50(smiles, gex, mut, cnv))
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            return -9999.0
        qed, sa, logp, composite = chem_quality(mol)
        return -ic50 + 2.0 * composite

    def evolve(
        self,
        initial_population: List[str],
        gex: np.ndarray,
        mut: np.ndarray,
        cnv: np.ndarray,
    ):
        population = list(dict.fromkeys(initial_population))[:MAX_POPULATION]
        population = [s for s in population if canonical_smiles(s)]
        print(f"[GraphGA] Starting population size: {len(population)}")

        history = []
        for generation in range(1, GENERATIONS + 1):
            offspring = []
            while len(offspring) < OFFSPRING_PER_GEN:
                if random.random() < MUTATION_RATE or len(population) < 2:
                    parent = random.choice(population)
                    child = mutate_smiles(parent)
                else:
                    p1, p2 = random.sample(population, 2)
                    child = crossover_smiles(p1, p2)
                if child and child not in offspring:
                    offspring.append(child)

            all_candidates = list(dict.fromkeys(population + offspring))
            candidates = [s for s in all_candidates if is_valid_smiles(s)]
            invalid_dropped = len(all_candidates) - len(candidates)
            if invalid_dropped > 0:
                print(f"[GraphGA] Gen {generation:02d}: dropped {invalid_dropped} invalid candidates before scoring")

            scored = []
            for smiles in candidates:
                score = self.fitness(smiles, gex, mut, cnv)
                scored.append((score, smiles))
            scored.sort(reverse=True, key=lambda x: x[0])

            # Keep the best QED candidates; if too few, fill with the next best.
            filtered = []
            for score, smiles in scored:
                mol = safe_mol_from_smiles(smiles)
                if mol is None:
                    continue
                qed_val = float(QED.qed(mol))
                if qed_val >= MIN_QED:
                    filtered.append((score, smiles))
                if len(filtered) >= MAX_POPULATION:
                    break

            if len(filtered) < MAX_POPULATION:
                filtered = filtered + [item for item in scored if item not in filtered][: MAX_POPULATION - len(filtered)]

            population = [smiles for _, smiles in filtered[:MAX_POPULATION]]
            best_score, best_smiles = filtered[0]
            mean_score = sum([s for s, _ in filtered[:MAX_POPULATION]]) / len(population)

            print(
                f"Gen {generation:02d} | Best score: {best_score:.4f} | "
                f"Mean score: {mean_score:.4f} | Best SMILES: {best_smiles}"
            )
            history.append((generation, best_score, best_smiles))

        ranked_candidates = []
        for score, smiles in scored[:MAX_POPULATION]:
            mol = safe_mol_from_smiles(smiles)
            if mol is None:
                continue
            qed, sa, logp, quality = chem_quality(mol)
            ic50 = float(self.inference.predict_ic50(smiles, gex, mut, cnv))
            ranked_candidates.append({
                "smiles": smiles,
                "ic50": ic50,
                "qed": qed,
                "sa": sa,
                "logp": logp,
                "quality": quality,
                "fitness": score,
            })
        self.save_ranked_population(ranked_candidates)

        return population, history

    def save_population(self, population: List[str], filename: str = "graphga_population.txt"):
        with open(filename, "w") as f:
            for smiles in population:
                f.write(smiles + "\n")
        print(f"[GraphGA] Saved top population to {filename}")

    def save_ranked_population(self, ranked_candidates, filename: str = "graphga_ranked_population.csv"):
        import csv

        fieldnames = ["rank", "smiles", "ic50", "qed", "sa", "logp", "quality", "fitness"]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, item in enumerate(ranked_candidates, 1):
                writer.writerow({
                    "rank": idx,
                    "smiles": item["smiles"],
                    "ic50": item["ic50"],
                    "qed": item["qed"],
                    "sa": item["sa"],
                    "logp": item["logp"],
                    "quality": item["quality"],
                    "fitness": item["fitness"],
                })
        print(f"[GraphGA] Saved ranked population to {filename}")


def build_dummy_omics() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(HP["gex_dim"], dtype=np.float32),
        np.zeros(HP["mut_dim"], dtype=np.float32),
        np.zeros(HP["cnv_dim"], dtype=np.float32),
    )


if __name__ == "__main__":
    optimizer = GraphGABiIntOptimizer()
    population = load_initial_population(MAX_POPULATION)
    gex, mut, cnv = build_dummy_omics()

    final_population, history = optimizer.evolve(population, gex, mut, cnv)
    optimizer.save_population(final_population)

    print("\nFinal top candidates:")
    for idx, smiles in enumerate(final_population[:10], 1):
        print(f"{idx:02d}. {smiles}")
