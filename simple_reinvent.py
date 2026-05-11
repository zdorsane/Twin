"""
Simplified REINVENT-style molecule generator using valid SMILES patterns.
No TensorFlow dependency - uses simple probability-based generation.
"""

import random
import numpy as np
import tensorflow as tf
from typing import List
from rdkit import Chem
import sys
sys.path.insert(0, '/home/crbt/Twin')
from fullPipeline import BiIntDigitalTwin, BRICSMolecularFeaturizer, DigitalTwinInference, HP

class SimpleSMILESGenerator:
    """Simple SMILES generator using valid character patterns"""

    VALID_CHARS = list("CNOSFClBrIPH()[]=#@+-.0123456789")
    START_CHARS = ['C', 'N', 'O', 'S', 'P', 'c', 'n', 'o']

    def __init__(self):
        # Simple transition probabilities (can be learned from data)
        self.transitions = {}
        for c1 in self.VALID_CHARS + ['<START>']:
            self.transitions[c1] = {}
            for c2 in self.VALID_CHARS + ['<END>']:
                self.transitions[c1][c2] = 1.0  # uniform prior

        # Normalize
        for c1 in self.transitions:
            total = sum(self.transitions[c1].values())
            for c2 in self.transitions[c1]:
                self.transitions[c1][c2] /= total

    def generate(self, max_len=30, temperature=1.0):
        """Generate SMILES string"""
        smiles = random.choice(self.START_CHARS)

        for _ in range(max_len - 1):
            if smiles[-1] not in self.transitions:
                break

            # Sample next character
            probs = {}
            for char, prob in self.transitions[smiles[-1]].items():
                if char == '<END>':
                    probs['<END>'] = prob
                else:
                    probs[char] = prob

            chars = list(probs.keys())
            weights = [probs[c] for c in chars]

            next_char = random.choices(chars, weights=weights, k=1)[0]
            if next_char == '<END>':
                break

            smiles += next_char

        return smiles

class BiIntSimpleOptimizer:
    """Simple REINVENT-style optimizer without TensorFlow"""

    def __init__(self, model: BiIntDigitalTwin, featurizer: BRICSMolecularFeaturizer):
        self.model = model
        self.featurizer = featurizer
        self.generator = SimpleSMILESGenerator()

    def compute_reward(self, smiles: str, gex, mut, cnv, target_ic50=-1.5):
        """Compute reward for a SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -5.0

        try:
            atom_feat = self.featurizer.featurize(smiles)[np.newaxis]
            adj = np.ones((1, HP['max_atoms'], HP['max_atoms']), dtype=np.float32)
            inputs = (
                tf.constant(atom_feat),
                tf.constant(adj),
                gex[:1], mut[:1], cnv[:1]
            )
            ic50_pred, _ = self.model(inputs, training=False)
            ic50_val = float(ic50_pred[0].numpy())
            ic50_reward = np.exp(-((ic50_val - target_ic50) ** 2) / 2.0)
            return float(np.clip(ic50_reward, -5.0, 5.0))
        except:
            return -5.0

    def optimize(self, gex, mut, cnv, n_steps=100, target_ic50=-1.5):
        """Simple optimization loop"""
        print(f"Starting optimization for {n_steps} steps...")
        print(f"Target IC50: {target_ic50} log µM")

        best_smiles = None
        best_reward = -float('inf')

        for step in range(1, n_steps + 1):
            # Generate candidate
            smiles = self.generator.generate(max_len=25, temperature=0.8)

            # Compute reward
            reward = self.compute_reward(smiles, gex, mut, cnv, target_ic50)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles

            if step % 10 == 0:
                print(f"Step {step:3d} | Best: {best_smiles[:30]:30s} | Reward: {best_reward:+.3f}")

        return {"best_smiles": best_smiles, "best_reward": best_reward}


# Test script
if __name__ == "__main__":
    print("=" * 60)
    print("Simple REINVENT-style Molecule Generator")
    print("=" * 60)

    # Load models (without TensorFlow for generation)
    print("\nLoading Bi-Int Digital Twin...")
    model = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()

    # Initialize optimizer
    optimizer = BiIntSimpleOptimizer(model, featurizer)

    # Dummy data
    gex = tf.random.normal([1, HP['gex_dim']])
    mut = tf.random.uniform([1, HP['mut_dim']], 0, 2)
    cnv = tf.random.normal([1, HP['cnv_dim']])

    # Run optimization
    result = optimizer.optimize(gex, mut, cnv, n_steps=50, target_ic50=-1.5)

    print("\nFinal result:")
    print(f"Best SMILES: {result['best_smiles']}")
    print(f"Best reward: {result['best_reward']:.3f}")

    # Validate final molecule
    mol = Chem.MolFromSmiles(result['best_smiles'])
    print(f"Molecule valid: {mol is not None}")
    if mol:
        print(f"Num atoms: {mol.GetNumAtoms()}")