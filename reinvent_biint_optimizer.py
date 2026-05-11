"""
REINVENT-style RL for SMILES generation with Bi-Int Digital Twin scoring.
Uses a simpler architecture that doesn't require pre-trained REINVENT weights.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple
from rdkit import Chem

# Import our components
import sys
sys.path.insert(0, '/home/crbt/Twin')
from fullPipeline import BiIntDigitalTwin, BRICSMolecularFeaturizer, SMILESVocabulary, DrugGeneratorPolicy, HP


def load_smiles_from_file(filepath: str) -> List[str]:
    path = Path(filepath)
    if not path.exists():
        return []
    with path.open('r') as f:
        smiles = [line.strip() for line in f if line.strip()]
    return smiles

class BiIntReinventOptimizer:
    """
    REINVENT-style RL optimization using Bi-Int Digital Twin as reward oracle.
    """

    def __init__(self, model: BiIntDigitalTwin, featurizer: BRICSMolecularFeaturizer,
                 vocab: SMILESVocabulary, hp=HP):
        self.model = model
        self.featurizer = featurizer
        self.vocab = vocab
        self.hp = hp

        self.policy = DrugGeneratorPolicy(vocab_size=vocab.vocab_size)
        self.optimizer = keras.optimizers.Adam(1e-4)

        print("BiIntReinventOptimizer initialized")

    def get_omics_latent(self, gex, mut, cnv):
        z, _, _ = self.model.omics_vae((gex, mut, cnv), training=False)
        return z

    def compute_reward(self, smiles_batch: List[str], gex, mut, cnv,
                      target_ic50=-1.5, validity_weight=0.6) -> np.ndarray:
        rewards = []
        for smiles in smiles_batch:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rewards.append(-5.0)
                continue

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
                reward = validity_weight * 1.0 + (1 - validity_weight) * (1 + ic50_reward) * 2.0
                reward = float(np.clip(reward, -5.0, 5.0))
                rewards.append(reward)
            except Exception:
                rewards.append(-5.0)
        return np.array(rewards, dtype=np.float32)

    def behavior_cloning_warmup(self, smiles_list: List[str], z, epochs=200):
        print("\n[BC Warmup] Behavior cloning on valid SMILES...")
        encoded = self.vocab.batch_encode(smiles_list)
        batch_size = int(encoded.shape[0])
        z = tf.convert_to_tensor(z)
        z = z[:batch_size]
        if z.shape[0] != batch_size:
            z = tf.repeat(z[:1], repeats=batch_size, axis=0)

        initial_loss = None
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                logits, _, _ = self.policy(encoded, z, training=True)
                targets = encoded[:, 1:]
                logits_shifted = logits[:, :-1, :]
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits_shifted
                )
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

            if epoch == 0:
                initial_loss = loss.numpy()
            if epoch % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

        final_loss = loss.numpy()
        print(f"[BC Warmup] Loss: {initial_loss:.4f} → {final_loss:.4f} (improvement: {initial_loss - final_loss:.4f})")

    def rl_step(self, gex, mut, cnv, batch_size=16, target_ic50=-1.5):
        z = self.get_omics_latent(gex, mut, cnv)
        z = z[:batch_size]
        token_ids = self.policy.generate(z, max_len=40, temperature=0.8, step=0, total_steps=100)
        smiles_list = self.vocab.batch_decode(token_ids.numpy())

        # Debug: check first few generated SMILES
        print(f"  Generated samples: {smiles_list[:3]}")

        rewards = self.compute_reward(smiles_list, gex, mut, cnv, target_ic50)

        with tf.GradientTape() as tape:
            logits, _, _ = self.policy(token_ids, z, training=True)
            log_probs = tf.nn.log_softmax(logits)
            token_logprobs = tf.reduce_sum(
                log_probs * tf.one_hot(token_ids, self.vocab.vocab_size),
                axis=-1
            )
            seq_logprobs = tf.reduce_sum(token_logprobs, axis=1)
            rewards_t = tf.constant(rewards, dtype=tf.float32)
            baseline = tf.reduce_mean(rewards_t)
            advantages = rewards_t - baseline
            policy_loss = -tf.reduce_mean(seq_logprobs * advantages)

        grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return {
            "smiles_list": smiles_list,
            "rewards": rewards,
            "mean_reward": rewards.mean(),
            "best_smiles": smiles_list[np.argmax(rewards)],
            "best_reward": rewards.max(),
            "policy_loss": float(policy_loss.numpy())
        }

    def optimize(self, gex, mut, cnv, n_steps=500, batch_size=16, target_ic50=-1.5,
                smiles_warmup=None):
        z = self.get_omics_latent(gex, mut, cnv)
        if smiles_warmup:
            self.behavior_cloning_warmup(smiles_warmup, z, epochs=100)

        print(f"\n[REINVENT RL] Starting {n_steps} steps of optimization...")
        print(f"  Target IC50: {target_ic50} log µM")
        print(f"  Batch size: {batch_size}")

        best_trajectory = []
        for step in range(1, n_steps + 1):
            stats = self.rl_step(gex, mut, cnv, batch_size, target_ic50)
            best_trajectory.append(stats['best_reward'])
            if step % 10 == 0 or step == 1:
                print(f"  Step {step:4d} | Mean Reward: {stats['mean_reward']:+.3f} | "
                      f"Best: {stats['best_smiles'][:35]:35s} | {stats['best_reward']:.3f}")

        print(f"\nOptimization complete!")
        return {
            "trajectory": best_trajectory,
            "best_smiles": stats['best_smiles'],
            "best_reward": stats['best_reward']
        }


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("  REINVENT + Bi-Int Digital Twin Integration")
    print("=" * 70)
    
    # Load models
    print("\n[Loading] Bi-Int Digital Twin...")
    dt_model = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()
    vocab = SMILESVocabulary()
    
    # Initialize optimizer
    print("[Loading] REINVENT Optimizer...")
    optimizer = BiIntReinventOptimizer(dt_model, featurizer, vocab, HP)
    
    # Dummy data
    gex = tf.random.normal([1, HP['gex_dim']])
    mut = tf.random.uniform([1, HP['mut_dim']], 0, 2)
    cnv = tf.random.normal([1, HP['cnv_dim']])

    # Test generation before warmup
    print("\n[Pre-Warmup] Testing generation...")
    z_test = optimizer.get_omics_latent(gex, mut, cnv)
    test_tokens = optimizer.policy.generate(z_test, max_len=20, temperature=1.0, step=0, total_steps=1)
    test_smiles = vocab.batch_decode(test_tokens.numpy())
    print(f"  Pre-warmup sample: {test_smiles[0]}")

    
    # Warmup SMILES: use the full file if available
    warmup_smiles = load_smiles_from_file('smiles_data.txt')
    if not warmup_smiles:
        warmup_smiles = [
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "CC(=O)Oc1ccccc1C(=O)O",
            "C1=CN=CC=C1",
        ]

    print(f"[Data] Loaded {len(warmup_smiles)} SMILES for warmup")
    print(f"[Data] Sample SMILES: {warmup_smiles[:3]}")

    # Run optimization
    result = optimizer.optimize(
        gex, mut, cnv,
        n_steps=50,
        batch_size=8,
        target_ic50=-1.5,
        smiles_warmup=warmup_smiles
    )
    
    print(f"\nFinal best SMILES: {result['best_smiles']}")
    print(f"Final reward: {result['best_reward']:.3f}")
