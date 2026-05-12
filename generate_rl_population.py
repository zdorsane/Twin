#!/usr/bin/env python3
import os
import tensorflow as tf
from fullPipeline import (
    SMILESVocabulary,
    BiIntDigitalTwin,
    BRICSMolecularFeaturizer,
    DrugGeneratorPolicy,
    PPODrugGenerator,
    PRETRAIN_SMILES,
    HP,
    load_pretrained_drug_encoder,
)


def main():
    print("[RL GEN] Building model and policy for RL candidate generation...")
    vocab = SMILESVocabulary()
    HP['vocab_size'] = vocab.vocab_size
    model = BiIntDigitalTwin(HP)
    if not load_pretrained_drug_encoder(model):
        print("[RL GEN] Warning: pre-trained drug encoder weights not found.")

    featurizer = BRICSMolecularFeaturizer()
    policy = DrugGeneratorPolicy(vocab_size=vocab.vocab_size)

    dummy_gex = tf.random.normal([16, HP['gex_dim']])
    dummy_mut = tf.random.uniform([16, HP['mut_dim']], 0, 2, dtype=tf.float32)
    dummy_cnv = tf.random.normal([16, HP['cnv_dim']])
    z_sample, _, _ = model.omics_vae((dummy_gex, dummy_mut, dummy_cnv), training=False)

    ppo = PPODrugGenerator(policy, model, vocab, featurizer, HP)
    valid_smiles_examples = PRETRAIN_SMILES[: min(len(PRETRAIN_SMILES), 16)]
    print(f"[RL GEN] Pretraining policy on {len(valid_smiles_examples)} valid SMILES...")
    ppo.pretrain_on_valid_smiles(valid_smiles_examples, z_sample[: len(valid_smiles_examples)], epochs=20)

    rl_smiles = []
    max_rl_smiles = 40
    for i in range(5):
        token_ids = policy.generate(z_sample, temperature=0.85)
        smiles_batch = vocab.batch_decode(token_ids.numpy())
        for smi in smiles_batch:
            smi = smi.strip()
            if smi and smi not in rl_smiles:
                rl_smiles.append(smi)
            if len(rl_smiles) >= max_rl_smiles:
                break
        if len(rl_smiles) >= max_rl_smiles:
            break

    if not rl_smiles:
        raise RuntimeError("No RL-generated SMILES were produced.")

    with open("rl_generated_smiles.txt", "w") as f:
        for smi in rl_smiles:
            f.write(smi + "\n")

    with open("smiles_data.txt", "w") as f:
        for smi in rl_smiles:
            f.write(smi + "\n")

    print(f"[RL GEN] Saved {len(rl_smiles)} RL-generated SMILES to rl_generated_smiles.txt and smiles_data.txt")
    print("[RL GEN] Sample candidates:")
    for smi in rl_smiles[:10]:
        print(f"  {smi}")


if __name__ == "__main__":
    main()
