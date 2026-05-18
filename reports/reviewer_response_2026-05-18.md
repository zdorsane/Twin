# Response to Reviewer Feedback
**Date:** 2026-05-18
**Reviewer comment:** *"I really loved the report and the work but just something
maybe you need try to use brics in tokenization and crossentropy as loss function
you will get better results than kl divergence"*

---

## Interpretation of Suggestions

### Suggestion 1 — BRICS tokenization in the DQN

**What BRICS is:** BRICS (Break Retrosynthetically Interesting Chemical Substructures)
is an RDKit algorithm that decomposes molecules into fragments following 16
retrosynthesis rules. Unlike SELFIES which encodes atoms one-by-one, each BRICS
fragment is a complete medicinal chemistry scaffold (phenyl, piperazine, indole,
morpholine, etc.) with attachment points `[*:N]`.

**Why this may help vs SELFIES token-by-token:**

| Aspect | SELFIES (v3–v5) | BRICS DQN |
|--------|-----------------|-----------|
| Action granularity | Single atom/bond token | Complete scaffold fragment |
| Tokens to encode benzene | 8 tokens (`[C][=C][C][=C][C][=C][Ring1][=Branch1]`) | 1 token (`[*:1]c1ccccc1`) |
| Drug-likeness guarantee | Reward penalty only | Fragments are by construction from drug-like ChEMBL molecules |
| Synthetic accessibility | Not guaranteed | BRICS fragments are retrosynthetically accessible by definition |
| Vocabulary size | 50 tokens | 50–150 fragments (freq ≥ 5 in 10k ChEMBL) |

**Root cause of SELFIES DQN failures (v3–v5):** The DQN consistently generated
acyclic molecules (arom_rings=0 in all Top-5) despite 9,349/10,000 corpus molecules
being aromatic. The aromatic closure token `[=Branch1]` is a SELFIES-specific
abstraction with no chemical meaning — the DQN cannot associate it with "aromaticity"
from the reward signal. BRICS fragments solve this: `[*:1]c1ccccc1` is visibly
aromatic and the DQN learns fragment-level semantics directly.

### Suggestion 2 — Cross-entropy loss in the VAE

**What the reviewer likely means:** The current VAE reconstruction loss is implicit
(MSE via the regression pathway, plus KL regularization). Replacing or supplementing
the KL term with a binary cross-entropy (BCE) reconstruction loss treats the omics
features as pseudo-probabilities, which is more appropriate for:
- Gene expression profiles: bounded after normalization, interpretable as relative
  activity levels
- CNV data: already close to binary (amplification / deletion / neutral)
- Mutation features: binary (0/1 per gene)

BCE is also more sensitive to reconstruction errors in sparse features (mutations)
than MSE, which is dominated by high-variance GEx features.

**Three modes implemented:**

| Mode | Loss | Regularization | Expected behavior |
|------|------|---------------|-------------------|
| `kl` (original) | MSE regression + β·KL | KL=64 nats → latent fully used | Baseline — unchanged |
| `cross_entropy` | MSE regression + BCE recon | No KL (deterministic-ish AE) | Better omics reconstruction, less regularization |
| `both` | MSE regression + β·(KL + BCE) | Strongest regularization | Potentially most stable latent space |

---

## Implementation Status

### Files created / modified

| File | Action | Description |
|------|--------|-------------|
| `brics_dqn_optimizer.py` | **Created** (860 lines) | Full BRICS fragment DQN — `BRICSVocabulary`, `BRICSEnv`, `BRICSDQNOptimizer` |
| `fullPipeline.py` | **Modified** | `UnifiedOmicsVAE.call(loss_mode)`, `BiIntDigitalTwin.call(loss_mode)`, `BiIntTrainer(loss_mode)`, `--loss-mode` argparse flag |
| `compare_vae_losses.py` | **Created** (176 lines) | Runs all 3 loss modes × 10 epochs, saves `Dataset/vae_loss_comparison.csv` |
| `reports/reviewer_response_2026-05-18.md` | **Created** (this file) | Detailed response to reviewer |

### Backward compatibility

`fullPipeline.py` default is `loss_mode='kl'` everywhere — existing runs are
strictly unaffected unless `--loss-mode cross_entropy` or `--loss-mode both` is
passed explicitly.

---

## How to run the experiments

### BRICS DQN

```bash
cd ~/Twin && source venv_tf/bin/activate
nohup python3 brics_dqn_optimizer.py > logs_brics_dqn.txt 2>&1 &
tail -f logs_brics_dqn.txt
# Duration: ~20-30 min (5000 episodes, BRICS build per step)
# Output: Dataset/brics_dqn_results.csv, dqn_weights_brics/
```

### VAE loss comparison

```bash
cd ~/Twin && source venv_tf/bin/activate
python3 compare_vae_losses.py --epochs 10 --no-pretrain
# Duration: ~30-60 min (3 modes × 10 epochs on CCLE 137k triplets)
# Output: Dataset/vae_loss_comparison.csv
```

### fullPipeline with cross-entropy

```bash
python3 fullPipeline.py --loss-mode cross_entropy --epochs 20 --no-ppo
python3 fullPipeline.py --loss-mode both --epochs 20 --no-ppo
```

---

## Expected Results

### BRICS DQN

Based on the SELFIES failures analysis:
- **Valid%**: Expected 70–90% (BRICS.BRICSBuild is more constrained than SELFIES decode)
- **Aromatic rings**: Expected > 0 in Top-5 (fragments are drug-like by construction)
- **Moy50**: Expected to become positive before episode 1000 (fragments encode
  drug-like structure directly, warm-start from SEED_SMILES BRICS decomposition)
- **Best reward**: Expected > 2.5 with genuinely drug-like molecules (QED > 0.5,
  at least one aromatic ring)

Key difference from SELFIES: the BRICS success bonus (+0.3) and fragment diversity
bonus (+0.2 × unique/total) create additional positive signal specifically for
chemically valid fragment combinations.

### VAE cross-entropy loss

- **`cross_entropy` mode**: Val RMSE likely similar to `kl` baseline (0.45–0.50) but
  potentially better Pearson r because BCE penalizes systematic reconstruction
  errors more than MSE for sparse mutation features
- **`both` mode**: May train slower (combined loss landscape) but should produce
  a more interpretable latent space — each dimension encodes a reconstructible
  omics feature, not just a direction in KL space
- **Hypothesis to test**: If `cross_entropy` RMSE < 0.45 on random split, it suggests
  the omics reconstruction quality matters for downstream IC50 prediction

---

## Preliminary Analysis (pre-execution)

**On BRICS:** The corpus analysis showed that 93.5% of the 10k ChEMBL molecules
have aromatic rings, and the most common BRICS fragments will include phenyl,
pyridyl, piperazinyl, and morpholinyl scaffolds. The DQN action space (50–150
fragments) is semantically richer per action than the SELFIES vocab (50 tokens).

**On cross-entropy:** The key question is whether the VAE latent space `z` is
currently encoding real biological signal or mostly absorbing GEx variance through
the MSE pathway. BCE reconstruction on sparse features (mutations: mostly 0) would
force the VAE to compress differently — potentially exposing more biologically
meaningful dimensions. This aligns with the reviewer's intuition.

**Limitation to acknowledge:** The cross-entropy comparison is confounded by the
fact that drug features are still random vectors. Any improvement in VAE loss mode
is limited by this ceiling. The drug SMILES mapping (`fetch_drug_smiles.py`,
still pending execution) remains the highest-priority fix.

---

*Response authored after implementation — scripts not yet executed, results pending.*
*Run `compare_vae_losses.py` and `brics_dqn_optimizer.py` to populate the results section.*
