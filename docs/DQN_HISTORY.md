# DQN History — Twin Project

This file records the DQN version history from v1 through v5.1.

## v1 — Baseline SELFIES DQN
- Initial proof-of-concept DQN using SELFIES token generation.
- Reward based on raw molecular validity and an IC50 proxy.
- Result: many invalid or acyclic molecules due to weak aromatic closure reward.

## v2 — Reward shaping
- Added synthetic accessibility (SA) component.
- Added Lipinski penalties and higher-level drug-likeness signals.
- Outcome: improved chemical realism but still unstable validity.

## v3 — Diversity and fairness
- Introduced Tanimoto diversity bonus.
- Penalized repeated fragments and rewarded novel BRICS combinations.
- Result: better exploration, but SELFIES aromatic closure still problematic.

## v4 — Stability improvements
- Adjusted reward scaling for QED and Lipinski compliance.
- Added stronger invalid SMILES penalties.
- Outcome: more consistent reward learning, but average validity remained < 65%.

## v5.0 — BRICS-driven generation
- Shifted to BRICS fragment assembly for medicinal chemistry scaffolds.
- Maintained SA score, hard Lipinski penalty (−2.0), and diversity bonus.
- Result: 60.5% valid molecules and a best reward of 6.124/6.5.

## v5.1 — Aromatic closure reward hack
- Added a small reward bonus (+0.20) for the SELFIES token `[=Branch1]`.
- Purpose: encourage aromatic ring closure in generated molecules.
- Status: improvement in aromaticity signal, but full DQN aromaticity robustness remains work in progress.

## Current conclusion
- The main issue is representation: the BRICS approach is structurally stronger than the SELFIES token-level hack.
- v5.1 is the latest generation that combines usable fragment chemistry with a multi-objective drug-likeness reward.
