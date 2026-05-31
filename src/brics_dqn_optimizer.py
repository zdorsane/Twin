r"""
================================================================================
  BRICS DQN Drug Optimizer -- Fragment Assembly -- Bi-Int Digital Twin
================================================================================

Architecture: replaces SELFIES token-by-token generation with BRICS fragment
assembly. The agent selects drug-like fragments from a vocabulary derived from
ChEMBL BRICS decomposition and assembles them into novel molecules.

Key differences from dqn_optimizer.py (SELFIES v5.x):
  - BRICSVocabulary: fragments with [*:N] attachment points, freq >= 5
  - BRICSEnv: episodes of up to 8 fragment selections, terminal on EOS
  - BRICS-specific rewards: build_success (+0.3), fragment_diversity (+0.2)
  - fragments_to_smiles: BRICS.BRICSBuild first, fallback to combined mol

References:
  Degen et al., "On the Art of Compiling and Using Drug-Relevant Substructure
    and Fragment Space", ChemMedChem 2008.
  Mnih et al., "Human-level control through deep RL", Nature 2015.
  Lipinski et al., "Experimental and computational approaches...", 1997.
================================================================================
"""

import os
import sys
import random
import warnings
import logging
import collections
import csv
import numpy as np

# Suppress TensorFlow / absl info logs before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Deque

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fullPipeline import (
    BiIntDigitalTwin,
    BRICSMolecularFeaturizer,
    HP,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors, QED as rdQED
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[ERROR] RDKit not available — pip install rdkit")
    sys.exit(1)


# ─── Seed SMILES (fallback if ChEMBL absent) ─────────────────────────────────
SEED_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1CCC[C@H]1c2cccnc2",
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "CN(C)CCOC(c1ccccc1)c2ccccc2",
    "O=C(O)c1ccccc1O",
    "Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)CCOCCO",
    "CC(=O)Nc1ccc(O)cc1",
    "Oc1ccc(cc1)C2CC(=O)c3c(O)cc(O)cc3O2",
    "c1ccc2ncccc2c1",
    "O=C(Nc1ccc(Cl)c(Cl)c1)N2CCC(CC2)N3CCOCC3",
    "Fc1ccc(cc1)C(=O)CCCN2CCC(CC2)c3noc4cc(F)ccc34",
    "c1ccc(cc1)CN2CCN(CC2)c3cccc(c3)Cl",
    "O=C1CCCN1",
    "c1ccc(cc1)c2cc(nn2c3ccccc3)C(F)(F)F",
    "CCOC(=O)c1cnc(N)c(Cl)c1F",
    "c1ccc(cc1)S(=O)(=O)Nc2ccc(cc2)N",
    "Cc1nc2ccccc2c(=O)n1Cc1ccccc1",
    "O=c1[nH]c2ccccc2n1Cc1ccccc1",
    "CCc1ccc(NC(=O)c2ccc(N)cc2)cc1",
    "O=C(O)c1ccc(Cl)cc1",
    "Cc1ccc(C(=O)O)cc1",
    "NC(=O)c1ccncc1",
    "Cc1ccc(-c2ccccc2)cc1",
]


# ─── ChEMBL SMILES loader (copied from dqn_optimizer.py) ─────────────────────
CHEMBL_SDF_PATH = os.path.join(_ROOT, "Dataset", "chembl_36.sdf")

def load_chembl_smiles(n: int = 10_000,
                       sdf_path: str = CHEMBL_SDF_PATH,
                       max_heavy: int = 40,
                       min_heavy: int = 8) -> List[str]:
    """
    Extract n drug-like SMILES from the ChEMBL SDF.
    Filter: 8-40 heavy atoms, QED >= 0.3, no metals, has carbon.
    Falls back to SEED_SMILES if SDF is not accessible.
    """
    if not os.path.exists(sdf_path):
        print(f"[Corpus] SDF not found ({sdf_path}) — using seed SMILES.")
        return SEED_SMILES

    FORBIDDEN_ATOMS = {5, 13, 14, 15, 33, 34, 50, 51, 52, 82, 83}

    smiles_list = []
    print(f"[Corpus] Extracting {n} drug-like SMILES from ChEMBL SDF...")
    try:
        supplier = Chem.ForwardSDMolSupplier(sdf_path, removeHs=True, sanitize=True)
        scanned = 0
        for mol in supplier:
            if len(smiles_list) >= n:
                break
            scanned += 1
            if scanned % 100_000 == 0:
                print(f"  {scanned:,} molecules scanned | {len(smiles_list):,} accepted")
            if mol is None:
                continue
            n_heavy = mol.GetNumHeavyAtoms()
            if not (min_heavy <= n_heavy <= max_heavy):
                continue
            atom_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
            if atom_nums & FORBIDDEN_ATOMS:
                continue
            if 6 not in atom_nums:
                continue
            try:
                qed_val = rdQED.qed(mol)
                if qed_val < 0.3:
                    continue
            except Exception:
                continue
            smi = Chem.MolToSmiles(mol)
            if smi:
                smiles_list.append(smi)
    except Exception as e:
        print(f"[Corpus] SDF read error: {e} — using seed SMILES.")
        return SEED_SMILES

    print(f"[Corpus] {len(smiles_list)} SMILES extracted ({scanned:,} scanned)")
    if len(smiles_list) < 100:
        smiles_list = smiles_list + SEED_SMILES
    return smiles_list


# ─── BRICS Hyperparameters ────────────────────────────────────────────────────
BRICS_DQN_HP = dict(
    replay_buffer_size  = 20_000,
    batch_size          = 64,
    gamma               = 0.99,
    lr                  = 3e-4,
    eps_start           = 1.0,
    eps_end             = 0.15,
    eps_decay_steps     = 15_000,
    target_update_freq  = 200,
    max_fragments       = 8,
    n_episodes          = 5_000,
    warmstart_episodes  = 500,
    hidden_dim          = 256,
    target_ic50         = -1.5,
    # Positive rewards
    qed_weight          = 3.0,
    logp_weight         = 0.5,
    lipinski_bonus      = 1.0,
    ic50_weight         = 0.8,
    diversity_weight    = 0.4,
    arom_bonus          = 1.2,
    # BRICS-specific rewards
    brics_success_bonus = 0.3,
    frag_diversity_coef = 0.2,
    # Chemical penalties
    carbon_penalty      = -0.5,
    min_carbon_frac     = 0.25,
    size_penalty_coef   = 0.05,
    max_heavy_atoms     = 30,
    repeat_penalty_coef = 0.3,
    max_token_repeat    = 4,
    stereo_penalty_coef = 0.15,
    max_stereo_centers  = 4,
    # Drug-likeness penalties
    charge_penalty_coef = 0.4,
    alkyne_penalty_coef = 0.5,
    max_alkynes         = 0,
    nonarom_penalty     = -1.0,
    log_interval        = 50,
)


# ─── BRICS Vocabulary ─────────────────────────────────────────────────────────
class BRICSVocabulary:
    PAD_TOKEN   = "[PAD]"
    EOS_TOKEN   = "[EOS]"
    START_TOKEN = "[START]"

    def __init__(self, smiles_list: List[str], min_freq: int = 5):
        freq: collections.Counter = collections.Counter()
        n_mol_processed = 0

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICSDecompose(mol)
                for f in frags:
                    freq[f] += 1
                n_mol_processed += 1
            except Exception:
                pass

        # Keep fragments meeting frequency threshold
        common_frags = {f for f, c in freq.items() if c >= min_freq}

        # Also decompose SEED_SMILES and add all their fragments regardless of freq
        for smi in SEED_SMILES:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICSDecompose(mol)
                common_frags.update(frags)
            except Exception:
                pass

        # Special tokens: idx 0=PAD, 1=EOS, 2=START
        self.idx2frag: List[str] = [self.PAD_TOKEN, self.EOS_TOKEN, self.START_TOKEN]
        self.idx2frag += sorted(common_frags)
        self.frag2idx = {f: i for i, f in enumerate(self.idx2frag)}

        self.PAD_IDX    = 0
        self.EOS_IDX    = 1
        self.START_IDX  = 2
        self.vocab_size = len(self.idx2frag)

        print(f"[BRICS Vocab] {self.vocab_size} fragments | "
              f"built from {n_mol_processed} molecules | min_freq={min_freq}")

    def smiles_to_fragments(self, smiles: str) -> List[str]:
        """Decompose a SMILES string into BRICS fragments."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        try:
            frags = BRICSDecompose(mol)
            return list(frags)
        except Exception:
            return []

    def fragments_to_smiles(self, fragments: List[str]) -> Tuple[str, bool]:
        """
        Assemble fragments into a SMILES string.
        Tries BRICSBuild first; falls back to combining via Chem.CombineMols.
        Returns (smiles, brics_success).
        """
        if not fragments:
            return "", False

        # Convert fragment SMILES strings to Mol objects
        frag_mols = []
        for f in fragments:
            m = Chem.MolFromSmiles(f)
            if m is not None:
                frag_mols.append(m)
        if not frag_mols:
            return "", False

        # Attempt BRICS assembly
        try:
            gen = BRICSBuild(frag_mols)
            result_mol = next(iter(gen), None)
            if result_mol is not None:
                try:
                    Chem.SanitizeMol(result_mol)
                    smi = Chem.MolToSmiles(result_mol)
                    if smi:
                        return smi, True
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback: combine mols (may produce disconnected SMILES with '.')
        try:
            combined = frag_mols[0]
            for m in frag_mols[1:]:
                combined = Chem.CombineMols(combined, m)
            smi = Chem.MolToSmiles(combined)
            return smi if smi else "", False
        except Exception:
            return "", False

    def encode(self, fragments: List[str]) -> List[int]:
        return [self.frag2idx.get(f, self.PAD_IDX) for f in fragments]

    def decode(self, indices: List[int]) -> List[str]:
        result = []
        for idx in indices:
            if idx == self.EOS_IDX:
                break
            if idx in (self.PAD_IDX, self.START_IDX):
                continue
            if 0 <= idx < self.vocab_size:
                result.append(self.idx2frag[idx])
        return result

    def random_fragment(self) -> int:
        """Return a random fragment index (excludes PAD, EOS, START)."""
        return random.randint(3, self.vocab_size - 1)


# ─── Replay Buffer ────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, n: int) -> List[Transition]:
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


# ─── Q-Network ────────────────────────────────────────────────────────────────
class QNetwork(keras.Model):
    def __init__(self, state_dim: int, vocab_size: int,
                 hidden_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu", input_shape=(state_dim,)),
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(hidden_dim // 2, activation="relu"),
            layers.Dense(vocab_size),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)


# ─── BRICS RL Environment ─────────────────────────────────────────────────────
class BRICSEnv:
    """
    RL environment for BRICS fragment assembly.
    State: z_omics (128-dim) || one-hot(last_fragment_idx) in R^(128 + vocab_size)
    Action: fragment index
    Episode: terminal on EOS or max_fragments steps
    """

    _brics_fail_count = 0  # class-level counter for throttled logging

    def __init__(self, twin, feat, vocab: BRICSVocabulary,
                 z_omics: tf.Tensor, hp: dict = BRICS_DQN_HP,
                 past_fps: List = None):
        self.twin      = twin
        self.feat      = feat
        self.vocab     = vocab
        self.hp        = hp
        self.past_fps  = past_fps or []
        self.max_frags = hp["max_fragments"]
        self._z_np     = z_omics.numpy().flatten()
        self.fragment_indices: List[int] = []
        self.step_count = 0
        self._last_brics_success = False

    def _state(self, frag_idx: int) -> np.ndarray:
        oh = np.zeros(self.vocab.vocab_size, dtype=np.float32)
        oh[min(frag_idx, self.vocab.vocab_size - 1)] = 1.0
        return np.concatenate([self._z_np, oh])

    def reset(self) -> np.ndarray:
        self.fragment_indices = []
        self.step_count = 0
        self._last_brics_success = False
        return self._state(self.vocab.START_IDX)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.fragment_indices.append(action)
        self.step_count += 1
        done = (action == self.vocab.EOS_IDX) or (self.step_count >= self.max_frags)
        if done:
            reward = self._compute_reward()
        else:
            reward = 0.0
        return self._state(action), reward, done

    def _compute_reward(self) -> float:
        # Decode fragment indices to fragment SMILES strings
        current_fragments = self.vocab.decode(self.fragment_indices)
        if not current_fragments:
            return -0.5

        smiles, brics_ok = self.vocab.fragments_to_smiles(current_fragments)
        self._last_brics_success = brics_ok

        if not smiles:
            BRICSEnv._brics_fail_count += 1
            if BRICSEnv._brics_fail_count % 500 == 1:
                print(f"[BRICS] Build failures: {BRICSEnv._brics_fail_count} "
                      f"(logging every 500)")
            return -0.5

        # Reject disconnected fragments (BRICS fallback produced dot notation)
        if '.' in smiles:
            return -1.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -0.5

        # Forbidden atoms: F(9), Cl(17), Br(35), I(53)
        _FORBIDDEN_ATOMS = {9, 17, 35, 53}
        atom_nums_set = {a.GetAtomicNum() for a in mol.GetAtoms()}
        if atom_nums_set & _FORBIDDEN_ATOMS:
            return -1.0

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 5:
            return -0.2

        penalties = 0.0
        atom_nums_list = [a.GetAtomicNum() for a in mol.GetAtoms()]

        # Carbon fraction
        n_carbon = atom_nums_list.count(6)
        if n_carbon == 0 or (n_carbon / n_heavy) < self.hp["min_carbon_frac"]:
            penalties += self.hp["carbon_penalty"]

        # Excessive size
        if n_heavy > self.hp["max_heavy_atoms"]:
            penalties -= (n_heavy - self.hp["max_heavy_atoms"]) * self.hp["size_penalty_coef"]

        # Formal charges
        charged_atoms = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
        if charged_atoms > 0:
            penalties -= charged_atoms * self.hp["charge_penalty_coef"]

        # Alkynes (C#C)
        n_alkynes = sum(
            1 for b in mol.GetBonds()
            if b.GetBondTypeAsDouble() == 3.0
            and b.GetBeginAtom().GetAtomicNum() == 6
            and b.GetEndAtom().GetAtomicNum() == 6
        )
        if n_alkynes > self.hp["max_alkynes"]:
            penalties -= (n_alkynes - self.hp["max_alkynes"]) * self.hp["alkyne_penalty_coef"]

        # Fragment-level repeat penalty (same fragment chosen multiple times)
        frag_counts = collections.Counter(self.fragment_indices)
        max_rep = max(frag_counts.values()) if frag_counts else 0
        if max_rep > self.hp["max_token_repeat"]:
            penalties -= (max_rep - self.hp["max_token_repeat"]) * self.hp["repeat_penalty_coef"]

        # Stereo centers
        try:
            stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            n_stereo = len(stereo_centers)
            if n_stereo > self.hp["max_stereo_centers"]:
                penalties -= (n_stereo - self.hp["max_stereo_centers"]) * self.hp["stereo_penalty_coef"]
        except Exception:
            pass

        # No aromatic ring penalty
        try:
            n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            if n_arom_rings == 0:
                penalties += self.hp["nonarom_penalty"]
        except Exception:
            pass

        reward = 0.0

        # QED
        try:
            reward += self.hp["qed_weight"] * rdQED.qed(mol)
        except Exception:
            pass

        # LogP gaussian centered on 2
        try:
            logp = Descriptors.MolLogP(mol)
            reward += self.hp["logp_weight"] * float(np.exp(-((logp - 2.0) ** 2) / 4.0))
        except Exception:
            pass

        # Aromatic ring bonus
        try:
            n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
            reward += self.hp["arom_bonus"] * min(n_arom, 3) / 3.0
        except Exception:
            pass

        # Lipinski Rule of 5
        try:
            mw  = Descriptors.MolWt(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            lp  = Descriptors.MolLogP(mol)
            if mw <= 500 and hbd <= 5 and hba <= 10 and lp <= 5:
                reward += self.hp["lipinski_bonus"]
        except Exception:
            pass

        # IC50 prediction
        try:
            af  = self.feat.featurize(smiles)[np.newaxis]
            adj = np.ones((1, HP["max_atoms"], HP["max_atoms"]), dtype=np.float32)
            inp = (tf.constant(af), tf.constant(adj),
                   tf.zeros([1, HP["gex_dim"]]),
                   tf.zeros([1, HP["mut_dim"]]),
                   tf.zeros([1, HP["cnv_dim"]]))
            ic50_val = float(self.twin(inp, training=False)[0][0].numpy())
            reward += self.hp["ic50_weight"] * float(
                np.exp(-((ic50_val - self.hp["target_ic50"]) ** 2) / 2.0))
        except Exception:
            pass

        # Tanimoto diversity vs past molecules
        if self.past_fps:
            try:
                fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                sim = DataStructs.BulkTanimotoSimilarity(fp, self.past_fps)
                reward += (1.0 - max(sim)) * self.hp["diversity_weight"]
            except Exception:
                pass

        # BRICS-specific bonuses
        if brics_ok:
            reward += self.hp["brics_success_bonus"]

        n_total = len(self.fragment_indices)
        n_unique = len(set(self.fragment_indices))
        if n_total > 0:
            reward += self.hp["frag_diversity_coef"] * (n_unique / n_total)

        return float(np.clip(reward + penalties, -1.0, 10.0))

    @property
    def current_smiles(self) -> str:
        frags = self.vocab.decode(self.fragment_indices)
        smi, _ = self.vocab.fragments_to_smiles(frags)
        return smi

    @property
    def last_brics_success(self) -> bool:
        return self._last_brics_success


# ─── BRICS DQN Agent ─────────────────────────────────────────────────────────
class BRICSDQNOptimizer:
    """
    DQN optimizer using BRICS fragment assembly.
    Same training loop as DQNDrugOptimizer but with BRICS vocabulary/environment.
    """

    def __init__(self, twin, feat, vocab: BRICSVocabulary,
                 hp: dict = BRICS_DQN_HP):
        self.twin   = twin
        self.feat   = feat
        self.vocab  = vocab
        self.hp     = hp

        state_dim  = HP["latent_dim"] + vocab.vocab_size
        vocab_size = vocab.vocab_size

        self.q_online = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_online")
        self.q_target = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_target")
        self._sync_target()

        self.optimizer    = keras.optimizers.Adam(hp["lr"])
        self.replay       = ReplayBuffer(hp["replay_buffer_size"])
        self.loss_fn      = keras.losses.Huber()
        self.global_step  = 0
        self.best_smiles  = ""
        self.best_reward  = -float("inf")
        self.past_fps: List = []

        print(f"[BRICS-DQN] state_dim={state_dim} | vocab_size={vocab_size}")

    def _warmstart_buffer(self, z: tf.Tensor, seed_smiles: List[str]):
        """
        Pre-fill the replay buffer with expert trajectories by decomposing
        SEED_SMILES into BRICS fragments and replaying the assembly sequence.
        """
        n_ws = self.hp.get("warmstart_episodes", 500)
        valid_pool = [s for s in seed_smiles if Chem.MolFromSmiles(s) is not None]
        if not valid_pool:
            return

        injected = 0
        for _ in range(n_ws):
            smi = random.choice(valid_pool)
            frags = self.vocab.smiles_to_fragments(smi)
            if not frags:
                continue
            # Keep only fragments known in the vocabulary
            frag_ids = [self.vocab.frag2idx[f] for f in frags
                        if f in self.vocab.frag2idx]
            if not frag_ids:
                continue
            frag_ids.append(self.vocab.EOS_IDX)

            env   = BRICSEnv(self.twin, self.feat, self.vocab, z,
                             hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            for action in frag_ids:
                next_state, reward, done = env.step(action)
                self.replay.push(state, action, np.float32(reward),
                                 next_state, np.float32(done))
                state = next_state
                if done:
                    break
            injected += 1

        print(f"[BRICS-DQN] Warm-start: {injected} expert trajectories injected "
              f"({len(self.replay)} transitions in buffer)")

    def _sync_target(self):
        self.q_target.set_weights(self.q_online.get_weights())

    def _epsilon(self) -> float:
        t = min(self.global_step, self.hp["eps_decay_steps"])
        return self.hp["eps_start"] + t / self.hp["eps_decay_steps"] * (
            self.hp["eps_end"] - self.hp["eps_start"])

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self._epsilon():
            return self.vocab.random_fragment()
        q = self.q_online(
            tf.expand_dims(tf.constant(state, dtype=tf.float32), 0),
            training=False).numpy()[0]
        q[self.vocab.PAD_IDX]   = -np.inf
        q[self.vocab.START_IDX] = -np.inf
        return int(np.argmax(q))

    @tf.function
    def _update_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            idx    = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa   = tf.gather_nd(self.q_online(states, training=True), idx)
            ba     = tf.argmax(self.q_online(next_states, training=False),
                               axis=1, output_type=tf.int32)
            idx_n  = tf.stack([tf.range(tf.shape(ba)[0]), ba], axis=1)
            q_next = tf.gather_nd(self.q_target(next_states, training=False), idx_n)
            target = rewards + self.hp["gamma"] * q_next * (1.0 - dones)
            loss   = self.loss_fn(tf.stop_gradient(target), q_sa)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.q_online.trainable_variables),
                self.q_online.trainable_variables))
        return loss

    def _learn(self):
        if len(self.replay) < self.hp["batch_size"]:
            return None
        b = self.replay.sample(self.hp["batch_size"])
        return self._update_step(
            tf.constant(np.stack([t.state      for t in b]), dtype=tf.float32),
            tf.constant(np.array( [t.action     for t in b]), dtype=tf.int32),
            tf.constant(np.array( [t.reward     for t in b]), dtype=tf.float32),
            tf.constant(np.stack([t.next_state for t in b]), dtype=tf.float32),
            tf.constant(np.array( [t.done       for t in b], dtype=np.float32)),
        )

    def optimize(self, gex, mut, cnv, n_episodes: int = None,
                 seed_smiles: List[str] = None) -> dict:
        n_episodes = n_episodes or self.hp["n_episodes"]
        z, _, _    = self.twin.omics_vae((gex, mut, cnv), training=False)

        if seed_smiles:
            self._warmstart_buffer(z, seed_smiles)

        rewards_hist, valid_count, top_mols = [], 0, []
        episode_log = []   # for CSV export

        print(f"\n[BRICS-DQN] {n_episodes} episodes | vocab={self.vocab.vocab_size} fragments")
        print(f"  eps: {self.hp['eps_start']} -> {self.hp['eps_end']} / "
              f"{self.hp['eps_decay_steps']} steps")
        print(f"  max_fragments={self.hp['max_fragments']} | "
              f"warmstart={self.hp.get('warmstart_episodes', 0)}\n")

        for ep in range(1, n_episodes + 1):
            env   = BRICSEnv(self.twin, self.feat, self.vocab, z,
                             hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            ep_r, ep_loss = 0.0, []

            while True:
                action              = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.replay.push(state, action, np.float32(reward),
                                 next_state, np.float32(done))
                ep_r = reward if done else ep_r
                state = next_state
                self.global_step += 1
                loss = self._learn()
                if loss is not None:
                    ep_loss.append(float(loss.numpy()))
                if self.global_step % self.hp["target_update_freq"] == 0:
                    self._sync_target()
                if done:
                    break

            smi = env.current_smiles
            brics_ok = env.last_brics_success
            n_frags = len([i for i in env.fragment_indices
                           if i not in (self.vocab.PAD_IDX,
                                        self.vocab.EOS_IDX,
                                        self.vocab.START_IDX)])
            rewards_hist.append(ep_r)
            is_valid = ep_r > -0.4 and bool(smi)

            episode_log.append({
                "episode":       ep,
                "reward":        ep_r,
                "valid":         is_valid,
                "smiles":        smi,
                "brics_success": brics_ok,
                "n_fragments":   n_frags,
            })

            if is_valid:
                valid_count += 1
                top_mols.append((ep_r, smi))
                top_mols.sort(key=lambda x: -x[0])
                top_mols = top_mols[:10]

            if ep_r > self.best_reward and smi:
                self.best_reward, self.best_smiles = ep_r, smi
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        self.past_fps.append(
                            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                except Exception:
                    pass

            if ep % self.hp["log_interval"] == 0 or ep == 1:
                mean_r = float(np.mean(rewards_hist[-50:]))
                mean_l = float(np.mean(ep_loss)) if ep_loss else float("nan")
                print(f"  Ep {ep:5d}/{n_episodes} | eps={self._epsilon():.3f} | "
                      f"R={ep_r:+.3f} | Avg50={mean_r:+.3f} | "
                      f"Valid={100*valid_count/ep:.1f}% | Loss={mean_l:.4f} | "
                      f"Best: {self.best_smiles[:40]!s:40s} ({self.best_reward:.3f})")

        print(f"\n[BRICS-DQN] Done.")
        print(f"  Best SMILES  : {self.best_smiles}")
        print(f"  Best reward  : {self.best_reward:.4f}")
        print(f"  Valid        : {valid_count}/{n_episodes} "
              f"({100*valid_count/n_episodes:.1f}%)")
        if top_mols:
            print(f"\n  Top 5:")
            for i, (r, s) in enumerate(top_mols[:5], 1):
                print(f"    {i}. R={r:.3f}  {s}")

        return dict(
            best_smiles      = self.best_smiles,
            best_reward      = self.best_reward,
            reward_trajectory= rewards_hist,
            valid_count      = valid_count,
            top_molecules    = top_mols,
            episode_log      = episode_log,
        )

    def save_results_csv(self, episode_log: List[dict],
                         path: str = None):
        if path is None:
            path = os.path.join(_ROOT, "Dataset", "brics_dqn_results.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = ["episode", "reward", "valid", "smiles",
                      "brics_success", "n_fragments"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_log)
        print(f"[BRICS-DQN] Results saved to {path}")

    def save_weights(self, path: str = "dqn_weights_brics"):
        os.makedirs(path, exist_ok=True)
        self.q_online.save_weights(os.path.join(path, "q_online.weights.h5"))
        self.q_target.save_weights(os.path.join(path, "q_target.weights.h5"))
        print(f"[BRICS-DQN] Weights -> {path}/")

    def load_weights(self, path: str = "dqn_weights_brics"):
        self.q_online.load_weights(os.path.join(path, "q_online.weights.h5"))
        self._sync_target()


# ─── Main entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  BRICS DQN Drug Optimizer v1.0 — Fragment Assembly + ChEMBL 10k")
    print("=" * 70)

    # 1. Load ChEMBL corpus
    corpus = load_chembl_smiles(10_000)

    # 2. Build BRICS vocabulary
    vocab = BRICSVocabulary(corpus)

    # Vocab sanity check: decompose aspirin and reassemble
    test_smi = "CC(=O)Oc1ccccc1C(=O)O"
    test_frags = vocab.smiles_to_fragments(test_smi)
    test_rebuilt, ok = vocab.fragments_to_smiles(test_frags)
    print(f"[Vocab test] Aspirin -> {len(test_frags)} fragments -> "
          f"rebuilt={test_rebuilt!r} (brics_ok={ok})")

    # 3. Load digital twin model
    dt_model   = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()

    # 4. Synthetic omics data (replace with real profile from fullPipeline)
    gex = tf.random.normal([1, HP["gex_dim"]])
    mut = tf.cast(tf.random.uniform([1, HP["mut_dim"]], 0, 2, dtype=tf.int32),
                  tf.float32)
    cnv = tf.random.normal([1, HP["cnv_dim"]])

    # 5. Run BRICS DQN optimization with warm-start
    agent  = BRICSDQNOptimizer(dt_model, featurizer, vocab, BRICS_DQN_HP)
    result = agent.optimize(gex, mut, cnv, seed_smiles=SEED_SMILES)

    # 6. Print top 5 molecules
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Best SMILES : {result['best_smiles']}")
    print(f"  Best reward : {result['best_reward']:.4f}")
    print(f"  Valid       : {result['valid_count']}/{BRICS_DQN_HP['n_episodes']} "
          f"({100*result['valid_count']/BRICS_DQN_HP['n_episodes']:.1f}%)")
    if result["top_molecules"]:
        print(f"\n  Top 5:")
        for i, (r, s) in enumerate(result["top_molecules"][:5], 1):
            print(f"    {i}. R={r:.3f}  {s}")

    # 7. Save results CSV and model weights
    agent.save_results_csv(result["episode_log"])
    agent.save_weights("dqn_weights_brics")

    # ── Comparison notes ──────────────────────────────────────────────────────
    # vs SELFIES DQN (dqn_optimizer.py v5.1):
    #   SELFIES: token-by-token (up to 20 tokens), vocab ~150 tokens
    #   BRICS  : fragment-by-fragment (up to 8 fragments), vocab ~several hundred frags
    #   BRICS advantage: each action encodes chemical knowledge (valid substructures)
    #   BRICS advantage: assembly guarantees valence-correct bond attachment points
    #   BRICS advantage: shorter episodes -> faster convergence
    #   BRICS limitation: BRICSBuild stochastic -> same fragments may yield diff mols
    #   BRICS limitation: vocabulary biased toward ChEMBL fragment distribution
