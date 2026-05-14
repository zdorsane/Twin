"""
================================================================================
  DQN Drug Optimizer — SELFIES-based (v3)  — Bi-Int Digital Twin
================================================================================

Architecture :
  - Espace d'action  : tokens SELFIES (Self-Referencing Embedded Strings)
                       → 100% des séquences produisent des molécules valides
  - Etat (state)     : z_omics (OmicsVAE latent) ‖ one-hot(dernier token)
  - Récompense       : QED + LogP + Lipinski + IC50 + diversité Tanimoto
  - Pas de masking   : SELFIES est grammaticalement clos par construction

Références :
  Krenn et al., "Self-Referencing Embedded Strings (SELFIES)", Machine Learning:
    Science and Technology, 2020.
  Mnih et al., "Human-level control through deep RL", Nature 2015.
================================================================================
"""

import os
import sys
import random
import warnings
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Deque

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/crbt/Twin")

from fullPipeline import (
    BiIntDigitalTwin,
    BRICSMolecularFeaturizer,
    HP,
)

try:
    import selfies as sf
    HAS_SELFIES = True
except ImportError:
    HAS_SELFIES = False
    print("[ERREUR] selfies non installé — pip install selfies")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors, QED as rdQED
    from rdkit.Chem import rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit non disponible — récompenses chimiques désactivées.")


# ─── Corpus de départ : médicaments approuvés FDA (SMILES) ───────────────────
SEED_SMILES = [
    # Petites molécules approuvées FDA (diversité chimique)
    "CC(=O)Oc1ccccc1C(=O)O",                     # Aspirine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                # Ibuprofène
    "CN1CCC[C@H]1c2cccnc2",                       # Nicotine
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",           # Pyrène
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",       # Testostérone
    "CN(C)CCOC(c1ccccc1)c2ccccc2",                # Diphenhydramine
    "O=C(O)c1ccccc1O",                            # Acide salicylique
    "Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)CCOCCO",   # Hydroxyzine
    "CC(=O)Nc1ccc(O)cc1",                         # Paracétamol
    "Oc1ccc(cc1)C2CC(=O)c3c(O)cc(O)cc3O2",       # Naringénine
    "c1ccc2ncccc2c1",                             # Quinoline
    "O=C(Nc1ccc(Cl)c(Cl)c1)N2CCC(CC2)N3CCOCC3", # Clozapine analogue
    "Fc1ccc(cc1)C(=O)CCCN2CCC(CC2)c3noc4cc(F)ccc34", # Haloperidol
    "CC(=O)[C@@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]12C", # Progestérone
    "CC1=CC2=C(C=C1C)N(C=N2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",       # Riboflavine (partiel)
    "c1ccc(cc1)CN2CCN(CC2)c3cccc(c3)Cl",         # Clomethiazole analogue
    "O=C1CCCN1",                                  # Pyrrolidinone
    "c1ccc(cc1)c2cc(nn2c3ccccc3)C(F)(F)F",       # Célécoxib analogue
    "CC1(C)OCC(O1)CN2C=NC3=CC=CC=C23",           # Noscapine analogue
    "CC(C)(C)c1ccc(cc1)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CC(C)C)CC(=O)O",  # Peptidomimétique
    "O=C(O)c1ccc2c(c1)N=C(c3ccccc23)c4ccccc4",  # Acridine
    "CCOC(=O)c1cnc(N)c(Cl)c1F",                  # Fluoroquinolone précurseur
    "CC12CC(=O)C3C(C1CC(O2)(CC3=O)C)C",          # Stéroïde synthétique
    "c1ccc(cc1)S(=O)(=O)Nc2ccc(cc2)N",           # Sulfanilamide
]


# ─── Hyper-paramètres DQN ─────────────────────────────────────────────────────
DQN_HP = dict(
    replay_buffer_size = 20_000,
    batch_size         = 64,
    gamma              = 0.99,
    lr                 = 3e-4,
    eps_start          = 1.0,
    eps_end            = 0.05,
    eps_decay_steps    = 8_000,
    target_update_freq = 200,
    max_selfies_len    = 35,        # tokens SELFIES (≈ 35 tokens ≈ 30 atomes lourds)
    n_episodes         = 2_000,
    hidden_dim         = 256,
    target_ic50        = -1.5,      # IC50 normalisé cible
    # Récompenses
    qed_weight         = 2.5,
    logp_weight        = 0.5,
    lipinski_bonus     = 0.8,
    ic50_weight        = 0.8,
    diversity_weight   = 0.4,
    log_interval       = 50,
)


# ─── Vocabulaire SELFIES ──────────────────────────────────────────────────────
class SELFIESVocabulary:
    """
    Construit un vocabulaire SELFIES à partir d'un corpus de SMILES.
    Chaque token SELFIES est un symbole atomique ou de liaison entre crochets.
    """

    PAD_TOKEN   = "[nop]"   # token de remplissage (no-op en SELFIES)
    START_TOKEN = "[nop]"   # début d'épisode = no-op
    END_TOKEN   = "[EOS]"   # token de fin personnalisé

    def __init__(self, smiles_list: List[str]):
        selfies_list = []
        for smi in smiles_list:
            try:
                sel = sf.encoder(smi)
                if sel:
                    selfies_list.append(sel)
            except Exception:
                pass

        # Collecter tous les tokens uniques
        token_set = set()
        for sel in selfies_list:
            for tok in sf.split_selfies(sel):
                token_set.add(tok)

        # Garder uniquement les tokens drug-like (C, N, O, S, F, Cl, Br, cycles, liaisons)
        DRUG_LIKE_ATOMS = {
            "[C]", "[=C]", "[#C]", "[C-1]", "[C@@Hexpl]", "[C@Hexpl]",
            "[N]", "[=N]", "[#N]", "[N+1]", "[NH1]", "[NH2]",
            "[O]", "[=O]", "[OH1]",
            "[S]", "[=S]", "[SH1]",
            "[F]", "[Cl]", "[Br]", "[I]",
            "[Ring1]", "[Ring2]", "[Branch1]", "[Branch2]",
            "[=Branch1]", "[=Branch2]", "[#Branch1]", "[#Branch2]",
            "[=Ring1]", "[=Ring2]", "[#Ring1]", "[#Ring2]",
            "[nop]",
        }
        # Ajouter les tokens des SMILES du corpus (déjà filtrés)
        # + tokens drug-like de l'alphabet standard
        try:
            alphabet = list(sf.get_semantic_robust_alphabet())
            for tok in alphabet:
                # Inclure seulement C, N, O, S, F, Cl, Br, liaisons structurales
                if any(atom in tok for atom in ["C", "N", "O", "S", "F", "l", "r", "Ring", "Branch", "nop"]):
                    if not any(exotic in tok for exotic in ["Si", "Se", "Te", "At", "Xe", "Kr", "Rn", "Sn", "Pb", "As", "Ge", "B]", "[B", "P]", "[P"]):
                        token_set.add(tok)
        except Exception:
            pass

        token_set.discard(self.END_TOKEN)
        token_set.discard(self.PAD_TOKEN)

        # Ordre déterministe
        sorted_tokens = sorted(token_set)
        self.idx2tok = [self.PAD_TOKEN, self.END_TOKEN] + sorted_tokens
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

        self.PAD_IDX   = 0
        self.END_IDX   = 1
        self.vocab_size = len(self.idx2tok)

        print(f"[SELFIES Vocab] {self.vocab_size} tokens | "
              f"corpus: {len(selfies_list)} molécules valides / {len(smiles_list)} SMILES")

    def encode(self, selfies_str: str) -> List[int]:
        tokens = list(sf.split_selfies(selfies_str))
        return [self.tok2idx.get(t, self.PAD_IDX) for t in tokens] + [self.END_IDX]

    def decode(self, indices: List[int]) -> str:
        """Convertit une liste d'indices en SELFIES puis en SMILES."""
        tokens = []
        for idx in indices:
            if idx == self.END_IDX:
                break
            if idx == self.PAD_IDX:
                continue
            tok = self.idx2tok[idx]
            if tok not in (self.PAD_TOKEN, self.END_TOKEN):
                tokens.append(tok)
        if not tokens:
            return ""
        selfies_str = "".join(tokens)
        try:
            smiles = sf.decoder(selfies_str)
            return smiles or ""
        except Exception:
            return ""

    def random_token(self) -> int:
        """Retourne un token aléatoire hors PAD/END."""
        return random.randint(2, self.vocab_size - 1)


# ─── Replay Buffer ────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─── Q-Network ────────────────────────────────────────────────────────────────
class QNetwork(keras.Model):
    def __init__(self, state_dim: int, vocab_size: int, hidden_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu", input_shape=(state_dim,)),
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(hidden_dim // 2, activation="relu"),
            layers.Dense(vocab_size),
        ])

    def call(self, state, training=False):
        return self.net(state, training=training)


# ─── Environnement SELFIES ────────────────────────────────────────────────────
class SELFIESEnv:
    """
    Environnement RL où chaque action ajoute un token SELFIES.
    Toute séquence SELFIES → molécule valide (propriété de SELFIES).
    """

    def __init__(
        self,
        digital_twin: BiIntDigitalTwin,
        featurizer: BRICSMolecularFeaturizer,
        vocab: SELFIESVocabulary,
        z_omics: tf.Tensor,
        hp: dict = DQN_HP,
        past_fps: List = None,
    ):
        self.twin     = digital_twin
        self.feat     = featurizer
        self.vocab    = vocab
        self.z        = z_omics
        self.hp       = hp
        self.past_fps = past_fps or []
        self.max_len  = hp["max_selfies_len"]

        self.tokens: List[int] = []
        self.step_count = 0

        self._z_np = z_omics.numpy().flatten()
        self._state_dim = len(self._z_np) + vocab.vocab_size

    def _build_state(self, last_token: int) -> np.ndarray:
        tok_onehot = np.zeros(self.vocab.vocab_size, dtype=np.float32)
        tok_onehot[min(last_token, self.vocab.vocab_size - 1)] = 1.0
        return np.concatenate([self._z_np, tok_onehot], axis=0).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.tokens     = []
        self.step_count = 0
        return self._build_state(self.vocab.PAD_IDX)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.tokens.append(action)
        self.step_count += 1

        done = (action == self.vocab.END_IDX) or (self.step_count >= self.max_len)

        reward = 0.0
        if done:
            reward = self._compute_reward()

        next_state = self._build_state(action)
        return next_state, reward, done

    def _compute_reward(self) -> float:
        smiles = self.vocab.decode(self.tokens)
        if not smiles:
            return -0.5

        mol = None
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                pass

        if mol is None:
            return -0.5

        # Filtre : au moins 5 atomes lourds
        if mol.GetNumHeavyAtoms() < 5:
            return -0.2

        reward = 0.0

        # QED — drug-likeness [0, 1]
        try:
            qed_val = rdQED.qed(mol)
            reward += self.hp["qed_weight"] * qed_val
        except Exception:
            pass

        # LogP — fenêtre idéale [1, 3]
        try:
            logp = Descriptors.MolLogP(mol)
            logp_rew = float(np.exp(-((logp - 2.0) ** 2) / 4.0))
            reward += self.hp["logp_weight"] * logp_rew
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

        # IC50 prédit par le Digital Twin
        try:
            atom_feat = self.feat.featurize(smiles)[np.newaxis]
            adj       = np.ones((1, HP["max_atoms"], HP["max_atoms"]), dtype=np.float32)
            gex_d     = tf.zeros([1, HP["gex_dim"]])
            mut_d     = tf.zeros([1, HP["mut_dim"]])
            cnv_d     = tf.zeros([1, HP["cnv_dim"]])
            inputs    = (tf.constant(atom_feat), tf.constant(adj), gex_d, mut_d, cnv_d)
            ic50_pred, _ = self.twin(inputs, training=False)
            ic50_val  = float(ic50_pred[0].numpy())
            ic50_rew  = float(np.exp(-((ic50_val - self.hp["target_ic50"]) ** 2) / 2.0))
            reward   += self.hp["ic50_weight"] * ic50_rew
        except Exception:
            pass

        # Diversité Tanimoto (encourage l'exploration)
        if HAS_RDKIT and self.past_fps and self.hp["diversity_weight"] > 0:
            try:
                fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                sim = DataStructs.BulkTanimotoSimilarity(fp, self.past_fps)
                max_sim = max(sim) if sim else 0.0
                reward += (1.0 - max_sim) * self.hp["diversity_weight"]
            except Exception:
                pass

        return float(np.clip(reward, -1.0, 10.0))

    @property
    def current_smiles(self) -> str:
        return self.vocab.decode(self.tokens)


# ─── Agent DQN ────────────────────────────────────────────────────────────────
class DQNDrugOptimizer:
    def __init__(
        self,
        digital_twin: BiIntDigitalTwin,
        featurizer: BRICSMolecularFeaturizer,
        vocab: SELFIESVocabulary,
        hp: dict = DQN_HP,
    ):
        self.twin  = digital_twin
        self.feat  = featurizer
        self.vocab = vocab
        self.hp    = hp

        latent_dim = HP["latent_dim"]
        vocab_size = vocab.vocab_size
        state_dim  = latent_dim + vocab_size

        self.q_online = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_online")
        self.q_target = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_target")
        self._sync_target()

        self.optimizer = keras.optimizers.Adam(hp["lr"])
        self.replay    = ReplayBuffer(hp["replay_buffer_size"])
        self.loss_fn   = keras.losses.Huber()

        self.global_step = 0
        self.best_smiles = ""
        self.best_reward = -float("inf")
        self.past_fps: List = []

        print(f"[DQN-SELFIES] Initialisé | state_dim={state_dim} | vocab_size={vocab_size}")

    def _sync_target(self):
        self.q_target.set_weights(self.q_online.get_weights())

    def _epsilon(self) -> float:
        t     = min(self.global_step, self.hp["eps_decay_steps"])
        ratio = t / self.hp["eps_decay_steps"]
        return self.hp["eps_start"] + ratio * (self.hp["eps_end"] - self.hp["eps_start"])

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self._epsilon():
            # Exploration : token aléatoire (PAD et END exclus au début)
            return self.vocab.random_token()
        state_t  = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
        q_values = self.q_online(state_t, training=False).numpy()[0]
        # Décourager PAD (index 0) mais autoriser END
        q_values[self.vocab.PAD_IDX] = -np.inf
        return int(np.argmax(q_values))

    @tf.function
    def _update_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_all = self.q_online(states, training=True)
            idx   = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa  = tf.gather_nd(q_all, idx)

            q_next_online = self.q_online(next_states, training=False)
            best_actions  = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
            q_next_target = self.q_target(next_states, training=False)
            idx_next      = tf.stack([tf.range(tf.shape(best_actions)[0]), best_actions], axis=1)
            q_next_best   = tf.gather_nd(q_next_target, idx_next)

            target = rewards + self.hp["gamma"] * q_next_best * (1.0 - dones)
            loss   = self.loss_fn(tf.stop_gradient(target), q_sa)

        grads = tape.gradient(loss, self.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_online.trainable_variables))
        return loss

    def _learn(self):
        if len(self.replay) < self.hp["batch_size"]:
            return None
        batch       = self.replay.sample(self.hp["batch_size"])
        states      = tf.constant(np.stack([t.state      for t in batch]), dtype=tf.float32)
        actions     = tf.constant(np.array( [t.action     for t in batch]), dtype=tf.int32)
        rewards     = tf.constant(np.array( [t.reward     for t in batch]), dtype=tf.float32)
        next_states = tf.constant(np.stack([t.next_state for t in batch]), dtype=tf.float32)
        dones       = tf.constant(np.array( [t.done       for t in batch], dtype=np.float32))
        return self._update_step(states, actions, rewards, next_states, dones)

    def _get_z(self, gex, mut, cnv) -> tf.Tensor:
        z, _, _ = self.twin.omics_vae((gex, mut, cnv), training=False)
        return z

    def optimize(self, gex, mut, cnv, n_episodes: int = None) -> dict:
        n_episodes = n_episodes or self.hp["n_episodes"]
        z = self._get_z(gex, mut, cnv)

        rewards_history  = []
        losses_history   = []
        valid_count      = 0
        top_molecules    = []   # (reward, smiles)

        print(f"\n[DQN-SELFIES] Démarrage — {n_episodes} épisodes")
        print(f"  ε: {self.hp['eps_start']:.2f} → {self.hp['eps_end']:.2f} "
              f"sur {self.hp['eps_decay_steps']} steps")
        print(f"  Récompenses : QED×{self.hp['qed_weight']} | "
              f"LogP×{self.hp['logp_weight']} | "
              f"Lipinski+{self.hp['lipinski_bonus']} | "
              f"IC50×{self.hp['ic50_weight']} | "
              f"Diversité×{self.hp['diversity_weight']}\n")

        for ep in range(1, n_episodes + 1):
            env   = SELFIESEnv(self.twin, self.feat, self.vocab, z,
                               hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            ep_reward = 0.0
            ep_loss   = []

            while True:
                action                   = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.replay.push(state, action, np.float32(reward),
                                 next_state, np.float32(done))
                state     = next_state
                ep_reward = reward if done else ep_reward
                self.global_step += 1

                loss = self._learn()
                if loss is not None:
                    ep_loss.append(float(loss.numpy()))

                if self.global_step % self.hp["target_update_freq"] == 0:
                    self._sync_target()

                if done:
                    break

            smiles = env.current_smiles
            rewards_history.append(ep_reward)

            # Une molécule est valide si RDKit la parse (reward > -0.4)
            if ep_reward > -0.4 and smiles:
                valid_count += 1
                top_molecules.append((ep_reward, smiles))
                top_molecules.sort(key=lambda x: -x[0])
                top_molecules = top_molecules[:10]  # garder les 10 meilleures

            if ep_reward > self.best_reward and smiles:
                self.best_reward = ep_reward
                self.best_smiles = smiles
                if HAS_RDKIT:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                            self.past_fps.append(fp)
                    except Exception:
                        pass

            if ep % self.hp["log_interval"] == 0 or ep == 1:
                mean_r    = float(np.mean(rewards_history[-50:]))
                mean_l    = float(np.mean(ep_loss)) if ep_loss else float("nan")
                valid_pct = 100.0 * valid_count / ep
                best_disp = (self.best_smiles[:35] if self.best_smiles else "<aucun>")
                print(
                    f"  Ep {ep:5d}/{n_episodes} | "
                    f"ε={self._epsilon():.3f} | "
                    f"R={ep_reward:+.3f} | "
                    f"Moy(50)={mean_r:+.3f} | "
                    f"Valid={valid_pct:.1f}% | "
                    f"Loss={mean_l:.4f} | "
                    f"Best: {best_disp!s:35s} ({self.best_reward:.3f})"
                )

        print(f"\n[DQN-SELFIES] Optimisation terminée.")
        print(f"  Meilleur SMILES  : {self.best_smiles}")
        print(f"  Meilleure reward : {self.best_reward:.4f}")
        print(f"  Molécules valides: {valid_count}/{n_episodes} ({100*valid_count/n_episodes:.1f}%)")

        if top_molecules:
            print(f"\n  Top 5 molécules générées :")
            for rank, (r, s) in enumerate(top_molecules[:5], 1):
                print(f"    {rank}. reward={r:.3f}  {s}")

        return {
            "best_smiles"       : self.best_smiles,
            "best_reward"       : self.best_reward,
            "reward_trajectory" : rewards_history,
            "loss_trajectory"   : losses_history,
            "valid_count"       : valid_count,
            "top_molecules"     : top_molecules,
        }

    def save_weights(self, path: str = "dqn_weights"):
        os.makedirs(path, exist_ok=True)
        self.q_online.save_weights(os.path.join(path, "q_online.weights.h5"))
        self.q_target.save_weights(os.path.join(path, "q_target.weights.h5"))
        print(f"[DQN] Poids sauvegardés dans {path}/")

    def load_weights(self, path: str = "dqn_weights"):
        self.q_online.load_weights(os.path.join(path, "q_online.weights.h5"))
        self._sync_target()
        print(f"[DQN] Poids chargés depuis {path}/")


# ─── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  DQN Drug Optimizer v3 — SELFIES — Bi-Int Digital Twin")
    print("=" * 70)

    if not HAS_SELFIES:
        print("[ERREUR] pip install selfies")
        sys.exit(1)

    print(f"\n[selfies] version {sf.__version__}")

    print("\n[Loading] BiIntDigitalTwin + vocabulaire SELFIES...")
    dt_model   = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()
    vocab      = SELFIESVocabulary(SEED_SMILES)

    # Test de décodage vocabulaire
    test_sel = sf.encoder("CC(=O)Oc1ccccc1C(=O)O")
    test_idx = vocab.encode(test_sel)
    test_smi = vocab.decode(test_idx)
    print(f"[Vocab test] Aspirine → SELFIES → SMILES : {test_smi}")

    # Données omiques synthétiques (remplacer par données réelles via fullPipeline)
    gex = tf.random.normal([1, HP["gex_dim"]])
    mut = tf.cast(tf.random.uniform([1, HP["mut_dim"]], 0, 2, dtype=tf.int32), tf.float32)
    cnv = tf.random.normal([1, HP["cnv_dim"]])

    agent  = DQNDrugOptimizer(dt_model, featurizer, vocab, DQN_HP)
    result = agent.optimize(gex, mut, cnv, n_episodes=DQN_HP["n_episodes"])

    print(f"\n{'='*70}")
    print(f"  RÉSULTAT FINAL")
    print(f"{'='*70}")
    print(f"  Meilleur SMILES  : {result['best_smiles']}")
    print(f"  Récompense       : {result['best_reward']:.4f}")
    print(f"  Molécules valides: {result['valid_count']}/{DQN_HP['n_episodes']} "
          f"({100*result['valid_count']/DQN_HP['n_episodes']:.1f}%)")

    agent.save_weights("dqn_weights_v3")
