"""
================================================================================
  DQN (Deep Q-Network) Drug Optimizer — Bi-Int Digital Twin
================================================================================

Formulation :
  - Etat  (state)  : vecteur latent omique z (produit par l'OmicsVAE) +
                     représentation de la molécule courante (tokens SMILES)
  - Action         : token SMILES suivant à appendre (espace discret = vocab)
                     avec action masking (empêche tokens impossibles)
  - Récompense     : IC50 prédit + validité RDKit + QED + LogP + SA + Lipinski
                     + diversité Tanimoto
  - Episode        : construction d'une molécule token par token jusqu'à
                     <END> ou max_len tokens

Améliorations v2 :
  - Reward shaping : pénalité invalide -0.2 (au lieu de -2.0)
  - Récompenses chimiques : QED, LogP, SA score, Lipinski
  - Action masking : empêche parenthèses/cycles impossibles
  - 2000 épisodes par défaut

Références :
  Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
  Olivecrona et al., "Molecular de novo design through deep RL", JCIM 2017
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
    SMILESVocabulary,
    HP,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors, QED as rdQED
    from rdkit.Chem import rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit non disponible — validité chimique désactivée.")


# ─── Hyper-paramètres DQN ─────────────────────────────────────────────────────
DQN_HP = dict(
    replay_buffer_size = 20_000,
    batch_size         = 64,
    gamma              = 0.99,
    lr                 = 1e-4,
    eps_start          = 1.0,
    eps_end            = 0.05,
    eps_decay_steps    = 10_000,   # plus de décroissance = meilleure exploration
    target_update_freq = 200,
    max_smiles_len     = 40,
    n_episodes         = 2_000,
    hidden_dim         = 256,
    target_ic50        = -1.5,     # IC50 cible (log µM)
    # Récompenses
    invalid_penalty    = -0.2,     # pénalité molécule invalide
    validity_bonus     = 0.5,      # bonus molécule valide RDKit
    ic50_weight        = 0.5,      # poids IC50
    qed_weight         = 1.5,      # poids QED fort (drug-likeness 0-1)
    logp_weight        = 0.3,      # poids LogP (idéal 1-3)
    sa_weight          = 0.2,      # poids SA score
    lipinski_bonus     = 0.5,      # bonus règle des 5 de Lipinski
    complexity_bonus   = 0.8,      # bonus molécules complexes (cycles, HetAt)
    simplicity_penalty = -1.5,     # pénalité molécules trop simples (<5 atomes lourds)
    diversity_weight   = 0.3,      # poids diversité Tanimoto
    min_heavy_atoms    = 5,        # nb minimum atomes lourds pour être considérée
    log_interval       = 50,
)


# ─── 1. Replay Buffer ─────────────────────────────────────────────────────────
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


# ─── 2. Q-Network ─────────────────────────────────────────────────────────────
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


# ─── 3. Action Masking ────────────────────────────────────────────────────────
class ActionMask:
    """
    Empêche les tokens impossibles selon le contexte SMILES courant.
    Règles :
      - parenthèse fermante ) impossible si pas de ( ouverte
      - chiffre de cycle impossible si trop de cycles ouverts (>9)
      - END impossible si molécule trop courte (<3 tokens)
    """
    RING_DIGITS = set("0123456789")

    def __init__(self, vocab: SMILESVocabulary):
        self.vocab = vocab
        self.idx2char = {v: k for k, v in vocab.char2idx.items()}

    def get_mask(self, tokens: List[int]) -> np.ndarray:
        """Retourne un masque booléen : True = action autorisée."""
        mask = np.ones(self.vocab.vocab_size, dtype=bool)
        smiles_so_far = "".join(self.idx2char.get(t, "") for t in tokens
                                if t not in (self.vocab.char2idx.get("<START>", -1),
                                             self.vocab.char2idx.get("<PAD>", -1),
                                             self.vocab.char2idx.get("<EOS>", -1)))

        open_parens = smiles_so_far.count("(") - smiles_so_far.count(")")
        open_rings  = sum(smiles_so_far.count(d) % 2 for d in self.RING_DIGITS)

        # Interdire ) si aucune ( ouverte
        close_idx = self.vocab.char2idx.get(")", -1)
        if close_idx >= 0 and open_parens <= 0:
            mask[close_idx] = False

        # Interdire EOS si molécule trop courte
        end_idx = self.vocab.char2idx.get("<EOS>", self.vocab.char2idx.get("<END>", -1))
        if end_idx >= 0 and len(smiles_so_far) < 3:
            mask[end_idx] = False

        # Interdire trop de cycles ouverts
        if open_rings >= 9:
            for d in self.RING_DIGITS:
                d_idx = self.vocab.char2idx.get(d, -1)
                if d_idx >= 0:
                    mask[d_idx] = False

        return mask


# ─── 4. Environnement de génération SMILES ────────────────────────────────────
class SMILESEnv:
    def __init__(
        self,
        digital_twin: BiIntDigitalTwin,
        featurizer: BRICSMolecularFeaturizer,
        vocab: SMILESVocabulary,
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
        self.masker   = ActionMask(vocab)

        self.max_len    = hp["max_smiles_len"]
        self.vocab_size = vocab.vocab_size
        self.PAD   = vocab.char2idx.get("<PAD>",   0)
        self.END   = vocab.char2idx.get("<EOS>", vocab.char2idx.get("<END>", 1))
        self.START = vocab.char2idx.get("<START>", self.PAD)

        self.tokens: List[int] = []
        self.step_count = 0

    def _build_state(self, last_token: int) -> np.ndarray:
        z_np = self.z.numpy().flatten()
        tok_onehot = np.zeros(self.vocab_size, dtype=np.float32)
        tok_onehot[last_token] = 1.0
        return np.concatenate([z_np, tok_onehot], axis=0)

    def reset(self) -> np.ndarray:
        self.tokens     = [self.START]
        self.step_count = 0
        return self._build_state(self.START)

    def get_action_mask(self) -> np.ndarray:
        return self.masker.get_mask(self.tokens)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.tokens.append(action)
        self.step_count += 1
        done = (action == self.END) or (self.step_count >= self.max_len)

        if done:
            reward = self._compute_reward()
        else:
            reward = 0.0

        next_state = self._build_state(action)
        return next_state, reward, done

    def _compute_reward(self) -> float:
        smiles = self.vocab.decode(self.tokens)

        # 1. Validité RDKit
        mol = None
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return float(self.hp["invalid_penalty"])

        # Pénalité molécule trop simple
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.hp["min_heavy_atoms"]:
            return float(self.hp["simplicity_penalty"])

        reward = self.hp["validity_bonus"]  # +0.5

        # 2. QED — drug-likeness (0 à 1, idéal > 0.6)
        try:
            qed_val = rdQED.qed(mol)
            reward += self.hp["qed_weight"] * qed_val  # max +1.5
        except Exception:
            pass

        # 3. LogP — lipophilicité (idéal entre 1 et 3)
        try:
            logp = Descriptors.MolLogP(mol)
            logp_rew = float(np.exp(-((logp - 2.0) ** 2) / 4.0))
            reward += self.hp["logp_weight"] * logp_rew
        except Exception:
            pass

        # 4. SA score — approximé par NumRotatableBonds + complexité
        try:
            n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            sa_rew = float(np.exp(-n_rot / 10.0))
            reward += self.hp["sa_weight"] * sa_rew
        except Exception:
            pass

        # 5. Complexité : bonus si cycles aromatiques et hétéroatomes
        try:
            n_arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_hetat = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
            complexity = min((n_arom * 0.4 + n_rings * 0.2 + n_hetat * 0.1), 1.0)
            reward += self.hp["complexity_bonus"] * complexity
        except Exception:
            pass

        # 6. Lipinski Rule of 5
        try:
            mw   = Descriptors.MolWt(mol)
            hbd  = rdMolDescriptors.CalcNumHBD(mol)
            hba  = rdMolDescriptors.CalcNumHBA(mol)
            logp = Descriptors.MolLogP(mol)
            if mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5:
                reward += self.hp["lipinski_bonus"]  # +0.5
        except Exception:
            pass

        # 6. IC50 prédit par le Digital Twin
        try:
            atom_feat = self.feat.featurize(smiles)[np.newaxis]
            adj = np.ones((1, HP["max_atoms"], HP["max_atoms"]), dtype=np.float32)
            gex_dummy = tf.zeros([1, HP["gex_dim"]])
            mut_dummy = tf.zeros([1, HP["mut_dim"]])
            cnv_dummy = tf.zeros([1, HP["cnv_dim"]])
            inputs = (tf.constant(atom_feat), tf.constant(adj),
                      gex_dummy, mut_dummy, cnv_dummy)
            ic50_pred, _ = self.twin(inputs, training=False)
            ic50_val = float(ic50_pred[0].numpy())
            ic50_rew = float(np.exp(-((ic50_val - self.hp["target_ic50"]) ** 2) / 2.0))
            reward += self.hp["ic50_weight"] * ic50_rew * 3.0  # max +1.5
        except Exception:
            pass

        # 7. Diversité Tanimoto
        if HAS_RDKIT and self.past_fps and self.hp["diversity_weight"] > 0:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                similarities = DataStructs.BulkTanimotoSimilarity(fp, self.past_fps)
                max_sim = max(similarities) if similarities else 0.0
                reward += (1.0 - max_sim) * self.hp["diversity_weight"] * 2.0
            except Exception:
                pass

        return float(np.clip(reward, -1.0, 10.0))

    @property
    def current_smiles(self) -> str:
        idx2char = {v: k for k, v in self.vocab.char2idx.items()}
        special = {self.START, self.END, self.PAD}
        return "".join(idx2char.get(t, "") for t in self.tokens if t not in special)


# ─── 5. Agent DQN ─────────────────────────────────────────────────────────────
class DQNDrugOptimizer:
    def __init__(
        self,
        digital_twin: BiIntDigitalTwin,
        featurizer: BRICSMolecularFeaturizer,
        vocab: SMILESVocabulary,
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

        print(f"[DQN] Initialisé | state_dim={state_dim} | vocab_size={vocab_size}")

    def _sync_target(self):
        self.q_target.set_weights(self.q_online.get_weights())

    def _epsilon(self) -> float:
        t     = min(self.global_step, self.hp["eps_decay_steps"])
        ratio = t / self.hp["eps_decay_steps"]
        return self.hp["eps_start"] + ratio * (self.hp["eps_end"] - self.hp["eps_start"])

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        if random.random() < self._epsilon():
            # Exploration : choisir parmi les actions valides uniquement
            valid_actions = np.where(mask)[0]
            return int(random.choice(valid_actions)) if len(valid_actions) > 0 \
                   else random.randint(0, self.vocab.vocab_size - 1)
        state_t  = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
        q_values = self.q_online(state_t, training=False).numpy()[0]
        # Masquer les actions invalides avec -inf
        q_values[~mask] = -np.inf
        return int(np.argmax(q_values))

    @tf.function
    def _update_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_all  = self.q_online(states, training=True)
            idx    = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa   = tf.gather_nd(q_all, idx)

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

        rewards_history = []
        losses_history  = []
        valid_count     = 0

        print(f"\n[DQN] Démarrage de l'optimisation — {n_episodes} épisodes")
        print(f"  ε start={self.hp['eps_start']:.2f}  ε end={self.hp['eps_end']:.2f}")
        print(f"  γ={self.hp['gamma']}  lr={self.hp['lr']}  batch={self.hp['batch_size']}")
        print(f"  Reward shaping : invalide={self.hp['invalid_penalty']} | "
              f"valide={self.hp['validity_bonus']} | QED={self.hp['qed_weight']} | "
              f"LogP={self.hp['logp_weight']} | Lipinski={self.hp['lipinski_bonus']}\n")

        for ep in range(1, n_episodes + 1):
            env   = SMILESEnv(self.twin, self.feat, self.vocab, z,
                              hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            ep_reward = 0.0
            ep_loss   = []

            while True:
                mask                           = env.get_action_mask()
                action                         = self.select_action(state, mask)
                next_state, reward, done       = env.step(action)
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

            # Compter les molécules valides
            if ep_reward > self.hp["invalid_penalty"]:
                valid_count += 1

            if ep_reward > self.best_reward:
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
                valid_pct = 100 * valid_count / ep
                print(
                    f"  Ep {ep:5d}/{n_episodes} | "
                    f"ε={self._epsilon():.3f} | "
                    f"Reward={ep_reward:+.3f} | "
                    f"Mean(50)={mean_r:+.3f} | "
                    f"Valid={valid_pct:.1f}% | "
                    f"Loss={mean_l:.4f} | "
                    f"Best: {self.best_smiles[:30]!s:30s} ({self.best_reward:.3f})"
                )

        print(f"\n[DQN] Optimisation terminée.")
        print(f"  Meilleur SMILES  : {self.best_smiles}")
        print(f"  Meilleure reward : {self.best_reward:.4f}")
        print(f"  Molécules valides: {valid_count}/{n_episodes} ({100*valid_count/n_episodes:.1f}%)")

        return {
            "best_smiles"       : self.best_smiles,
            "best_reward"       : self.best_reward,
            "reward_trajectory" : rewards_history,
            "loss_trajectory"   : losses_history,
            "valid_count"       : valid_count,
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


# ─── 6. Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  DQN Drug Optimizer v2 — Bi-Int Digital Twin")
    print("=" * 70)

    print("\n[Loading] BiIntDigitalTwin...")
    dt_model   = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()
    vocab      = SMILESVocabulary()

    gex = tf.random.normal([1, HP["gex_dim"]])
    mut = tf.cast(tf.random.uniform([1, HP["mut_dim"]], 0, 2, dtype=tf.int32), tf.float32)
    cnv = tf.random.normal([1, HP["cnv_dim"]])

    agent = DQNDrugOptimizer(dt_model, featurizer, vocab, DQN_HP)
    result = agent.optimize(gex, mut, cnv, n_episodes=DQN_HP["n_episodes"])

    print(f"\n=== RÉSULTAT FINAL ===")
    print(f"  Meilleur SMILES  : {result['best_smiles']}")
    print(f"  Récompense       : {result['best_reward']:.4f}")
    print(f"  Molécules valides: {result['valid_count']}/{DQN_HP['n_episodes']}")

    agent.save_weights("dqn_weights_v2")
