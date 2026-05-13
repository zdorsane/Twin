"""
================================================================================
  DQN (Deep Q-Network) Drug Optimizer — Bi-Int Digital Twin
================================================================================

Formulation :
  - Etat  (state)  : vecteur latent omiqueS z (produit par l'OmicsVAE) +
                     représentation de la molécule courante (tokens SMILES)
  - Action         : token SMILES suivant à appendre (espace discret = vocab)
  - Récompense     : fonction de l'IC50 prédit par le Digital Twin
                     + validité chimique (RDKit) + diversité de Tanimoto
  - Episode        : construction d'une molécule token par token jusqu'à
                     <END> ou max_len tokens

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
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit non disponible — validité chimique désactivée.")


# ─── Hyper-paramètres DQN ─────────────────────────────────────────────────────
DQN_HP = dict(
    replay_buffer_size = 20_000,   # taille du replay buffer
    batch_size         = 64,       # mini-batch pour la mise à jour Q
    gamma              = 0.99,     # facteur d'actualisation
    lr                 = 1e-4,     # taux d'apprentissage
    eps_start          = 1.0,      # epsilon initial (exploration)
    eps_end            = 0.05,     # epsilon minimal
    eps_decay_steps    = 5_000,    # décroissance linéaire
    target_update_freq = 200,      # fréquence de synchro du réseau cible
    max_smiles_len     = 40,       # longueur maximale d'un SMILES généré
    n_episodes         = 2_000,    # nombre d'épisodes d'entraînement
    hidden_dim         = 256,      # taille cachée du Q-network
    target_ic50        = -1.5,     # IC50 cible (log µM) — plus bas = plus actif
    ic50_weight        = 0.6,      # poids de la récompense IC50
    validity_bonus     = 1.0,      # bonus molécule valide
    diversity_weight   = 0.2,      # poids de la diversité Tanimoto
    log_interval       = 50,       # affichage toutes les N étapes
)


# ─── 1. Replay Buffer ─────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Experience replay avec deque de taille fixe."""

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
    """
    Réseau Q(s, a) paramétré par theta.

    Entrée : concat(z_omics, token_embedding_courant) → scalaire Q pour chaque action
    Sortie : vecteur [vocab_size] de valeurs Q
    """

    def __init__(self, state_dim: int, vocab_size: int, hidden_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu", input_shape=(state_dim,)),
                layers.LayerNormalization(),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim // 2, activation="relu"),
                layers.Dense(vocab_size),   # Q-valeurs pour chaque token
            ]
        )

    def call(self, state, training=False):
        return self.net(state, training=training)


# ─── 3. Environnement de génération SMILES ────────────────────────────────────
class SMILESEnv:
    """
    Environnement RL pour la construction token-par-token d'un SMILES.

    - reset()  : démarre un nouvel épisode (molécule vide)
    - step(a)  : ajoute un token, retourne (next_state, reward, done)
    """

    def __init__(
        self,
        digital_twin: BiIntDigitalTwin,
        featurizer: BRICSMolecularFeaturizer,
        vocab: SMILESVocabulary,
        z_omics: tf.Tensor,
        hp: dict = DQN_HP,
        past_fps: List = None,   # fingerprints des molécules déjà trouvées
    ):
        self.twin       = digital_twin
        self.feat       = featurizer
        self.vocab      = vocab
        self.z          = z_omics                  # [1, latent_dim]
        self.hp         = hp
        self.past_fps   = past_fps or []

        self.max_len    = hp["max_smiles_len"]
        self.vocab_size = vocab.vocab_size

        # token spéciaux
        self.START = vocab.char2idx.get("<START>", 0)
        self.END   = vocab.char2idx.get("<END>",   1)
        self.PAD   = vocab.char2idx.get("<PAD>",   2)

        # état interne
        self.tokens: List[int] = []
        self.step_count = 0

    # ── Etat = [z_omics | embedding du dernier token]
    def _build_state(self, last_token: int) -> np.ndarray:
        z_np = self.z.numpy().flatten()                           # [latent_dim]
        tok_onehot = np.zeros(self.vocab_size, dtype=np.float32)
        tok_onehot[last_token] = 1.0
        return np.concatenate([z_np, tok_onehot], axis=0)        # [latent+vocab]

    def reset(self) -> np.ndarray:
        self.tokens     = [self.START]
        self.step_count = 0
        return self._build_state(self.START)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.tokens.append(action)
        self.step_count += 1
        done = (action == self.END) or (self.step_count >= self.max_len)

        if done:
            reward = self._compute_reward()
        else:
            reward = 0.0   # récompense terminale uniquement

        next_state = self._build_state(action)
        return next_state, reward, done

    # ── Récompense terminale ──────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        smiles = self.vocab.decode(self.tokens)

        # 1. Validité chimique
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles)
            valid = mol is not None
        else:
            valid = len(smiles) > 2

        if not valid:
            return -2.0

        reward = self.hp["validity_bonus"]

        # 2. IC50 prédit par le Digital Twin
        try:
            atom_feat = self.feat.featurize(smiles)[np.newaxis]           # [1, max_at, feat]
            adj = np.ones((1, HP["max_atoms"], HP["max_atoms"]), dtype=np.float32)

            # gex/mut/cnv factices (dimension cohérente avec z)
            gex_dummy = tf.zeros([1, HP["gex_dim"]])
            mut_dummy = tf.zeros([1, HP["mut_dim"]])
            cnv_dummy = tf.zeros([1, HP["cnv_dim"]])

            inputs = (
                tf.constant(atom_feat),
                tf.constant(adj),
                gex_dummy, mut_dummy, cnv_dummy,
            )
            ic50_pred, _ = self.twin(inputs, training=False)
            ic50_val = float(ic50_pred[0].numpy())

            # Récompense gaussienne centrée sur target_ic50
            ic50_rew = float(
                np.exp(-((ic50_val - self.hp["target_ic50"]) ** 2) / 2.0)
            )
            reward += self.hp["ic50_weight"] * ic50_rew * 5.0

        except Exception:
            pass   # si le modèle échoue, on garde juste la validité

        # 3. Diversité Tanimoto (bonus si la molécule est différente des précédentes)
        if HAS_RDKIT and self.past_fps and self.hp["diversity_weight"] > 0:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                similarities = DataStructs.BulkTanimotoSimilarity(fp, self.past_fps)
                max_sim = max(similarities) if similarities else 0.0
                # Bonus de diversité : 0 si identique, 1 si totalement nouveau
                diversity_bonus = (1.0 - max_sim) * self.hp["diversity_weight"] * 2.0
                reward += diversity_bonus
            except Exception:
                pass

        return float(np.clip(reward, -5.0, 10.0))

    @property
    def current_smiles(self) -> str:
        return self.vocab.decode(self.tokens)


# ─── 4. Agent DQN ─────────────────────────────────────────────────────────────
class DQNDrugOptimizer:
    """
    Agent DQN pour optimiser des candidats médicamenteux via le Digital Twin.

    Algorithme :
      - Double DQN : réseau Q_online + réseau Q_target (mise à jour périodique)
      - Experience Replay : transitions stockées dans un ReplayBuffer
      - Epsilon-greedy : décroissance linéaire de l'exploration
    """

    def __init__(
        self,
        digital_twin: BiIntDigitalTwin,
        featurizer: BRICSMolecularFeaturizer,
        vocab: SMILESVocabulary,
        hp: dict = DQN_HP,
    ):
        self.twin      = digital_twin
        self.feat      = featurizer
        self.vocab     = vocab
        self.hp        = hp

        latent_dim  = HP["latent_dim"]
        vocab_size  = vocab.vocab_size
        state_dim   = latent_dim + vocab_size   # z + one-hot dernier token

        # Double DQN : réseau en ligne + réseau cible
        self.q_online = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_online")
        self.q_target = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_target")
        self._sync_target()

        self.optimizer = keras.optimizers.Adam(hp["lr"])
        self.replay    = ReplayBuffer(hp["replay_buffer_size"])
        self.loss_fn   = keras.losses.Huber()   # plus robuste que MSE

        self.global_step    = 0
        self.best_smiles    = ""
        self.best_reward    = -float("inf")
        self.past_fps: List = []

        print(
            f"[DQN] Initialisé | state_dim={state_dim} | vocab_size={vocab_size}"
        )

    def _sync_target(self):
        """Copie les poids de Q_online → Q_target."""
        self.q_target.set_weights(self.q_online.get_weights())

    def _epsilon(self) -> float:
        t      = min(self.global_step, self.hp["eps_decay_steps"])
        ratio  = t / self.hp["eps_decay_steps"]
        return self.hp["eps_start"] + ratio * (self.hp["eps_end"] - self.hp["eps_start"])

    # ── Sélection d'action epsilon-greedy ────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self._epsilon():
            return random.randint(0, self.vocab.vocab_size - 1)
        state_t  = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
        q_values = self.q_online(state_t, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    # ── Mise à jour du réseau (Double DQN) ────────────────────────────────────
    @tf.function
    def _update_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            # Q(s, a) courant
            q_all    = self.q_online(states, training=True)
            idx      = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa     = tf.gather_nd(q_all, idx)

            # Cible Double DQN : a* choisi par Q_online, évalué par Q_target
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
        batch = self.replay.sample(self.hp["batch_size"])

        states      = tf.constant(np.stack([t.state      for t in batch]), dtype=tf.float32)
        actions     = tf.constant(np.array([t.action     for t in batch]), dtype=tf.int32)
        rewards     = tf.constant(np.array([t.reward     for t in batch]), dtype=tf.float32)
        next_states = tf.constant(np.stack([t.next_state for t in batch]), dtype=tf.float32)
        dones       = tf.constant(np.array([t.done       for t in batch], dtype=np.float32))

        return self._update_step(states, actions, rewards, next_states, dones)

    # ── Calcul du vecteur latent omique ──────────────────────────────────────
    def _get_z(self, gex, mut, cnv) -> tf.Tensor:
        z, _, _ = self.twin.omics_vae((gex, mut, cnv), training=False)
        return z   # [batch, latent_dim]

    # ── Boucle d'optimisation principale ──────────────────────────────────────
    def optimize(
        self,
        gex: tf.Tensor,
        mut: tf.Tensor,
        cnv: tf.Tensor,
        n_episodes: int = None,
    ) -> dict:
        """
        Lance n_episodes épisodes DQN.
        Retourne le meilleur SMILES trouvé et sa récompense.
        """
        n_episodes = n_episodes or self.hp["n_episodes"]
        z = self._get_z(gex, mut, cnv)

        rewards_history = []
        losses_history  = []
        best_per_ep     = []

        print(f"\n[DQN] Démarrage de l'optimisation — {n_episodes} épisodes")
        print(f"  ε start={self.hp['eps_start']:.2f}  ε end={self.hp['eps_end']:.2f}")
        print(f"  γ={self.hp['gamma']}  lr={self.hp['lr']}  batch={self.hp['batch_size']}\n")

        for ep in range(1, n_episodes + 1):
            env   = SMILESEnv(self.twin, self.feat, self.vocab, z,
                              hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            ep_reward = 0.0
            ep_loss   = []

            while True:
                action                         = self.select_action(state)
                next_state, reward, done       = env.step(action)
                self.replay.push(state, action, np.float32(reward),
                                 next_state, np.float32(done))
                state     = next_state
                ep_reward = reward if done else ep_reward   # récompense terminale
                self.global_step += 1

                # Apprentissage
                loss = self._learn()
                if loss is not None:
                    ep_loss.append(float(loss.numpy()))

                # Synchronisation du réseau cible
                if self.global_step % self.hp["target_update_freq"] == 0:
                    self._sync_target()

                if done:
                    break

            smiles = env.current_smiles
            rewards_history.append(ep_reward)
            best_per_ep.append(ep_reward)

            # Sauvegarde du meilleur candidat
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                self.best_smiles = smiles
                # Mémorise le fingerprint pour la diversité
                if HAS_RDKIT:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                            self.past_fps.append(fp)
                    except Exception:
                        pass

            # Logging périodique
            if ep % self.hp["log_interval"] == 0 or ep == 1:
                mean_r  = float(np.mean(rewards_history[-50:]))
                mean_l  = float(np.mean(ep_loss)) if ep_loss else float("nan")
                eps_val = self._epsilon()
                print(
                    f"  Ep {ep:5d}/{n_episodes} | "
                    f"ε={eps_val:.3f} | "
                    f"Reward={ep_reward:+.3f} | "
                    f"Mean(50)={mean_r:+.3f} | "
                    f"Loss={mean_l:.4f} | "
                    f"Best: {self.best_smiles[:35]!s:35s} ({self.best_reward:.3f})"
                )

        print(f"\n[DQN] Optimisation terminée.")
        print(f"  Meilleur SMILES : {self.best_smiles}")
        print(f"  Meilleure récompense : {self.best_reward:.4f}")

        return {
            "best_smiles"      : self.best_smiles,
            "best_reward"      : self.best_reward,
            "reward_trajectory": rewards_history,
            "loss_trajectory"  : losses_history,
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


# ─── 5. Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  DQN Drug Optimizer — Bi-Int Digital Twin")
    print("=" * 70)

    # Chargement du modèle
    print("\n[Loading] BiIntDigitalTwin...")
    dt_model  = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()
    vocab      = SMILESVocabulary()

    # Données omiques de démonstration
    gex = tf.random.normal([1, HP["gex_dim"]])
    mut = tf.cast(tf.random.uniform([1, HP["mut_dim"]], 0, 2, dtype=tf.int32), tf.float32)
    cnv = tf.random.normal([1, HP["cnv_dim"]])

    # Instanciation de l'agent
    agent = DQNDrugOptimizer(dt_model, featurizer, vocab, DQN_HP)

    # Lancement de l'optimisation (réduit pour le test)
    result = agent.optimize(gex, mut, cnv, n_episodes=200)

    print(f"\n=== RÉSULTAT FINAL ===")
    print(f"  Meilleur SMILES : {result['best_smiles']}")
    print(f"  Récompense      : {result['best_reward']:.4f}")

    # Sauvegarde optionnelle
    agent.save_weights("dqn_weights_demo")
