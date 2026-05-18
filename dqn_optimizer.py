r"""
================================================================================
  DQN Drug Optimizer -- SELFIES v5.0 -- Bi-Int Digital Twin
================================================================================

Historique des versions :
  v3.0 : SELFIES de base (100% valides), polysulfides + cumulenes hackes
  v3.1 : +carbon_frac + cumul_penalty -> stereocha ines hackees
  v3.2 : +repeat/stereo/size penalties -> collapse reward (4% valides)
  v3.3 : penalites douces -> 87.5% valides, charges/halogenes exploites
  v3.4 : +charge/isotope/halogen penalties -> 72.2% valides
  v3.5 : +alkyne penalty, max_halogens=1 -> 62.7% valides, reward starvation
  v3.7 : qed*3 + acyclic_penalty -> 64.2% valides, stereochain+cyclopropane
         Probleme : agent evite acyclic_penalty avec petit cycle aliphatique
         [C@@H1] chaine reste dominant car max_token_repeat trop permissif
  v3.7 : Forcer les aromatiques + punir stereo-CH en chaine
    - nonarom_penalty=-0.5 si aucun cycle AROMATIQUE (cyclopropane ne suffit plus)
    - max_token_repeat 8 -> 4  (limite repetition du meme token)
    - repeat_penalty_coef 0.1 -> 0.3  (penalite plus forte par repetition)
    - arom_bonus 0.8 -> 1.2  (recompense plus forte pour les aromatiques)
    - BUG : fragments disconnectes acceptes (SMILES avec '.'), iode x2 autorise
  v3.8 : Bloquer les exploits de fragments disconnectes + renforcer penalites
    - Rejet immediat (-1.0) si SMILES contient '.' (fragment disconnecte)
    - max_halogens 2 -> 0  (aucun halogene tolere)
    - charge_penalty_coef 0.4 -> 2.0  (charges = rejet quasi-certain)
    - nonarom_penalty -0.5 -> -2.0  (sans aromatique = forte penalite)
    - PROBLEME : reward starvation — penalites trop fortes, signal positif absent
    - Vocab contient encore Br/F/charges via filtre trop permissif ("r", "F")
  v3.9 : Filtrage vocab a la source + penalites equilibrees
    - SELFIESVocabulary filtre les tokens halogenes/charges/stereo a la construction
    - Retrait de F, Cl, Br, I, [+], [-] du vocab directement
    - nonarom_penalty revient a -0.5 (penalite douce, pas bloquante)
    - PROBLEME : regex blacklist trop large — a retire Ring/Branch -> 37 tokens
    - Paracetamol encode en CCC=O : cycles aromatiques impossibles
    - Exploit C\CP#S\[C@]#N : triple liaison soufre acceptee par RDKit
  v3.10 : Whitelist corpus-only + blacklist chirurgicale
    - token_set construit uniquement depuis le corpus ChEMBL drug-like (pas d'alphabet)
    - Blacklist ciblee : Cl, Br, I, charges [+-], isotopes, metaux
    - Ring1/Ring2/Branch1/Branch2 conserves (essentiels aux cycles)
    - PROBLEME : SELFIES decoder peut produire Br/I meme sans leurs tokens dans le vocab
    - Exploit Br\OC[OH0]/I : [F] token -> decode parfois en Br ou I via semantique SELFIES
  v3.11 : Rejet hard post-decode sur atomes interdits
    - Apres MolFromSmiles, verifier les numeros atomiques : Cl(17)/Br(35)/I(53) -> -1.0
    - PROBLEME residuel : F (atomicNum=9) toujours dans le vocab -> exploit [N@@]\S\S\N/F
    - Cause structurelle : 2000 episodes trop court pour converger (Moy50 jamais positif)
    - Best SMILES trouve ep 1-50 et fige -> agent n explore pas, il garde le 1er coup de chance
  v4.0 : Refonte structurelle — F retire, reward shaping, 10k episodes
    - F retire du vocab ET du rejet post-decode (forbidden={9,17,35,53})
    - Reward shaping : +0.03 par token aromatique [=C]/[=N] durant l episode
    - n_episodes 2000 -> 10000, eps_decay_steps 8000 -> 20000
    - max_selfies_len 30 -> 20 (molecules plus courtes = plus facile a apprendre)
    - RESULTAT : Moy50 jamais positif (plateau -0.42), best SMILES fige ep.200
    - CAUSE : eps_min=0.05 trop bas (exploration stoppe trop tot), reward shaping
      +0.03 trop faible vs bruit -0.5 (nonarom), buffer rempli d exemples negatifs
  v5.0 : Warm-start expert + exploration persistante + reward shaping fort
    - Warm-start : pre-remplissage replay buffer avec 500 trajectoires
      extraites des SEED_SMILES (molecules drug-like connues encodees token-par-token)
      -> buffer contient des exemples positifs (R > 0) des le debut
    - eps_min 0.05 -> 0.15 (garder exploration residuelle tout au long)
    - Reward shaping : +0.15 par token [Ring1]/[Ring2] (cycle = signal fort),
      +0.05 par token [=C]/[=N] (aromaticite) — 3x plus que v4.0
    - nonarom_penalty -0.5 -> -1.0 (penalite plus claire sans etre bloquante)
    - RESULTAT : Valid%=60.4%, Moy50 jamais positif, best=3.153 (acyclique, arom=0)
    - CAUSE : reward shaping rate le token cle de fermeture aromatique :
      SELFIES encode les cycles arom avec [Ring1][=Branch1] (pas [Ring1] seul)
      [=Branch1] (20k occurrences) est le discriminant aromatique, absent du shaping
      -> agent apprend [Ring1] seul = cycles aliphatiques (thianes, dioxolanes)
  v5.1 : Fix chirurgical reward shaping — cibler [=Branch1]/[#Branch1]
    - Reward shaping : +0.20 par token [=Branch1] (fermeture aromatique),
      +0.10 par token [Ring1] (ouverture cycle), +0.05 par [=C]/[=N]
    - Blacklist etendue : [P] retire (phosphore non-drug-like dans ce contexte)
    - nonarom_penalty maintenu a -1.0

References :
  Krenn et al., "SELFIES", Mach. Learn.: Sci. Technol. 2020.
  Mnih et al., "Human-level control through deep RL", Nature 2015.
  Lipinski et al., "Experimental and computational approaches...", 1997.
================================================================================
"""

import os
import re
import sys
import random
import warnings
import logging
import collections
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
    print("[WARN] RDKit non disponible.")


# ─── Extraction SMILES depuis ChEMBL SDF ─────────────────────────────────────
CHEMBL_SDF_PATH = "/home/crbt/Twin/Dataset/chembl_36.sdf"

def load_chembl_smiles(n: int = 10_000,
                       sdf_path: str = CHEMBL_SDF_PATH,
                       max_heavy: int = 40,
                       min_heavy: int = 8) -> List[str]:
    """
    Extrait n SMILES drug-like depuis le SDF ChEMBL.
    Filtre : 8–40 atomes lourds, QED ≥ 0.3, pas de métaux.
    Si le SDF n'est pas accessible, retourne SEED_SMILES.
    """
    if not os.path.exists(sdf_path):
        print(f"[Corpus] SDF non trouvé ({sdf_path}) — utilisation seed SMILES.")
        return SEED_SMILES

    FORBIDDEN_ATOMS = {5, 13, 14, 15, 33, 34, 50, 51, 52, 82, 83}  # B, Al, Si, P, As, Se, Sn, Sb, Te, Pb, Bi

    smiles_list = []
    print(f"[Corpus] Extraction de {n} SMILES drug-like depuis ChEMBL SDF...")
    try:
        supplier = Chem.ForwardSDMolSupplier(sdf_path, removeHs=True, sanitize=True)
        scanned = 0
        for mol in supplier:
            if len(smiles_list) >= n:
                break
            scanned += 1
            if scanned % 100_000 == 0:
                print(f"  {scanned:,} molécules scannées | {len(smiles_list):,} acceptées")
            if mol is None:
                continue
            n_heavy = mol.GetNumHeavyAtoms()
            if not (min_heavy <= n_heavy <= max_heavy):
                continue
            atom_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
            if atom_nums & FORBIDDEN_ATOMS:
                continue
            if 6 not in atom_nums:   # must have carbon
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
        print(f"[Corpus] Erreur lecture SDF : {e} — utilisation seed SMILES.")
        return SEED_SMILES

    print(f"[Corpus] {len(smiles_list)} SMILES extraits ({scanned:,} scannés)")
    if len(smiles_list) < 100:
        smiles_list = smiles_list + SEED_SMILES
    return smiles_list


# ─── Seed SMILES (fallback si ChEMBL absent) ─────────────────────────────────
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


# ─── Hyper-paramètres DQN ─────────────────────────────────────────────────────
DQN_HP = dict(
    replay_buffer_size = 20_000,
    batch_size         = 64,
    gamma              = 0.99,
    lr                 = 3e-4,
    eps_start          = 1.0,
    eps_end            = 0.15,     # v5.0: exploration residuelle persistante
    eps_decay_steps    = 15_000,   # v5.0: decay plus rapide (warm-start compense)
    target_update_freq = 200,
    max_selfies_len    = 20,
    n_episodes         = 5_000,    # v5.0: warm-start rend 5k suffisant
    hidden_dim         = 256,
    target_ic50        = -1.5,
    # ── Récompenses positives (v3.7)
    qed_weight         = 3.0,
    logp_weight        = 0.5,
    lipinski_bonus     = 1.0,
    ic50_weight        = 0.8,
    diversity_weight   = 0.4,
    arom_bonus         = 1.2,    # augmenté (était 0.8) : forte récompense aromatique
    # ── Pénalités chimiques
    carbon_penalty     = -0.5,
    cumul_penalty      = -0.5,
    min_carbon_frac    = 0.25,
    size_penalty_coef  = 0.05,
    max_heavy_atoms    = 30,
    repeat_penalty_coef= 0.3,    # renforcé (était 0.1)
    max_token_repeat   = 4,      # réduit (était 8) : bloque [C@@H1]×5+
    stereo_penalty_coef= 0.15,    # v3.9: modéré
    max_stereo_centers = 4,       # v3.9: raisonnable
    # ── Pénalités drug-likeness
    charge_penalty_coef   = 0.4,  # v3.9: retour valeur saine (vocab filtre les charges)
    isotope_penalty       = -0.5,
    halogen_penalty_coef  = 0.4,
    max_halogens          = 1,    # v3.9: 1 halogène toléré (F médicinal courant)
    alkyne_penalty_coef   = 0.5,
    max_alkynes           = 0,
    # ── Pénalité v5.0 : pas de cycle aromatique (plus nette — orienter vers arom)
    nonarom_penalty       = -1.0,  # v5.0: signal plus clair
    log_interval          = 50,
    # ── Warm-start v5.0
    warmstart_episodes    = 500,   # nb trajectoires expert pre-injectees dans le buffer
)


# ─── Vocabulaire SELFIES ──────────────────────────────────────────────────────
class SELFIESVocabulary:
    PAD_TOKEN = "[nop]"
    END_TOKEN = "[EOS]"

    def __init__(self, smiles_list: List[str]):
        selfies_list = []
        for smi in smiles_list:
            try:
                sel = sf.encoder(smi)
                if sel:
                    selfies_list.append(sel)
            except Exception:
                pass

        # v3.9 — Whitelist stricte : uniquement les tokens du corpus ChEMBL drug-like
        # On prend les tokens qui apparaissent réellement dans les SMILES drug-like,
        # puis on retire les tokens non-drug (halogènes lourds, charges, isotopes).
        token_set = set()
        for sel in selfies_list:
            for tok in sf.split_selfies(sel):
                token_set.add(tok)

        # Tokens interdits : halogènes lourds, charges formelles, isotopes, métaux
        _BANNED = re.compile(
            r"\[Cl"         # chlore
            r"|\[Br"        # brome
            r"|\[I[^n]"     # iode (mais pas [In] = indium, absent de toute façon)
            r"|\[.*[+\-]\d*\]"   # charges formelles : [N+1], [C-1], [NH2+], etc.
            r"|\[\d+"        # isotopes : [11C], [125I]
            r"|\[Si"  r"|\[Se"  r"|\[Te"  r"|\[Sn"
            r"|\[Pb"  r"|\[As"  r"|\[Ge"  r"|\[B\b" r"|\[P\b"
        )
        token_set = {t for t in token_set if not _BANNED.search(t)}

        token_set.discard(self.END_TOKEN)
        token_set.discard(self.PAD_TOKEN)

        self.idx2tok   = [self.PAD_TOKEN, self.END_TOKEN] + sorted(token_set)
        self.tok2idx   = {t: i for i, t in enumerate(self.idx2tok)}
        self.PAD_IDX   = 0
        self.END_IDX   = 1
        self.vocab_size = len(self.idx2tok)

        # Index des tokens stéréochimiques (pour pénalité)
        self.stereo_idxs = {
            i for i, t in enumerate(self.idx2tok)
            if any(s in t for s in ["@@", "@H", "@]"])
        }

        print(f"[SELFIES Vocab] {self.vocab_size} tokens | "
              f"{len(selfies_list)}/{len(smiles_list)} SMILES convertis | "
              f"{len(self.stereo_idxs)} tokens stéréo")

    def encode(self, selfies_str: str) -> List[int]:
        return [self.tok2idx.get(t, self.PAD_IDX)
                for t in sf.split_selfies(selfies_str)] + [self.END_IDX]

    def decode(self, indices: List[int]) -> str:
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
        try:
            return sf.decoder("".join(tokens)) or ""
        except Exception:
            return ""

    def random_token(self) -> int:
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
    def sample(self, n: int) -> List[Transition]:
        return random.sample(self.buffer, n)
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
    def call(self, x, training=False):
        return self.net(x, training=training)


# ─── Environnement SELFIES ────────────────────────────────────────────────────
class SELFIESEnv:
    def __init__(self, twin, feat, vocab: SELFIESVocabulary,
                 z_omics: tf.Tensor, hp: dict = DQN_HP, past_fps: List = None):
        self.twin     = twin
        self.feat     = feat
        self.vocab    = vocab
        self.hp       = hp
        self.past_fps = past_fps or []
        self.max_len  = hp["max_selfies_len"]
        self._z_np    = z_omics.numpy().flatten()
        self.tokens: List[int] = []
        self.step_count = 0

    def _state(self, tok_idx: int) -> np.ndarray:
        oh = np.zeros(self.vocab.vocab_size, dtype=np.float32)
        oh[min(tok_idx, self.vocab.vocab_size - 1)] = 1.0
        return np.concatenate([self._z_np, oh])

    def reset(self) -> np.ndarray:
        self.tokens, self.step_count = [], 0
        return self._state(self.vocab.PAD_IDX)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.tokens.append(action)
        self.step_count += 1
        done = (action == self.vocab.END_IDX) or (self.step_count >= self.max_len)
        if done:
            reward = self._compute_reward()
        else:
            # v5.1 reward shaping : cibler les tokens de fermeture aromatique SELFIES
            # SELFIES encode les cycles arom comme [C][=C]...[Ring1][=Branch1]
            # [=Branch1] = discriminant des aromatiques (20k vs 36k Ring1 dans corpus)
            tok = self.vocab.idx2tok[action] if action < self.vocab.vocab_size else ""
            if tok in ("[=Branch1]", "[#Branch1]"):
                reward = 0.20      # fermeture aromatique — signal le plus fort
            elif tok in ("[Ring1]", "[Ring2]"):
                reward = 0.10      # ouverture/fermeture cycle
            elif tok in ("[=C]", "[=N]"):
                reward = 0.05      # double liaison carbone/azote
            else:
                reward = 0.0
        return self._state(action), reward, done

    def _compute_reward(self) -> float:
        smiles = self.vocab.decode(self.tokens)
        if not smiles:
            return -0.5

        # Rejeter les fragments disconnectés
        if '.' in smiles:
            return -1.0

        mol = Chem.MolFromSmiles(smiles) if HAS_RDKIT else None
        if mol is None:
            return -0.5

        # v4.0 — Rejet hard : F/Cl/Br/I tous interdits (F=9 aussi — exploit v4.0)
        _FORBIDDEN_ATOMS = {9, 17, 35, 53}  # F, Cl, Br, I
        atom_nums_set = {a.GetAtomicNum() for a in mol.GetAtoms()}
        if atom_nums_set & _FORBIDDEN_ATOMS:
            return -1.0

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 5:
            return -0.2

        # ── Pénalités douces (déductions, jamais de retour anticipé)
        penalties = 0.0

        # Fraction carbone insuffisante
        atom_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        n_carbon  = atom_nums.count(6)
        if n_carbon == 0 or (n_carbon / n_heavy) < self.hp["min_carbon_frac"]:
            penalties += self.hp["carbon_penalty"]   # -0.5

        # Cumulènes
        can_smi = Chem.MolToSmiles(mol)
        if can_smi.count("=C=") + can_smi.count("=c=") >= 3:
            penalties += self.hp["cumul_penalty"]    # -0.5

        # Taille excessive
        if n_heavy > self.hp["max_heavy_atoms"]:
            penalties -= (n_heavy - self.hp["max_heavy_atoms"]) * self.hp["size_penalty_coef"]

        # Répétition token
        counts  = collections.Counter(self.tokens)
        max_rep = max(counts.values()) if counts else 0
        if max_rep > self.hp["max_token_repeat"]:
            penalties -= (max_rep - self.hp["max_token_repeat"]) * self.hp["repeat_penalty_coef"]

        # Stéréochimie excessive
        stereo_count = sum(counts[i] for i in self.vocab.stereo_idxs if i in counts)
        if stereo_count > self.hp["max_stereo_centers"]:
            penalties -= (stereo_count - self.hp["max_stereo_centers"]) * self.hp["stereo_penalty_coef"]

        # ── Pénalités drug-likeness v3.7 ──────────────────────────────────────
        # Charges formelles (Cl[C+1]=[C+1]... exploitation)
        charged_atoms = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
        if charged_atoms > 0:
            penalties -= charged_atoms * self.hp["charge_penalty_coef"]

        # Isotopes (tokens filtrés du vocab, mais un token peut encoder un isotope
        # via SELFIES sémantique — vérification sur la molécule RDKit)
        if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
            penalties += self.hp["isotope_penalty"]   # valeur négative

        # Halogènes excessifs (F=9, Cl=17, Br=35, I=53)
        HALOGENS = {9, 17, 35, 53}
        n_halogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in HALOGENS)
        if n_halogens > self.hp["max_halogens"]:
            penalties -= (n_halogens - self.hp["max_halogens"]) * self.hp["halogen_penalty_coef"]

        # Polynes (C#C carbone-carbone — chaînes acétyléniques non drug-like)
        n_alkynes = sum(
            1 for b in mol.GetBonds()
            if b.GetBondTypeAsDouble() == 3.0
            and b.GetBeginAtom().GetAtomicNum() == 6
            and b.GetEndAtom().GetAtomicNum() == 6
        )
        if n_alkynes > self.hp["max_alkynes"]:
            penalties -= (n_alkynes - self.hp["max_alkynes"]) * self.hp["alkyne_penalty_coef"]

        # Pas de cycle aromatique — cyclopropane ne suffit plus (v3.7)
        try:
            n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            if n_arom_rings == 0:
                penalties += self.hp["nonarom_penalty"]   # valeur négative
        except Exception:
            pass

        reward = 0.0

        # QED
        try:
            reward += self.hp["qed_weight"] * rdQED.qed(mol)
        except Exception:
            pass

        # LogP gaussien centré sur 2
        try:
            logp = Descriptors.MolLogP(mol)
            reward += self.hp["logp_weight"] * float(np.exp(-((logp - 2.0) ** 2) / 4.0))
        except Exception:
            pass

        # Bonus cycles aromatiques
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

        # IC50 prédit
        try:
            af  = self.feat.featurize(smiles)[np.newaxis]
            adj = np.ones((1, HP["max_atoms"], HP["max_atoms"]), dtype=np.float32)
            inp = (tf.constant(af), tf.constant(adj),
                   tf.zeros([1, HP["gex_dim"]]),
                   tf.zeros([1, HP["mut_dim"]]),
                   tf.zeros([1, HP["cnv_dim"]]))
            ic50_val = float(self.twin(inp, training=False)[0][0].numpy())
            reward  += self.hp["ic50_weight"] * float(
                np.exp(-((ic50_val - self.hp["target_ic50"]) ** 2) / 2.0))
        except Exception:
            pass

        # Diversité Tanimoto
        if self.past_fps:
            try:
                fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                sim = DataStructs.BulkTanimotoSimilarity(fp, self.past_fps)
                reward += (1.0 - max(sim)) * self.hp["diversity_weight"]
            except Exception:
                pass

        return float(np.clip(reward + penalties, -1.0, 10.0))

    @property
    def current_smiles(self) -> str:
        return self.vocab.decode(self.tokens)


# ─── Agent DQN ────────────────────────────────────────────────────────────────
class DQNDrugOptimizer:
    def __init__(self, twin, feat, vocab: SELFIESVocabulary, hp: dict = DQN_HP):
        self.twin, self.feat, self.vocab, self.hp = twin, feat, vocab, hp

        state_dim  = HP["latent_dim"] + vocab.vocab_size
        vocab_size = vocab.vocab_size

        self.q_online = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_online")
        self.q_target = QNetwork(state_dim, vocab_size, hp["hidden_dim"], name="q_target")
        self._sync_target()

        self.optimizer   = keras.optimizers.Adam(hp["lr"])
        self.replay      = ReplayBuffer(hp["replay_buffer_size"])
        self.loss_fn     = keras.losses.Huber()
        self.global_step = 0
        self.best_smiles = ""
        self.best_reward = -float("inf")
        self.past_fps: List = []

        print(f"[DQN-SELFIES v5.0] state_dim={state_dim} | vocab_size={vocab_size}")

    def _warmstart_buffer(self, z: tf.Tensor, seed_smiles: List[str]):
        """
        v5.0 — Pré-remplit le replay buffer avec des trajectoires expert.
        Pour chaque SMILES drug-like seed, encode en SELFIES token-par-token,
        calcule la récompense réelle à la fin, et injecte toutes les transitions.
        """
        n_ws = self.hp.get("warmstart_episodes", 500)
        smiles_pool = [s for s in seed_smiles if Chem.MolFromSmiles(s) is not None]
        if not smiles_pool:
            return
        injected = 0
        for _ in range(n_ws):
            smi = random.choice(smiles_pool)
            try:
                sel = sf.encoder(smi)
                if not sel:
                    continue
                token_ids = self.vocab.encode(sel)
            except Exception:
                continue

            env   = SELFIESEnv(self.twin, self.feat, self.vocab, z,
                               hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            for i, action in enumerate(token_ids):
                action = min(action, self.vocab.vocab_size - 1)
                next_state, reward, done = env.step(action)
                self.replay.push(state, action, np.float32(reward),
                                 next_state, np.float32(done))
                state = next_state
                if done:
                    break
            injected += 1

        print(f"[DQN v5.0] Warm-start: {injected} trajectoires expert injectees "
              f"({len(self.replay)} transitions dans le buffer)")

    def _sync_target(self):
        self.q_target.set_weights(self.q_online.get_weights())

    def _epsilon(self) -> float:
        t = min(self.global_step, self.hp["eps_decay_steps"])
        return self.hp["eps_start"] + t / self.hp["eps_decay_steps"] * (
            self.hp["eps_end"] - self.hp["eps_start"])

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self._epsilon():
            return self.vocab.random_token()
        q = self.q_online(
            tf.expand_dims(tf.constant(state, dtype=tf.float32), 0),
            training=False).numpy()[0]
        q[self.vocab.PAD_IDX] = -np.inf
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

        # v5.0 — warm-start : injecter des trajectoires expert avant l'entraînement
        if seed_smiles:
            self._warmstart_buffer(z, seed_smiles)

        rewards_hist, valid_count, top_mols = [], 0, []

        print(f"\n[DQN v5.0] {n_episodes} épisodes | vocab={self.vocab.vocab_size} tokens")
        print(f"  ε: {self.hp['eps_start']} → {self.hp['eps_end']} / {self.hp['eps_decay_steps']} steps")
        print(f"  Warm-start: {self.hp.get('warmstart_episodes', 0)} trajectoires expert")
        print(f"  Penalites : taille(>{self.hp['max_heavy_atoms']}) | "
              f"repetition(>{self.hp['max_token_repeat']}x, -{self.hp['repeat_penalty_coef']}/exc) | "
              f"stereo(>{self.hp['max_stereo_centers']}) | "
              f"charges(-{self.hp['charge_penalty_coef']}/atome) | "
              f"halogenes(>{self.hp['max_halogens']}) | "
              f"polynes(>{self.hp['max_alkynes']} C#C) | "
              f"sans_arom({self.hp['nonarom_penalty']})\n")

        for ep in range(1, n_episodes + 1):
            env   = SELFIESEnv(self.twin, self.feat, self.vocab, z,
                               hp=self.hp, past_fps=self.past_fps)
            state = env.reset()
            ep_r, ep_loss = 0.0, []

            while True:
                action              = self.select_action(state)
                state, reward, done = env.step(action)
                self.replay.push(state, action, np.float32(reward),
                                 state, np.float32(done))
                ep_r = reward if done else ep_r
                self.global_step += 1
                loss = self._learn()
                if loss is not None:
                    ep_loss.append(float(loss.numpy()))
                if self.global_step % self.hp["target_update_freq"] == 0:
                    self._sync_target()
                if done:
                    break

            smi = env.current_smiles
            rewards_hist.append(ep_r)

            if ep_r > -0.4 and smi:
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
                print(f"  Ep {ep:5d}/{n_episodes} | ε={self._epsilon():.3f} | "
                      f"R={ep_r:+.3f} | Moy50={mean_r:+.3f} | "
                      f"Valid={100*valid_count/ep:.1f}% | Loss={mean_l:.4f} | "
                      f"Best: {self.best_smiles[:40]!s:40s} ({self.best_reward:.3f})")

        print(f"\n[DQN v5.0] Terminé.")
        print(f"  Meilleur SMILES  : {self.best_smiles}")
        print(f"  Meilleure reward : {self.best_reward:.4f}")
        print(f"  Valides          : {valid_count}/{n_episodes} ({100*valid_count/n_episodes:.1f}%)")
        if top_mols:
            print(f"\n  Top 5 :")
            for i, (r, s) in enumerate(top_mols[:5], 1):
                print(f"    {i}. R={r:.3f}  {s}")
        return dict(best_smiles=self.best_smiles, best_reward=self.best_reward,
                    reward_trajectory=rewards_hist, valid_count=valid_count,
                    top_molecules=top_mols)

    def save_weights(self, path: str = "dqn_weights"):
        os.makedirs(path, exist_ok=True)
        self.q_online.save_weights(os.path.join(path, "q_online.weights.h5"))
        self.q_target.save_weights(os.path.join(path, "q_target.weights.h5"))
        print(f"[DQN] Poids → {path}/")

    def load_weights(self, path: str = "dqn_weights"):
        self.q_online.load_weights(os.path.join(path, "q_online.weights.h5"))
        self._sync_target()


# ─── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  DQN Drug Optimizer v5.1 — SELFIES + ChEMBL 10k corpus + Warm-start")
    print("=" * 70)

    # 1. Charger corpus ChEMBL
    corpus = load_chembl_smiles(n=10_000)

    # 2. Construire vocabulaire
    vocab = SELFIESVocabulary(corpus)

    # Test round-trip
    test_sel = sf.encoder("CC(=O)Nc1ccc(O)cc1")   # paracétamol
    print(f"[Vocab test] Paracétamol → {vocab.decode(vocab.encode(test_sel))}")

    # 3. Charger modèle
    dt_model   = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()

    # 4. Données omiques (synthétiques — remplacer par profil réel via fullPipeline)
    gex = tf.random.normal([1, HP["gex_dim"]])
    mut = tf.cast(tf.random.uniform([1, HP["mut_dim"]], 0, 2, dtype=tf.int32), tf.float32)
    cnv = tf.random.normal([1, HP["cnv_dim"]])

    # 5. Optimisation avec warm-start sur SEED_SMILES drug-like
    agent  = DQNDrugOptimizer(dt_model, featurizer, vocab, DQN_HP)
    result = agent.optimize(gex, mut, cnv, seed_smiles=SEED_SMILES)

    print(f"\n{'='*70}")
    print(f"  RÉSULTAT FINAL")
    print(f"{'='*70}")
    print(f"  Meilleur SMILES : {result['best_smiles']}")
    print(f"  Récompense      : {result['best_reward']:.4f}")
    print(f"  Valides         : {result['valid_count']}/{DQN_HP['n_episodes']} "
          f"({100*result['valid_count']/DQN_HP['n_episodes']:.1f}%)")

    agent.save_weights("dqn_weights_v5.1")
