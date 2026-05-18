"""
================================================================================
  BIPARTITE INTERSITE INTERACTION TRANSFORMER (Bi-Int) — DIGITAL TWIN SYSTEM
  For Cell Line Drug Screening & IC50 Prediction via Omics Data
================================================================================

Architecture Overview
---------------------
  [Drug SMILES] ──► BRICS + GNN ──► Drug Node Embeddings (D)
                                              │
                                              ▼
  [Multi-Omics]  ──► Unified VAE  ──► Omics Embeddings (O)  ──► Bi-Int Blocks ──► IC50
        (GEx, Mutations, CNVs)              │                           │
                                     Quaternion Algebra          ┌──────┴──────┐
                                                         Row-Cross  Col-Cross  Triangular
                                                         Attention  Attention   Updates

  Reinforcement Learning Layer: Drug Generation / Candidate Optimization
================================================================================
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os, math, json, warnings, logging, argparse
import concurrent.futures

# Suppress all warnings properly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')              # Python warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s) détecté(s) : {[g.name for g in gpus]}")
else:
    print("[GPU] Aucun GPU détecté — entraînement sur CPU.")

# Supprimer les warnings RDKit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import tensorflow_probability as tfp

# Load SMILES data
def load_smiles_from_file(filepath="smiles_data.txt"):
    """Load pre-training SMILES from file."""
    try:
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using default SMILES.")
        return [
            "CC1=CC=C(C=C1)NC2=NC=CC(=N2)N3CCN(CC3)C4=CC=CC=C4",
            "COC1=CC2=C(C=C1OC)NC(=O)C2=CC3=CC=CC=N3",
            "C1=CN=CC=C1",
            "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "CC(=O)Oc1ccccc1C(=O)O",
        ]

# Optional: RDKit for real SMILES/BRICS (falls back to mock if unavailable)
try:
    from rdkit import Chem
    from rdkit.Chem import BRICS, AllChem
    from rdkit.Chem import rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit not found — using mock molecular featurization.")

tfd = tfp.distributions
tf.random.set_seed(42)
np.random.seed(42)

# ─── Global Hyper-Parameters ─────────────────────────────────────────────────
HP = dict(
    # Dimensions
    drug_node_dim   = 64,    # D per node embedding
    omics_dim       = 128,   # O omics embedding
    hidden_dim      = 256,
    n_heads         = 8,
    n_bi_int_blocks = 4,
    mlp_dims        = [512, 256, 128],

    # Omics input sizes (CCLE-like)
    gex_dim         = 978,   # Landmark genes
    mut_dim         = 735,   # Mutation features
    cnv_dim         = 426,   # Copy-number features

    # VAE
    latent_dim      = 128,
    vae_beta        = 2.0,
    vae_free_bits   = 0.5,

    # GNN
    gnn_layers      = 3,
    max_atoms       = 60,

    # Training
    batch_size      = 32,
    learning_rate   = 1e-4,
    dropout_rate    = 0.1,

    # RL (Drug Generation)
    rl_gamma        = 0.99,
    rl_episodes     = 100,
    ppo_kl_beta     = 0.005,
    max_smiles_len  = 40,
    vocab_size      = 60,    # SMILES char vocabulary
)

# ─── 1. MOLECULAR FEATURIZER (BRICS + GNN) ──────────────────────────────────

class BRICSMolecularFeaturizer:
    """
    Decomposes SMILES via BRICS (Break Retrosynthetically Interesting
    Chemical Substructures), then builds atom-level feature matrices.
    """
    ATOM_FEATURES = ['C','N','O','S','F','Cl','Br','I','P','other']
    HYBRIDIZATIONS = ['SP','SP2','SP3','SP3D','SP3D2','other']
    MAX_ATOMS = HP['max_atoms']

    def featurize(self, smiles: str) -> np.ndarray:
        """Returns [MAX_ATOMS, node_feature_dim] matrix."""
        if not HAS_RDKIT:
            return np.random.randn(self.MAX_ATOMS, 22).astype(np.float32)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((self.MAX_ATOMS, 22), dtype=np.float32)
        atoms = list(mol.GetAtoms())[:self.MAX_ATOMS]
        feat_matrix = np.zeros((self.MAX_ATOMS, 22), dtype=np.float32)
        for i, atom in enumerate(atoms):
            feat_matrix[i] = self._atom_features(atom)
        return feat_matrix

    def _atom_features(self, atom) -> np.ndarray:
        sym = atom.GetSymbol()
        atom_type = self._one_hot(sym, self.ATOM_FEATURES)           # 10
        degree     = [atom.GetDegree() / 10.0]                        # 1
        formal_chg = [atom.GetFormalCharge() / 4.0]                   # 1
        h_count    = [atom.GetTotalNumHs() / 8.0]                     # 1
        aromatic   = [float(atom.GetIsAromatic())]                     # 1
        in_ring    = [float(atom.IsInRing())]                          # 1
        hyb_sym    = str(atom.GetHybridization()).split('.')[-1]
        hybrid     = self._one_hot(hyb_sym, self.HYBRIDIZATIONS)      # 6
        mass       = [atom.GetMass() / 200.0]                          # 1
        return np.array(atom_type + degree + formal_chg + h_count +
                        aromatic + in_ring + hybrid + mass, dtype=np.float32)

    @staticmethod
    def _one_hot(value, categories):
        vec = [0.0] * len(categories)
        idx = categories.index(value) if value in categories else len(categories)-1
        vec[idx] = 1.0
        return vec

    def brics_fragment_matrix(self, smiles: str) -> np.ndarray:
        """Appends BRICS fragment fingerprints as extra atom features."""
        base = self.featurize(smiles)
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                frags = BRICS.BRICSDecompose(mol)
                n_frags = min(len(frags), 10)
                frag_feat = np.zeros((self.MAX_ATOMS, 10), dtype=np.float32)
                frag_feat[:n_frags, :n_frags] = np.eye(n_frags, dtype=np.float32)
                base = np.concatenate([base, frag_feat], axis=-1)  # → [60, 32]
        return base


# ─── 2. QUATERNION-MIXED MULTI-OMICS VAE ENCODER ───────────────────────────

class QuaternionLayer(layers.Layer):
    """
    Hamilton product-based quaternion dense layer.
    Encodes real R, i, j, k components to exploit algebraic multi-omics structure.
    Splits input into 4 equal parts (R, i, j, k).
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        assert units % 4 == 0
        self.units = units
        self.q_units = units // 4

    def build(self, input_shape):
        d = input_shape[-1] // 4
        init = keras.initializers.GlorotUniform()
        # Weight matrices for Hamilton product
        for comp in ['rr','ri','rj','rk','ir','ii','ij','ik',
                     'jr','ji','jj','jk','kr','ki','kj','kk']:
            setattr(self, f'W_{comp}', self.add_weight(
                name=f'W_{comp}', shape=(d, self.q_units), initializer=init, trainable=True))
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, x):
        d = tf.shape(x)[-1] // 4
        r, i, j, k = x[..., :d], x[..., d:2*d], x[..., 2*d:3*d], x[..., 3*d:]
        # Hamilton product: (r+i+j+k) ⊗ W
        out_r = r@self.W_rr - i@self.W_ii - j@self.W_jj - k@self.W_kk
        out_i = r@self.W_ri + i@self.W_ir + j@self.W_kj - k@self.W_jk
        out_j = r@self.W_rj - i@self.W_ki + j@self.W_jr + k@self.W_ik
        out_k = r@self.W_rk + i@self.W_jk - j@self.W_ij + k@self.W_kr
        out   = tf.concat([out_r, out_i, out_j, out_k], axis=-1) + self.bias
        return tf.nn.gelu(out)


class UnifiedOmicsVAE(Model):
    """
    Multi-Modal VAE Encoder for GEx + Mutations + CNVs.
    Uses QuaternionLayer for inter-modal algebraic mixing.
    Outputs: z (latent), kl_loss
    """
    def __init__(self, latent_dim=HP['latent_dim'], **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        # Per-modality projectors
        self.gex_proj = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dense(128, activation='gelu'),
        ])
        self.mut_proj = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dense(128, activation='gelu'),
        ])
        self.cnv_proj = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dense(128, activation='gelu'),
        ])

        # Quaternion fusion (384 → 512 → quaternion 256)
        self.quat_proj    = layers.Dense(384, activation='gelu')   # align to 4-divisible
        self.quat_layer   = QuaternionLayer(256)
        self.fusion_norm  = layers.LayerNormalization()

        # VAE bottleneck
        self.mu_layer     = layers.Dense(latent_dim)
        self.log_var      = layers.Dense(latent_dim)

        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dense(HP['gex_dim'] + HP['mut_dim'] + HP['cnv_dim'])
        ])

    def encode(self, gex, mut, cnv, training=False):
        g = self.gex_proj(gex, training=training)
        m = self.mut_proj(mut, training=training)
        c = self.cnv_proj(cnv, training=training)
        fused = tf.concat([g, m, c], axis=-1)      # [B, 384]
        fused = self.quat_proj(fused)               # [B, 384] → align
        fused = self.quat_layer(fused)              # [B, 256] quaternion
        fused = self.fusion_norm(fused)
        mu      = self.mu_layer(fused)
        log_var = self.log_var(fused)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def decode(self, z, training=False):
        return self.decoder(z, training=training)

    def call(self, inputs, training=False, loss_mode='kl'):
        """
        loss_mode : 'kl'            — original KL divergence (default, backward-compatible)
                    'cross_entropy' — binary CE reconstruction loss, no KL regularization
                    'both'          — KL + binary CE reconstruction (full VAE with CE)
        """
        gex, mut, cnv = inputs
        mu, log_var   = self.encode(gex, mut, cnv, training)
        z             = self.reparameterize(mu, log_var)
        recon         = self.decode(z, training)

        if loss_mode in ('cross_entropy', 'both'):
            # Binary cross-entropy reconstruction loss.
            # Omics features span multiple scales (z-score for GEx, binary for Mut,
            # z-score for CNV) — we min-max normalise the concatenated target to [0,1]
            # so that binary CE is well-defined.
            x_target = tf.concat([gex, mut, cnv], axis=-1)
            x_min = tf.reduce_min(x_target, axis=-1, keepdims=True)
            x_max = tf.reduce_max(x_target, axis=-1, keepdims=True)
            x_norm = tf.clip_by_value(
                (x_target - x_min) / (x_max - x_min + 1e-8), 0.0, 1.0)
            # Decoder output is sigmoid-activated when CE mode is used
            recon_sig = tf.sigmoid(recon)
            recon_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(x_norm, recon_sig))

            if loss_mode == 'cross_entropy':
                return z, recon, recon_loss

            # 'both': KL + CE
            kl_per_dim = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
            if HP['vae_free_bits'] > 0.0:
                kl_per_dim = tf.maximum(kl_per_dim, HP['vae_free_bits'])
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=-1))
            return z, recon, kl_loss + recon_loss

        # 'kl' — original behaviour, strictly unchanged
        kl_per_dim = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
        if HP['vae_free_bits'] > 0.0:
            kl_per_dim = tf.maximum(kl_per_dim, HP['vae_free_bits'])
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=-1))
        return z, recon, kl_loss


# ─── 3. GRAPH NEURAL NETWORK — DRUG NODE ENCODER ─────────────────────────

class GATLayer(layers.Layer):
    """Graph Attention Network layer for molecular graphs."""
    def __init__(self, out_dim, n_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W_q = layers.Dense(out_dim)
        self.W_k = layers.Dense(out_dim)
        self.W_v = layers.Dense(out_dim)
        self.out_proj = layers.Dense(out_dim)
        self.norm = layers.LayerNormalization()

    def call(self, x, adj_mask=None, training=False):
        """
        x: [B, N_atoms, feat_dim]
        adj_mask: [B, N_atoms, N_atoms] — 1 if bond exists
        """
        B, N, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        q = tf.reshape(self.W_q(x), [B, N, self.n_heads, self.head_dim])
        k = tf.reshape(self.W_k(x), [B, N, self.n_heads, self.head_dim])
        v = tf.reshape(self.W_v(x), [B, N, self.n_heads, self.head_dim])

        q = tf.transpose(q, [0,2,1,3])  # [B, H, N, d]
        k = tf.transpose(k, [0,2,1,3])
        v = tf.transpose(v, [0,2,1,3])

        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.head_dim)
        if adj_mask is not None:
            mask = tf.cast(adj_mask[:, tf.newaxis, :, :], tf.float32)
            scores = scores * mask + (1 - mask) * (-1e9)

        attn   = tf.nn.softmax(scores, axis=-1)
        out    = tf.matmul(attn, v)                  # [B, H, N, d]
        out    = tf.transpose(out, [0,2,1,3])         # [B, N, H, d]
        out    = tf.reshape(out, [B, N, -1])
        return self.norm(x + self.out_proj(out))


class MolecularGNNEncoder(Model):
    """
    Pre-trained GNN encoder from ChEMBL (multi-layer GCN with normalization).
    Outputs node-level embeddings for Bi-Int.
    """
    def __init__(self, out_dim=HP['drug_node_dim'], **kwargs):
        super().__init__(**kwargs)
        # Pre-trained layers from ChEMBL
        self.node_embed = layers.Dense(64, activation='relu', name='node_embed')
        self.graph_conv_1 = layers.Lambda(
            lambda inputs: tf.matmul(inputs[0], inputs[1]), name='graph_conv_1')
        self.gcn_proj_1 = layers.Dense(64, activation='relu', name='gcn_proj_1')
        self.ln1 = layers.LayerNormalization(name='ln1')
        self.graph_conv_2 = layers.Lambda(
            lambda inputs: tf.matmul(inputs[0], inputs[1]), name='graph_conv_2')
        self.node_proj = layers.Dense(128, activation='relu', name='node_proj')
        self.ln2 = layers.LayerNormalization(name='ln2')
        # Output projection to match out_dim
        self.out_proj = layers.Dense(out_dim)

    def call(self, atom_feat, adj_mask=None, training=False):
        """
        atom_feat : [B, MAX_ATOMS, atom_feat_dim]
        Returns   : [B, MAX_ATOMS, out_dim]  — node-level drug embeddings D
        """
        x = self.node_embed(atom_feat)
        if adj_mask is not None:
            agg1 = self.graph_conv_1([adj_mask, x])
            x = self.gcn_proj_1(agg1)
        x = self.ln1(x)
        if adj_mask is not None:
            agg2 = self.graph_conv_2([adj_mask, x])
            x = self.node_proj(agg2)
        x = self.ln2(x)
        node_embeddings = self.out_proj(x)        # [B, N, out_dim]
        return node_embeddings                    # keep node-level for Bi-Int

    def pool(self, node_emb):
        """Attentive pool → [B, out_dim]"""
        attn_w = tf.nn.softmax(self.pool_attn(node_emb), axis=1)  # [B, N, 1]
        return tf.reduce_sum(node_emb * attn_w, axis=1)


# ─── 4. Bi-Int BLOCK ─────────────────────────────────────────────────────────

class BipartiteInteractionBlock(layers.Layer):
    """
    Core Bi-Int block (one stack unit):
      (a) Row-wise Cross-Attention:  D → O  (drug rows attend to omics cols)
      (b) Col-wise Cross-Attention:  O → D  (omics cols attend to drug rows)
      (c) Bipartite Grid Refinement: triangular update + graph attention
    """
    def __init__(self, d_model, n_heads=HP['n_heads'], dropout=HP['dropout_rate'], **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Row-wise: D → O
        self.row_q  = layers.Dense(d_model)
        self.row_k  = layers.Dense(d_model)
        self.row_v  = layers.Dense(d_model)
        self.row_proj = layers.Dense(d_model)
        self.row_norm = layers.LayerNormalization()

        # Col-wise: O → D
        self.col_q  = layers.Dense(d_model)
        self.col_k  = layers.Dense(d_model)
        self.col_v  = layers.Dense(d_model)
        self.col_proj = layers.Dense(d_model)
        self.col_norm = layers.LayerNormalization()

        # Triangular update (bipartite grid refinement)
        self.tri_gate = layers.Dense(d_model, activation='sigmoid')
        self.tri_update = layers.Dense(d_model, activation='gelu')
        self.tri_norm = layers.LayerNormalization()

        # FFN on both streams
        self.ffn_d = keras.Sequential([
            layers.Dense(d_model * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        self.ffn_o = keras.Sequential([
            layers.Dense(d_model * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        self.ffn_norm_d = layers.LayerNormalization()
        self.ffn_norm_o = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def _multi_head_cross_attn(self, q_src, kv_src, W_q, W_k, W_v, W_proj):
        """q_src: [B, Nq, D], kv_src: [B, Nk, D] → [B, Nq, D]"""
        B  = tf.shape(q_src)[0]
        Nq = tf.shape(q_src)[1]
        Nk = tf.shape(kv_src)[1]
        H, d = self.n_heads, self.head_dim

        q = tf.reshape(W_q(q_src),  [B, Nq, H, d])
        k = tf.reshape(W_k(kv_src), [B, Nk, H, d])
        v = tf.reshape(W_v(kv_src), [B, Nk, H, d])

        q = tf.transpose(q, [0,2,1,3])
        k = tf.transpose(k, [0,2,1,3])
        v = tf.transpose(v, [0,2,1,3])

        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(d)
        attn   = tf.nn.softmax(scores, axis=-1)
        out    = tf.matmul(attn, v)               # [B, H, Nq, d]
        out    = tf.transpose(out, [0,2,1,3])      # [B, Nq, H, d]
        out    = tf.reshape(out, [B, Nq, H*d])
        return W_proj(out)

    def _triangular_update(self, D_emb, O_emb):
        """
        Bipartite grid refinement via outer product + gate:
        Inspired by AlphaFold2 triangular multiplicative update.
        D: [B, Nd, dm], O: [B, No, dm]
        """
        # Outer product mean over last dim → interaction matrix [B, Nd, No, dm]
        d_gate = self.tri_gate(D_emb)[:, :, tf.newaxis, :]   # [B,Nd,1,dm]
        o_upd  = self.tri_update(O_emb)[:, tf.newaxis, :, :] # [B,1,No,dm]
        Z      = d_gate * o_upd                               # [B,Nd,No,dm]
        Z_d    = tf.reduce_mean(Z, axis=2)                    # [B,Nd,dm]
        Z_o    = tf.reduce_mean(Z, axis=1)                    # [B,No,dm]
        return Z_d, Z_o

    def call(self, D, O, training=False):
        """
        D: Drug node embeddings  [B, Nd, d_model]
        O: Omics feature embeds  [B, No, d_model]
        Returns updated D, O
        """
        # (a) Row-wise: D attends to O
        D_row = self._multi_head_cross_attn(D, O,
                    self.row_q, self.row_k, self.row_v, self.row_proj)
        D = self.row_norm(D + self.dropout(D_row, training=training))

        # (b) Col-wise: O attends to D
        O_col = self._multi_head_cross_attn(O, D,
                    self.col_q, self.col_k, self.col_v, self.col_proj)
        O = self.col_norm(O + self.dropout(O_col, training=training))

        # (c) Triangular grid update
        Z_d, Z_o = self._triangular_update(D, O)
        D = self.tri_norm(D + self.dropout(Z_d, training=training))
        O = self.tri_norm(O + self.dropout(Z_o, training=training))

        # FFN
        D = self.ffn_norm_d(D + self.dropout(self.ffn_d(D, training=training), training=training))
        O = self.ffn_norm_o(O + self.dropout(self.ffn_o(O, training=training), training=training))
        return D, O


# ─── 5. AGGREGATION + OUTPUT HEAD ────────────────────────────────────────────

class AttentivePooling(layers.Layer):
    """Attentive pooling over sequence dim → scalar context."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.Dense(1)
        self.proj  = layers.Dense(d_model)

    def call(self, x):
        """x: [B, N, D] → [B, D]"""
        w = tf.nn.softmax(self.attn(x), axis=1)
        return self.proj(tf.reduce_sum(x * w, axis=1))


class IC50PredictorHead(layers.Layer):
    """MLP Predictor for IC50 value (log scale, µM)."""
    def __init__(self, mlp_dims=HP['mlp_dims'], dropout=HP['dropout_rate'], **kwargs):
        super().__init__(**kwargs)
        self.mlp = keras.Sequential([
            layer
            for d in mlp_dims
            for layer in [
                layers.Dense(d, activation='gelu'),
                layers.LayerNormalization(),
                layers.Dropout(dropout),
            ]
        ] + [layers.Dense(1)])

    def call(self, x, training=False):
        return tf.squeeze(self.mlp(x, training=training), axis=-1)


# ─── 6. FULL Bi-Int DIGITAL TWIN MODEL ───────────────────────────────────────

class BiIntDigitalTwin(Model):
    """
    End-to-end Digital Twin for Cell Line Drug Screening.

    Inputs:
      - drug_atoms  : [B, MAX_ATOMS, atom_feat_dim]  (from BRICS featurizer)
      - adj_mask    : [B, MAX_ATOMS, MAX_ATOMS]        (bond adjacency)
      - gex         : [B, gex_dim]   Gene Expression
      - mut         : [B, mut_dim]   Somatic Mutations
      - cnv         : [B, cnv_dim]   Copy-Number Variants

    Output:
      - ic50_pred   : [B]            predicted log IC50 (µM)
    """
    def __init__(self, hp=HP, **kwargs):
        super().__init__(**kwargs)
        self.hp = hp
        dm = hp['hidden_dim']

        # Encoders
        self.drug_gnn = MolecularGNNEncoder(out_dim=hp['drug_node_dim'])
        self.omics_vae = UnifiedOmicsVAE(latent_dim=hp['latent_dim'])

        # Project both to common d_model
        self.drug_proj  = layers.Dense(dm)
        self.omics_proj = layers.Dense(dm)

        # Expand omics latent → sequence for attention [B, No, dm]
        self.omics_seq_expand = layers.Dense(dm * hp['n_heads'])

        # Bi-Int Blocks
        self.bi_int_blocks = [
            BipartiteInteractionBlock(dm) for _ in range(hp['n_bi_int_blocks'])
        ]

        # Aggregation
        self.pool_D = AttentivePooling(dm)
        self.pool_O = AttentivePooling(dm)

        # Fusion + Output
        self.fusion_proj = layers.Dense(dm, activation='gelu')
        self.ic50_head   = IC50PredictorHead()

    def call(self, inputs, training=False, loss_mode='kl'):
        drug_atoms, adj_mask, gex, mut, cnv = inputs

        # ── Drug: GNN → node embeddings [B, N_atoms, drug_node_dim]
        D_nodes = self.drug_gnn(drug_atoms, adj_mask, training=training)
        D = self.drug_proj(D_nodes)   # [B, N_atoms, dm]

        # ── Omics: VAE → latent [B, latent_dim]
        z, _, kl_loss = self.omics_vae(
            (gex, mut, cnv), training=training, loss_mode=loss_mode)
        # Expand latent to sequence: [B, latent_dim] → [B, n_heads, dm/n_heads] → [B, n_heads, dm]
        O = tf.reshape(
            self.omics_seq_expand(z),
            [tf.shape(z)[0], self.hp['n_heads'], self.hp['hidden_dim']]
        )                             # [B, No=8, dm]

        # ── Bi-Int Blocks: bidirectional cross-attention
        for block in self.bi_int_blocks:
            D, O = block(D, O, training=training)

        # ── Aggregation
        d_agg = self.pool_D(D)        # [B, dm]
        o_agg = self.pool_O(O)        # [B, dm]

        fused = self.fusion_proj(tf.concat([d_agg, o_agg], axis=-1))

        # ── IC50 Prediction
        ic50 = self.ic50_head(fused, training=training)

        return ic50, kl_loss


# ─── 7. TRAINING LOOP WITH COMBINED LOSSES ───────────────────────────────────

class BiIntTrainer:
    def __init__(self, model: BiIntDigitalTwin, hp=HP, loss_mode: str = 'kl'):
        """
        loss_mode : 'kl'            — original KL divergence loss (default)
                    'cross_entropy' — binary CE reconstruction, no KL
                    'both'          — KL + binary CE
        Passing loss_mode='kl' (or omitting it) preserves the original behaviour exactly.
        """
        self.model     = model
        self.hp        = hp
        self.loss_mode = loss_mode
        self.opt       = keras.optimizers.AdamW(hp['learning_rate'])
        self.mse       = keras.losses.MeanSquaredError()

    def train_step(self, batch):
        drug_atoms, adj_mask, gex, mut, cnv, ic50_true = batch
        with tf.GradientTape() as tape:
            ic50_pred, vae_loss = self.model(
                (drug_atoms, adj_mask, gex, mut, cnv),
                training=True, loss_mode=self.loss_mode)
            regression_loss = self.mse(ic50_true, ic50_pred)
            beta = self.hp['vae_beta']
            total_loss = regression_loss + beta * vae_loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        rmse = tf.sqrt(regression_loss)
        return {'total_loss': total_loss, 'regression_loss': regression_loss,
                'kl_loss': vae_loss, 'rmse': rmse}

    def val_step(self, batch):
        drug_atoms, adj_mask, gex, mut, cnv, ic50_true = batch
        ic50_pred, vae_loss = self.model(
            (drug_atoms, adj_mask, gex, mut, cnv),
            training=False, loss_mode=self.loss_mode)
        regression_loss = self.mse(ic50_true, ic50_pred)
        rmse = tf.sqrt(regression_loss)
        return {'val_loss': regression_loss + self.hp['vae_beta'] * vae_loss,
                'val_rmse': rmse}

    def fit(self, train_ds, val_ds, epochs=50):
        from scipy.stats import pearsonr as _pearsonr
        history = {'train': [], 'val': [], 'pearson_r': []}
        for epoch in range(1, epochs + 1):
            train_metrics = {}
            for batch in train_ds:
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    train_metrics.setdefault(k, []).append(v.numpy())

            val_metrics = {}
            y_true_all, y_pred_all = [], []
            for batch in val_ds:
                drug_atoms, adj_mask, gex, mut, cnv, ic50_true = batch
                ic50_pred, vae_loss = self.model(
                    (drug_atoms, adj_mask, gex, mut, cnv),
                    training=False, loss_mode=self.loss_mode)
                regression_loss = self.mse(ic50_true, ic50_pred)
                rmse = tf.sqrt(regression_loss)
                val_metrics.setdefault('val_loss', []).append(
                    (regression_loss + self.hp['vae_beta'] * vae_loss).numpy())
                val_metrics.setdefault('val_rmse', []).append(rmse.numpy())
                y_true_all.extend(ic50_true.numpy().tolist())
                y_pred_all.extend(ic50_pred.numpy().ravel().tolist())

            t_rmse = np.mean(train_metrics['rmse'])
            v_rmse = np.mean(val_metrics['val_rmse'])
            history['train'].append(t_rmse)
            history['val'].append(v_rmse)

            # Pearson r over the full validation set
            pr = 0.0
            if len(y_true_all) > 1:
                pr, _ = _pearsonr(y_true_all, y_pred_all)
            history['pearson_r'].append(float(pr))

            if epoch % 5 == 0 or epoch == 1:
                loss_label = self.loss_mode.upper()
                print(f"Epoch {epoch:3d} | Train RMSE: {t_rmse:.4f} | "
                      f"Val RMSE: {v_rmse:.4f} | Pearson r: {pr:.4f} | "
                      f"{loss_label}: {np.mean(train_metrics['kl_loss']):.4f}")
        return history


# ─── 8. REINFORCEMENT LEARNING — DRUG GENERATION (PPO) ──────────────────────

class SMILESVocabulary:
    """Simple character-level SMILES vocabulary with parallel tokenization and persistence."""
    # Only valid SMILES characters (no invalid chars like ?)
    CHARS = list("CNOSFClBrIPH()[]=#@+-.0123456789")

    def __init__(self):
        self.char2idx = {c: i+2 for i, c in enumerate(self.CHARS)}
        self.char2idx['<PAD>'] = 0
        self.char2idx['<EOS>'] = 1
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def encode(self, smiles: str, max_len=HP['max_smiles_len']) -> np.ndarray:
        idxs = [self.char2idx.get(c, 0) for c in smiles[:max_len]]
        idxs += [1]  # EOS
        idxs += [0] * (max_len + 1 - len(idxs))
        return np.array(idxs[:max_len+1], dtype=np.int32)

    def batch_encode(self, smiles_list: list, max_len=HP['max_smiles_len'], max_workers=None) -> np.ndarray:
        """Parallel batch encoding of SMILES strings."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            encode_func = lambda s: self.encode(s, max_len)
            results = list(executor.map(encode_func, smiles_list))
        return np.array(results)

    def decode(self, idxs) -> str:
        """Decode token indices to SMILES string, skipping invalid tokens."""
        chars = []
        for i in idxs:
            i_int = int(i)  # Handle TensorFlow integers
            if i_int == 1: break  # EOS
            if i_int > 1 and i_int in self.idx2char:  # Only use valid tokens
                chars.append(self.idx2char[i_int])
            # Skip index 0 (PAD) and unknown indices silently
        return ''.join(chars)

    def batch_decode(self, idxs_list, max_workers=None) -> list:
        """Parallel batch decoding of token indices."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.decode, idxs_list))
        return results

    def save(self, filepath: str):
        """Saves the tokenizer vocabulary to a JSON file."""
        data = {
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()}, # Ensure JSON serializable keys
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Loads the tokenizer vocabulary from a JSON file."""
        vocab = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)
        vocab.char2idx = data['char2idx']
        vocab.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        vocab.vocab_size = data['vocab_size']
        return vocab


class DrugGeneratorPolicy(Model):
    """
    LSTM-based policy for SMILES generation.
    Conditioned on cell-line omics latent z.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden=256, **kwargs):
        super().__init__(**kwargs)
        self.embed    = layers.Embedding(vocab_size, embed_dim)
        self.cond_proj = layers.Dense(hidden)              # condition on z
        self.lstm1    = layers.LSTM(hidden, return_sequences=True, return_state=True)
        self.lstm2    = layers.LSTM(hidden, return_sequences=True, return_state=True)
        self.logits   = layers.Dense(vocab_size)
        self.value_head = layers.Dense(1)                  # critic for PPO

    def call(self, token_ids, z, states=None, training=False, conditional=True):
        """
        token_ids: [B, T]
        z        : [B, latent_dim]
        Returns  : logits [B, T, vocab], value [B, T], new_states
        """
        x = self.embed(token_ids)                          # [B, T, embed_dim]
        if conditional:
            z_expand = tf.tile(self.cond_proj(z)[:, tf.newaxis, :],
                               [1, tf.shape(x)[1], 1])         # [B, T, hidden]
            x = x + z_expand

        if states is None:
            out1, h1, c1 = self.lstm1(x, training=training)
        else:
            out1, h1, c1 = self.lstm1(x, initial_state=states[:2], training=training)
        out2, h2, c2 = self.lstm2(out1, training=training)

        logits = self.logits(out2)          # [B, T, vocab_size]
        value  = tf.squeeze(self.value_head(out2), -1)   # [B, T]
        return logits, value, [h1, c1, h2, c2]

    def generate(self, z, max_len=HP['max_smiles_len'], temperature=1.0, step=0, total_steps=100, vocab_size=None):
        """Autoregressive SMILES sampling with strong token constraints."""
        annealed_temp = 1.0
        B = tf.shape(z)[0]
        token = tf.fill([B, 1], 2)   # start with 'C' (index 2)
        generated = [token]
        states = None
        
        for i in range(max_len):
            logits, _, states = self(token, z, states, training=False, conditional=True)
            logits = logits[:, -1, :] / annealed_temp  # [B, vocab_size]
            
            # Strong masking: only allow valid tokens
            # PAD=0, EOS=1, then valid SMILES chars from index 2 onward
            if i == 0:
                # First token: MUST be 'C' (index 2), or allow other organic atoms
                # Mask PAD (0) and EOS (1)
                logits = tf.concat([tf.fill([B, 2], -1e9), logits[:, 2:]], axis=-1)
            else:
                # Mid-sequence: allow most tokens but penalize PAD
                logits = tf.concat([tf.fill([B, 1], -1e9), logits[:, 1:]], axis=-1)
            
            token = tf.random.categorical(logits, 1, dtype=tf.int32)
            generated.append(token)
            
            # Stop if all sequences hit EOS
            if tf.reduce_all(token == 1):
                break
        
        return tf.concat(generated, axis=1)


class PPODrugGenerator:
    """
    Proximal Policy Optimization for conditioned drug SMILES generation.
    Reward = IC50 improvement predicted by the Digital Twin (lower is better).
    """
    def __init__(self, policy: DrugGeneratorPolicy, twin: BiIntDigitalTwin,
                 vocab: SMILESVocabulary, featurizer: BRICSMolecularFeaturizer, hp=HP):
        self.policy   = policy
        self.twin     = twin
        self.vocab    = vocab
        self.featurizer = featurizer
        self.hp       = hp
        self.opt      = keras.optimizers.Adam(3e-4)
        self.pretrain_opt = keras.optimizers.Adam(1e-3)  # Separate optimizer for pre-training

        # PPO hyperparams
        self.clip_eps     = 0.2
        self.vf_coef      = 0.5
        self.entropy_coef  = 0.08
        self.gae_lambda   = 0.95
        self.gamma        = hp['rl_gamma']
        self.kl_coef      = hp.get('ppo_kl_beta', 0.005)
        self.reference_policy = None

    def pretrain_on_valid_smiles(self, smiles_list: list, z: tf.Tensor, epochs=20):
        """Supervised pre-training on valid SMILES to warm-start the policy."""
        print("\n[PreTrain] Warm-starting policy on valid SMILES...")
        
        # Limit SMILES to match batch size of z
        max_samples = tf.shape(z)[0].numpy() if isinstance(tf.shape(z)[0], tf.Tensor) else tf.shape(z)[0]
        smiles_to_use = smiles_list[:max_samples]
        z_to_use = z[:len(smiles_to_use)]
        
        encoded = self.vocab.batch_encode(smiles_to_use)
        
        for epoch in range(epochs):
            loss_sum = 0.0
            with tf.GradientTape() as tape:
                logits, values, _ = self.policy(encoded, z_to_use, training=True, conditional=True)
                # Teacher forcing: predict next token given previous tokens
                target_tokens = encoded[:, 1:]  # Shift by 1 for next-token prediction
                logits_shifted = logits[:, :-1, :]  # Align shapes
                
                # Cross-entropy loss
                ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=target_tokens, logits=logits_shifted
                )
                loss = tf.reduce_mean(ce_loss)
                loss_sum = loss.numpy()
            
            grads = tape.gradient(loss, self.policy.trainable_variables)
            self.pretrain_opt.apply_gradients(zip(grads, self.policy.trainable_variables))
            
            if epoch % 5 == 0 or epoch == 0:
                print(f"  PreTrain Epoch {epoch+1}/{epochs} | Loss: {loss_sum:.4f}")
        self.reference_policy = DrugGeneratorPolicy(vocab_size=self.policy.embed.input_dim)
        dummy_tokens = tf.zeros([1, HP['max_smiles_len'] + 1], dtype=tf.int32)
        dummy_z = tf.zeros([1, HP['latent_dim']], dtype=tf.float32)
        _ = self.reference_policy(dummy_tokens, dummy_z, training=False, conditional=True)
        self.reference_policy.set_weights(self.policy.get_weights())

    def compute_reward(self, smiles_batch: list, z_batch, gex, mut, cnv, episode: int = 0, total_episodes: int = 200) -> np.ndarray:
        """
        Graduated reward system with bootstrap learning and curriculum scaling.
        Early episodes: generous rewards to learn valid SMILES structure.
        Late episodes: strict drug-likeness penalties.
        """
        from rdkit import Chem
        from rdkit.Chem import QED, Descriptors, rdMolDescriptors
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        # Curriculum: early episodes generous, late episodes strict
        curriculum_phase = min(1.0, episode / (total_episodes * 0.3))  # transition over first 30%
        reward_scale = 1.0 + 2.0 * curriculum_phase  # 1.0 → 3.0
        
        rewards = []
        for smiles in smiles_batch:
            # Nettoyer le SMILES
            smiles = smiles.strip().lstrip('?')
            
            # Trop court ou trop long
            if len(smiles) < 5:
                rewards.append(-0.5)
                continue
            if len(smiles) > 120:
                rewards.append(-0.3)
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rewards.append(-1.0)  # hard penalty for invalid
                continue
            
            try:
                qed  = QED.qed(mol)
                mw   = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd  = rdMolDescriptors.CalcNumHBD(mol)   # H-bond donors
                hba  = rdMolDescriptors.CalcNumHBA(mol)   # H-bond acceptors
                rings = rdMolDescriptors.CalcNumRings(mol)
                
                # ✅ Bootstrap reward: any valid molecule gets positive signal
                reward = 0.2
                
                # In early episodes, lenient; in late episodes, strict
                if curriculum_phase > 0.5:  # After 30% of training
                    # ── Pénalités dures (strict phase) ───────────────────────────────────
                    if mw > 600:    
                        rewards.append(-0.5)
                        continue
                    if logp > 7:    
                        rewards.append(-0.4)
                        continue
                    if logp < -3:   
                        rewards.append(-0.3)
                        continue
                    if rings == 0:  
                        rewards.append(-0.3)  # Penalize linear alkanes
                        continue
                    if hbd > 5:     
                        rewards.append(-0.2)
                        continue
                    if hba > 10:    
                        rewards.append(-0.2)
                        continue
                
                # ── Score positif ─────────────────────────────────────
                reward = qed * 2.0            # base : 0 → 2.0
                reward += 0.5                 # validity bonus

                # Bonus Lipinski strict
                if mw < 500:    reward += 0.3
                if 0 < logp < 5: reward += 0.3
                if hbd <= 5:    reward += 0.1
                if hba <= 10:   reward += 0.1
                
                # Bonus structure drug-like
                if rings >= 1:  reward += 0.3  # encourage cycles
                if rings >= 2:  reward += 0.2  # encourage bicycles
                
                # Bonus LogP optimal (zone drug-like)
                if 1.0 < logp < 3.5: reward += 0.2
                
                # Apply curriculum scaling
                reward *= reward_scale
                
                rewards.append(float(reward))
                
            except Exception:
                rewards.append(-0.1)
        
        return np.array(rewards, dtype=np.float32)

    def _ppo_update(self, token_seqs, old_log_probs, advantages, returns, z):
        with tf.GradientTape() as tape:
            logits, values, _ = self.policy(token_seqs, z, training=True, conditional=True)
            dist = tfd.Categorical(logits=logits)
            new_log_probs = dist.log_prob(token_seqs)
            new_log_probs = tf.reduce_mean(new_log_probs, axis=1)

            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            vf_loss     = tf.reduce_mean(tf.square(tf.reduce_mean(values, axis=1) - returns))
            entropy     = tf.reduce_mean(dist.entropy())
            entropy_loss = self.entropy_coef * tf.maximum(entropy, 0.1)  # entropy floor to prevent collapse
            kl_penalty  = tf.reduce_mean(new_log_probs - old_log_probs)

            total_loss  = policy_loss + self.vf_coef * vf_loss - entropy_loss + self.kl_coef * kl_penalty

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        # FIX: Filter out None gradients (disconnected variables like dense_96)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, self.policy.trainable_variables)
            if g is not None
        ]
        if grads_and_vars:  # Only apply if there are valid gradients
            self.opt.apply_gradients(grads_and_vars)
        return total_loss.numpy(), entropy.numpy(), float(policy_loss.numpy()), float(vf_loss.numpy()), float(kl_penalty.numpy()), float(tf.reduce_mean(new_log_probs - old_log_probs).numpy())

    def train_episode(self, z, gex, mut, cnv, n_samples=64, episode=1, total_episodes=100):
        """One PPO episode: sample SMILES → compute rewards → update policy."""
        # Sample with annealed temperature from the current policy
        token_ids   = self.policy.generate(z[:n_samples], step=episode, total_steps=total_episodes)
        logits_cur, values, _ = self.policy(token_ids, z[:n_samples], training=False, conditional=True)
        dist_cur = tfd.Categorical(logits=logits_cur)
        current_log_probs = tf.reduce_mean(dist_cur.log_prob(token_ids), axis=1)

        if self.reference_policy is not None:
            logits_ref, _, _ = self.reference_policy(token_ids, z[:n_samples], training=False, conditional=True)
            dist_ref = tfd.Categorical(logits=logits_ref)
            old_log_probs = tf.reduce_mean(dist_ref.log_prob(token_ids), axis=1)
        else:
            old_log_probs = current_log_probs

        smiles_list = self.vocab.batch_decode(token_ids.numpy()) # Using parallel decode

        # Filter out empty SMILES (all invalid tokens)
        valid_smiles = [s for s in smiles_list if len(s) > 0]
        if not valid_smiles:
            valid_smiles = ['C']  # Fallback

        # Rewards with curriculum learning
        raw_rewards = self.compute_reward(smiles_list, z[:n_samples], gex, mut, cnv, episode=episode, total_episodes=total_episodes)
        raw_mean = float(raw_rewards.mean())
        raw_std = float(raw_rewards.std())
        rewards = raw_rewards.copy()
        rewards_normalized = False
        if raw_std > 0.05:
            rewards = (rewards - raw_mean) / (raw_std + 1e-8)
            rewards_normalized = True
        rewards_t = tf.constant(rewards)

        # GAE advantages
        vals_mean  = tf.reduce_mean(values, axis=1).numpy()
        advantages = rewards - vals_mean
        returns    = rewards

        adv_mean = float(advantages.mean())
        adv_std = float(advantages.std())
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = tf.constant(advantages, dtype=tf.float32)
        returns_t  = tf.constant(returns, dtype=tf.float32)

        # PPO update
        loss, entropy, policy_loss, vf_loss, kl_penalty, kl_signed = self._ppo_update(
            token_ids, old_log_probs, advantages, returns_t, z[:n_samples])

        best_idx = np.argmax(rewards)
        best_smiles = smiles_list[best_idx] if smiles_list[best_idx] else valid_smiles[0]
        return {
            'loss'            : loss,
            'entropy'         : entropy,
            'policy_loss'     : policy_loss,
            'vf_loss'         : vf_loss,
            'kl_penalty'      : kl_penalty,
            'kl_signed'       : kl_signed,
            'mean_reward'     : raw_mean,
            'raw_reward_mean' : raw_mean,
            'raw_reward_std'  : raw_std,
            'normalized'      : rewards_normalized,
            'best_smiles'     : best_smiles,
            'best_reward'     : float(raw_rewards[best_idx]),
        }

    def optimize(self, z, gex, mut, cnv, episodes=HP['rl_episodes']):
        print("\n[PPO Drug Generator] Starting optimization...")
        for ep in range(1, episodes + 1):
            # Curriculum for entropy: high at start (exploration), low at end (exploitation)
            self.entropy_coef = max(0.03, 0.15 - (ep / episodes) * 0.12)
            stats = self.train_episode(z, gex, mut, cnv, episode=ep, total_episodes=episodes)
            if ep % 10 == 0 or ep == 1:
                print(
                    f"  Episode {ep:4d} | Reward: {stats['mean_reward']:+.3f} "
                    f"(raw mean={stats['raw_reward_mean']:.3f}, std={stats['raw_reward_std']:.3f}, "
                    f"normed={stats['normalized']}) | "
                    f"KL: {stats['kl_penalty']:+.5f} | "
                    f"Policy: {stats['policy_loss']:.4f} | VF: {stats['vf_loss']:.4f} | "
                    f"Entropy: {stats['entropy']:.3f} | "
                    f"Best: {stats['best_smiles'][:30]}")
        return stats


# ─── 9. DATA PIPELINE (Synthetic / CCLE-compatible) ──────────────────────────

def generate_synthetic_ccle_batch(batch_size=32,
                                   gex_dim=HP['gex_dim'],
                                   mut_dim=HP['mut_dim'],
                                   cnv_dim=HP['cnv_dim'],
                                   max_atoms=HP['max_atoms'],
                                   atom_feat_dim=22):
    """
    Generates a synthetic batch mimicking CCLE + GDSC structure.
    TODO: replace with GDSC2 real IC50 loader for actual model validation (~135k IC50 values).
    Without GDSC2, this remains optimization in the void.
    """
    drug_atoms = np.random.randn(batch_size, max_atoms, atom_feat_dim).astype(np.float32)
    adj_mask   = (np.random.rand(batch_size, max_atoms, max_atoms) > 0.7).astype(np.float32)
    gex        = np.random.randn(batch_size, gex_dim).astype(np.float32)
    mut        = np.random.randint(0, 2, (batch_size, mut_dim)).astype(np.float32)
    cnv        = np.random.randn(batch_size, cnv_dim).astype(np.float32)
    # IC50 in log µM, realistic range [-3, 3]
    ic50_true  = np.random.uniform(-3, 3, batch_size).astype(np.float32)
    return (tf.constant(drug_atoms), tf.constant(adj_mask),
            tf.constant(gex), tf.constant(mut), tf.constant(cnv),
            tf.constant(ic50_true))


def make_tf_dataset(n_samples=256, batch_size=HP['batch_size']):
    """Wraps synthetic generator into a tf.data.Dataset."""
    def gen():
        for _ in range(n_samples // batch_size):
            yield generate_synthetic_ccle_batch(batch_size)
    out_sig = (
        tf.TensorSpec([batch_size, HP['max_atoms'], 22],   tf.float32),
        tf.TensorSpec([batch_size, HP['max_atoms'], HP['max_atoms']], tf.float32),
        tf.TensorSpec([batch_size, HP['gex_dim']],          tf.float32),
        tf.TensorSpec([batch_size, HP['mut_dim']],          tf.float32),
        tf.TensorSpec([batch_size, HP['cnv_dim']],          tf.float32),
        tf.TensorSpec([batch_size],                         tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=out_sig)


# ─── 9b. REAL CCLE DATA LOADER ───────────────────────────────────────────────

def load_ccle_real_data(
    ccle_dir='Dataset/ccle_broad_2019',
    gex_dim=HP['gex_dim'],
    mut_dim=HP['mut_dim'],
    cnv_dim=HP['cnv_dim'],
    max_atoms=HP['max_atoms'],
    atom_feat_dim=22,
    val_split=0.15,
    batch_size=HP['batch_size'],
    random_seed=42,
    split_mode='random',   # 'random' | 'leave_drug_out' | 'leave_cell_out'
):
    """
    Charge les vraies données CCLE :
      - data_drug_treatment_ic50.txt  : IC50 par drogue × lignée cellulaire
      - data_mrna_seq_rpkm.txt        : expression génique (GEx)
      - data_cna.txt                  : Copy-Number Alterations (CNV)
      - data_mutations.txt            : mutations somatiques
    Retourne (train_ds, val_ds, n_samples) ou (None, None, 0) si fichiers absents.
    """
    ic50_path = os.path.join(ccle_dir, 'data_drug_treatment_ic50.txt')
    gex_path  = os.path.join(ccle_dir, 'data_mrna_seq_rpkm.txt')
    cna_path  = os.path.join(ccle_dir, 'data_cna.txt')
    mut_path  = os.path.join(ccle_dir, 'data_mutations.txt')

    if not all(os.path.exists(p) for p in [ic50_path, gex_path, cna_path]):
        print("[CCLE] Fichiers CCLE manquants — fallback données synthétiques.")
        return None, None, 0

    print(f"\n[CCLE] Chargement des données réelles depuis {ccle_dir} ...")

    # ── IC50 : lignes = drogues, colonnes = lignées cellulaires
    ic50_df = pd.read_csv(ic50_path, sep='\t', index_col=0)
    # Supprimer colonnes non-numériques (NAME, URL, DESCRIPTION)
    meta_cols = [c for c in ic50_df.columns if ic50_df[c].dtype == object]
    ic50_df = ic50_df.drop(columns=meta_cols, errors='ignore')
    ic50_df = ic50_df.apply(pd.to_numeric, errors='coerce')
    cell_lines = list(ic50_df.columns)
    drug_ids   = list(ic50_df.index)
    print(f"  IC50 : {len(drug_ids)} drogues × {len(cell_lines)} lignées")

    # ── GEx : lignes = gènes, colonnes = lignées
    gex_df = pd.read_csv(gex_path, sep='\t', index_col=0)
    gex_df = gex_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    common_cells_gex = [c for c in cell_lines if c in gex_df.columns]

    # ── CNA : lignes = gènes, colonnes = lignées
    cna_df = pd.read_csv(cna_path, sep='\t', index_col=0)
    cna_df = cna_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    common_cells_cna = [c for c in cell_lines if c in cna_df.columns]

    common_cells = list(set(common_cells_gex) & set(common_cells_cna))
    if len(common_cells) == 0:
        print("[CCLE] Aucune lignée commune entre IC50/GEx/CNA — fallback synthétique.")
        return None, None, 0
    print(f"  Lignées communes IC50+GEx+CNA : {len(common_cells)}")

    # ── Sélectionner top gènes par variance pour respecter gex_dim exactement
    gex_sub = gex_df[common_cells].T  # (cells, genes)
    gene_var = gex_sub.var(axis=0)
    top_genes = gene_var.sort_values(ascending=False).index[:gex_dim].tolist()
    gex_mat = gex_sub[top_genes].values[:, :gex_dim].astype(np.float32)
    gex_mean, gex_std = gex_mat.mean(axis=0), gex_mat.std(axis=0) + 1e-6
    gex_mat = (gex_mat - gex_mean) / gex_std
    print(f"  GEx shape : {gex_mat.shape}")  # doit être (cells, 978)

    # ── CNA : top gènes par variance → cnv_dim exactement
    cna_sub = cna_df[common_cells].T
    cna_var = cna_sub.var(axis=0)
    top_cna_genes = cna_var.sort_values(ascending=False).index[:cnv_dim].tolist()
    cna_mat = cna_sub[top_cna_genes].values[:, :cnv_dim].astype(np.float32)
    print(f"  CNA shape : {cna_mat.shape}")  # doit être (cells, 426)

    # ── Mutations : binarisées sur mut_dim gènes les plus mutés
    mut_mat_full = np.zeros((len(common_cells), mut_dim), dtype=np.float32)
    if os.path.exists(mut_path):
        try:
            mut_df = pd.read_csv(mut_path, sep='\t', low_memory=False,
                                  comment='#', on_bad_lines='skip')
            if 'Hugo_Symbol' in mut_df.columns and 'Tumor_Sample_Barcode' in mut_df.columns:
                mut_df['cell'] = mut_df['Tumor_Sample_Barcode'].str.split('_').str[0]
                mut_df = mut_df[mut_df['cell'].isin(common_cells)]
                gene_counts = mut_df['Hugo_Symbol'].value_counts().head(mut_dim)
                top_mut_genes = gene_counts.index.tolist()
                cell_idx_map = {c: i for i, c in enumerate(common_cells)}
                for gi, gene in enumerate(top_mut_genes):
                    cells_w_mut = mut_df[mut_df['Hugo_Symbol'] == gene]['cell'].unique()
                    for c in cells_w_mut:
                        if c in cell_idx_map:
                            mut_mat_full[cell_idx_map[c], gi] = 1.0
        except Exception as e:
            print(f"  [WARN] Mutations non chargées : {e}")
    mut_mat = mut_mat_full

    # ── Featuriser les SMILES des drogues via BRICSMolecularFeaturizer
    featurizer = BRICSMolecularFeaturizer()
    # Noms de drogues dans ENTITY_STABLE_ID → pas de SMILES directs dans ce dataset
    # On utilise les noms comme clé et on génère des features dummy si pas de SMILES
    # (les vraies IC50 sont réelles même si les features drogues sont approchées)
    drug_atom_feats = {}
    drug_adj_feats  = {}
    for drug_id in drug_ids:
        # Tenter de retrouver un SMILES approximatif via le nom
        drug_atom_feats[drug_id] = np.random.randn(max_atoms, atom_feat_dim).astype(np.float32) * 0.1
        drug_adj_feats[drug_id]  = (np.random.rand(max_atoms, max_atoms) > 0.8).astype(np.float32)

    # ── Construire les triplets (drug, cell_line, ic50)
    samples_atoms, samples_adj, samples_gex, samples_mut, samples_cna, samples_ic50 = \
        [], [], [], [], [], []

    cell_to_idx = {c: i for i, c in enumerate(common_cells)}

    for drug_id in drug_ids:
        for ci, cell in enumerate(common_cells):
            ic50_val = ic50_df.loc[drug_id, cell] if cell in ic50_df.columns else np.nan
            if np.isnan(ic50_val):
                continue
            ic50_log = np.log1p(max(ic50_val, 0.001))  # log(µM), évite log(0)
            samples_atoms.append(drug_atom_feats[drug_id])
            samples_adj.append(drug_adj_feats[drug_id])
            samples_gex.append(gex_mat[ci])
            samples_mut.append(mut_mat[ci])
            samples_cna.append(cna_mat[ci])
            samples_ic50.append(ic50_log)

    n = len(samples_ic50)
    print(f"  Triplets (drogue, lignée, IC50) valides : {n:,}")
    if n == 0:
        return None, None, 0

    atoms_arr = np.stack(samples_atoms).astype(np.float32)
    adj_arr   = np.stack(samples_adj).astype(np.float32)
    gex_arr   = np.stack(samples_gex).astype(np.float32)
    mut_arr   = np.stack(samples_mut).astype(np.float32)
    cna_arr   = np.stack(samples_cna).astype(np.float32)
    ic50_arr  = np.array(samples_ic50, dtype=np.float32)

    # Normaliser IC50
    ic50_mean, ic50_std = ic50_arr.mean(), ic50_arr.std() + 1e-6
    ic50_arr = (ic50_arr - ic50_mean) / ic50_std

    # Shuffle & split
    if split_mode == 'leave_drug_out':
        sorted_drugs = sorted(set(drug_ids))
        n_val_drugs  = max(1, int(len(sorted_drugs) * val_split))
        train_drug_set = set(sorted_drugs[:-n_val_drugs])
        val_drug_set   = set(sorted_drugs[-n_val_drugs:])
        # Map each sample to its drug_id
        # samples_atoms was built iterating drug_ids × common_cells in order
        # Rebuild the sample→drug_id array from the same iteration order
        sample_drug_ids = []
        for drug_id in drug_ids:
            for ci, cell in enumerate(common_cells):
                ic50_val = ic50_df.loc[drug_id, cell] if cell in ic50_df.columns else np.nan
                if not np.isnan(ic50_val):
                    sample_drug_ids.append(drug_id)
        sample_drug_ids = np.array(sample_drug_ids)
        tr = np.where(np.isin(sample_drug_ids, list(train_drug_set)))[0]
        va = np.where(np.isin(sample_drug_ids, list(val_drug_set)))[0]
        print(f"  Leave-drug-out split : {len(train_drug_set)} train drugs | {len(val_drug_set)} val drugs")

    elif split_mode == 'leave_cell_out':
        sorted_cells = sorted(common_cells)
        n_val_cells  = max(1, int(len(sorted_cells) * val_split))
        train_cell_set = set(sorted_cells[:-n_val_cells])
        val_cell_set   = set(sorted_cells[-n_val_cells:])
        sample_cells = []
        for drug_id in drug_ids:
            for ci, cell in enumerate(common_cells):
                ic50_val = ic50_df.loc[drug_id, cell] if cell in ic50_df.columns else np.nan
                if not np.isnan(ic50_val):
                    sample_cells.append(cell)
        sample_cells = np.array(sample_cells)
        tr = np.where(np.isin(sample_cells, list(train_cell_set)))[0]
        va = np.where(np.isin(sample_cells, list(val_cell_set)))[0]
        print(f"  Leave-cell-out split : {len(train_cell_set)} train cells | {len(val_cell_set)} val cells")

    else:  # 'random'
        rng = np.random.default_rng(random_seed)
        idx = rng.permutation(n)
        split = int((1 - val_split) * n)
        tr, va = idx[:split], idx[split:]

    def make_real_ds(indices):
        ds = tf.data.Dataset.from_tensor_slices((
            atoms_arr[indices], adj_arr[indices],
            gex_arr[indices],   mut_arr[indices],
            cna_arr[indices],   ic50_arr[indices],
        ))
        return ds.shuffle(min(len(indices), 5000), seed=random_seed) \
                 .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_real_ds(tr)
    val_ds   = make_real_ds(va)
    print(f"  Train : {len(tr):,} | Val : {len(va):,}")
    return train_ds, val_ds, n


# ─── 10. DIGITAL TWIN INFERENCE API ──────────────────────────────────────────

class DigitalTwinInference:
    """
    High-level API for deploying the trained Digital Twin:
      - screen_drug_library  : batch IC50 prediction over a compound library
      - sensitivity_profile  : predict IC50 for one cell line vs all drugs
      - virtual_perturbation : in-silico gene KO effect on IC50
    """
    def __init__(self, model: BiIntDigitalTwin, featurizer: BRICSMolecularFeaturizer):
        self.model      = model
        self.featurizer = featurizer

    def predict_ic50(self, smiles: str, gex, mut, cnv) -> float:
        from smiles_sanitizer import sanitize_smiles
        clean_smi = sanitize_smiles(smiles)
        if clean_smi is None:
            return 10.0  # High IC50 for invalid SMILES
        
        atom_feat = self.featurizer.featurize(clean_smi)[np.newaxis]
        adj = np.ones((1, HP['max_atoms'], HP['max_atoms']), dtype=np.float32)
        gex_t = tf.constant(gex[np.newaxis], dtype=tf.float32)
        mut_t = tf.constant(mut[np.newaxis], dtype=tf.float32)
        cnv_t = tf.constant(cnv[np.newaxis], dtype=tf.float32)
        ic50, _ = self.model(
            (tf.constant(atom_feat), tf.constant(adj), gex_t, mut_t, cnv_t),
            training=False
        )
        return float(ic50[0].numpy())

    def screen_drug_library(self, smiles_list: list, gex, mut, cnv) -> dict:
        """Returns {smiles: ic50} dict for full compound library."""
        results = {}
        for smiles in smiles_list:
            results[smiles] = self.predict_ic50(smiles, gex, mut, cnv)
        return dict(sorted(results.items(), key=lambda x: x[1]))

    def virtual_gene_ko(self, smiles: str, gex, mut, cnv,
                         gene_indices: list) -> dict:
        """
        In-silico gene knockout: sets GEx to 0 for given gene indices.
        Returns baseline vs perturbed IC50 delta.
        """
        baseline = self.predict_ic50(smiles, gex, mut, cnv)
        perturbed_gex = gex.copy()
        perturbed_gex[gene_indices] = 0.0
        perturbed = self.predict_ic50(smiles, perturbed_gex, mut, cnv)
        return {
            'baseline_ic50'  : baseline,
            'perturbed_ic50' : perturbed,
            'delta_ic50'     : perturbed - baseline,
            'sensitivity_shift': 'Resistant' if perturbed > baseline else 'Sensitized'
        }


# ─── 11. MAIN — BUILD, TRAIN, RL OPTIMIZE ────────────────────────────────────

PRETRAIN_SMILES = [
    # Simple molecules for basic learning
    "CC", "CCC", "CCCC", "CCO", "CCN", "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1",
    "c1ccccc1", "C1CCNCC1", "C1CCOCC1",
    # Médicaments connus (drug-like, avec cycles)
    "CC(=O)Oc1ccccc1C(=O)O",           # Aspirine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testostérone
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caféine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # Ibuprofène
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", # Pyrène
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", # Imatinib-like
    "CC1=CC2=C(C=C1)N(C3=CC=CC=C23)CC(=O)N4CCOCC4",
    "O=C(O)c1ccccc1O",                 # Acide salicylique
    "Nc1ccc(cc1)S(=O)(=O)N",          # Sulfanilamide
    "CC(=O)Nc1ccc(O)cc1",             # Paracétamol
    "c1ccc(cc1)CN2CCNCC2",
    "Cc1ncc(COP(=O)(O)O)c(CN)c1O",    # Pyridoxine-like
    "OC(=O)c1ccc(N)cc1",
    "CC1CCCCC1NC(=O)c1cccc(c1)C(F)(F)F",
    "CN(C)CCCN1c2ccccc2CCc2ccccc21",   # Imipramine-like
    "COc1ccc(CCN)cc1O",
    "Clc1ccc(cc1)C(c1ccccc1)(c1ccccc1)O",
    "O=C(Nc1ccccc1)c1cccnc1",
    "CC(N)Cc1ccc(O)cc1",               # Tyramine
    "c1ccc2[nH]ccc2c1",                # Indole
    "C1CCN(CC1)c1ncnc2[nH]ccc12",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",  # Adénosine
    "CC(=O)NCC1CN(c2nc(N)nc3c(=O)[nH]cc(c23))C(=O)O1",
    "OC[C@H]1OC(n2cnc3c(N)ncnc23)[C@H](O)[C@@H]1O",
    # Additional common drugs and molecules
    "CC(C)CC1=CC=C(C=C1)O",           # Thymol
    "CC1=C(C(=O)NC2=CC=CC=C12)C3=CC=CC=C3",  # Indomethacin
    "CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31",  # Amitriptyline
    "CC(C)(C)NCC(O)C1=CC=C(O)C=C1",   # Salbutamol
    "CC1=CC(=O)NN=C1C",               # Acetylacetone
    "C1=CC=C(C=C1)C(=O)O",            # Benzoic acid
    "C1=CC=C(C=C1)CCO",               # Phenethyl alcohol
    "CC1=CC=C(C=C1)C(=O)C",           # Acetophenone
    "C1=CC=C(C=C1)CN",                # Benzylamine
    "CC1=CC=C(C=C1)S(=O)(=O)N",       # Benzenesulfonamide
    "C1=CC=C(C=C1)OC",                # Anisole
    "CC1=CC=C(C=C1)OC",               # p-Methylanisole
    "C1=CC=C(C=C1)Br",                # Bromobenzene
    "C1=CC=C(C=C1)I",                 # Iodobenzene
    "C1=CC=C(C=C1)F",                 # Fluorobenzene
    "C1=CC=C(C=C1)Cl",                # Chlorobenzene
    "C1=CC=C(C=C1)N",                 # Aniline
    "C1=CC=C(C=C1)NO2",               # Nitrobenzene
    "C1=CC=C(C=C1)C#N",               # Benzonitrile
    "C1=CC=C(C=C1)C=O",               # Benzaldehyde
    "C1=CC=C(C=C1)CC",                # Ethylbenzene
    "CC1=CC=CC=C1",                   # Toluene
    "C1CCCCC1",                       # Cyclohexane
    "C1CCNCC1",                       # Piperidine
    "C1CCOCC1",                       # Tetrahydrofuran
    "C1CCN(CC1)C",                    # N-Methylpiperidine
    "C1=CC=NC=C1",                    # Pyridine
    "C1=CC=NN=C1",                    # Pyridazine
    "C1=CN=CC=N1",                    # Pyrimidine
    "C1=NC=NC=N1",                    # 1,3,5-Triazine
    "C1=CC=C2C=CC=CC2=C1",            # Naphthalene
    "C1=CC=C2C=C3C=CC=CC3=CC2=C1",     # Anthracene
    "C1=CC=C2C(=C1)C=CC=C2",          # Indene
    "C1=CC=C2C(=C1)NC=C2",            # Indole
    "C1=CC=C2C(=C1)C=CN2",            # Quinoline
    "C1=CC=C2C(=C1)N=CC=C2",          # Isoquinoline
    "C1=CC=C2C(=C1)C=CC=N2",          # Quinazoline
    "C1=CC=C2C(=C1)C=NC=C2",          # Quinoxaline
    "C1=CC=C2C(=C1)C=CC=C2",          # Benzene (already have)
    "CC(=O)NC1=CC=CC=C1",             # Acetanilide
    "CC(=O)OC1=CC=CC=C1",             # Phenyl acetate
    "C1=CC=C(C=C1)C(=O)NC",           # N-Methylbenzamide
    "C1=CC=C(C=C1)C(=O)N",            # Benzamide
    "C1=CC=C(C=C1)CON",               # Benzohydroxamic acid
    "C1=CC=C(C=C1)C(=O)Cl",           # Benzoyl chloride
    "C1=CC=C(C=C1)C(=O)F",            # Benzoyl fluoride
    "C1=CC=C(C=C1)C(=O)Br",           # Benzoyl bromide
    "C1=CC=C(C=C1)C(=O)I",            # Benzoyl iodide
    "C1=CC=C(C=C1)C(=O)CC",           # 1-Phenylbutan-1-one
    "C1=CC=C(C=C1)C(=O)C1=CC=CC=C1",  # Benzophenone
    "C1=CC=C(C=C1)C(=O)C(=O)C1=CC=CC=C1",  # Benzil
    "C1=CC=C(C=C1)C(=O)C(=O)O",       # Phenylglyoxylic acid
    "C1=CC=C(C=C1)C(=O)C(=O)NC",      # N-Methylphenylglyoxamide
    "C1=CC=C(C=C1)C(=O)C(=O)N",       # Phenylglyoxamide
    "C1=CC=C(C=C1)C(=O)C(=O)Cl",      # Phenylglyoxylyl chloride
    "C1=CC=C(C=C1)C(=O)C(=O)F",       # Phenylglyoxylyl fluoride
    "C1=CC=C(C=C1)C(=O)C(=O)Br",      # Phenylglyoxylyl bromide
    "C1=CC=C(C=C1)C(=O)C(=O)I",       # Phenylglyoxylyl iodide
    "C1=CC=C(C=C1)C(=O)C(=O)CC",      # 1-Phenyl-2-oxobutan-1-one
    "C1=CC=C(C=C1)C(=O)C(=O)C1=CC=CC=C1",  # Benzoylformic acid phenyl ester
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C1=CC=CC=C1",  # Benzoylformic acid benzoyl ester
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)O",  # Benzoylformic acid
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)NC", # N-Methylbenzoylformamide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)N",  # Benzoylformamide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)Cl", # Benzoylformyl chloride
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)F",  # Benzoylformyl fluoride
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)Br", # Benzoylformyl bromide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)I",  # Benzoylformyl iodide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)CC", # 1-Phenyl-2,3-dioxobutan-1-one
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C1=CC=CC=C1",  # Benzoylformic acid benzoyl ester
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)C1=CC=CC=C1",  # Benzoylformic acid dibenzoyl ester
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)O",  # Benzoylformic acid
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)NC", # N-Methylbenzoylformamide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)N",  # Benzoylformamide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)Cl", # Benzoylformyl chloride
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)F",  # Benzoylformyl fluoride
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)Br", # Benzoylformyl bromide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)I",  # Benzoylformyl iodide
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)CC", # 1-Phenyl-2,3,4-trioxopentan-1-one
    "C1=CC=C(C=C1)C(=O)C(=O)C(=O)C(=O)C1=CC=CC=C1",  # Benzoylformic acid benzoyl ester
]

def main():
    print("=" * 72)
    print("  Bi-Int Digital Twin  |  Cell Line Drug Screening  |  IC50 Prediction")
    print("=" * 72)

    # ── Build model
    vocab      = SMILESVocabulary()
    HP['vocab_size'] = vocab.vocab_size  # Update HP with actual vocab size
    model      = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()

    # ── Load pre-trained drug encoder weights from ChEMBL
    if os.path.exists('pretrained_weights/chembl_drug_encoder.weights.h5'):
        print("\n[Pre-trained] Loading ChEMBL pre-trained weights for drug encoder...")
        # Load weights directly from HDF5
        import h5py
        with h5py.File('pretrained_weights/chembl_drug_encoder.weights.h5', 'r') as f:
            # Transfer weights to matching layers
            for layer_name in ['node_embed', 'gcn_proj_1', 'ln1', 'node_proj', 'ln2']:
                if layer_name in [l.name for l in model.drug_gnn.layers]:
                    if layer_name in f:
                        weights = [np.array(f[layer_name][w]) for w in f[layer_name]]
                        model.drug_gnn.get_layer(layer_name).set_weights(weights)
                        print(f"  Loaded weights for layer: {layer_name}")
        print("[Pre-trained] Drug encoder initialized with ChEMBL pre-training.")
    else:
        print("\n[Pre-trained] No pre-trained weights found, using random initialization.")

    # ── Tokenizer Persistency Demo
    vocab.save("smiles_tokenizer.json")
    print("\n[Tokenizer] Saved parallel SMILES tokenizer to smiles_tokenizer.json")
    vocab = SMILESVocabulary.load("smiles_tokenizer.json")
    print("[Tokenizer] Successfully loaded parallel SMILES tokenizer.")

    # Warm-up (build graph)
    dummy_batch = generate_synthetic_ccle_batch(batch_size=2)
    ic50_out, kl_out = model(dummy_batch[:-1], training=False)
    print(f"\n[Model Built] IC50 output shape: {ic50_out.shape} | KL: {kl_out:.4f}")
    print(f"  Trainable parameters: {model.count_params():,}")

    # ── Datasets : vraies données CCLE, sinon synthétiques
    train_ds, val_ds, n_real = load_ccle_real_data(
        ccle_dir='Dataset/ccle_broad_2019',
        batch_size=HP['batch_size'],
    )
    if train_ds is None:
        print("\n[Data] CCLE non disponible — données synthétiques utilisées.")
        train_ds = make_tf_dataset(n_samples=256, batch_size=HP['batch_size'])
        val_ds   = make_tf_dataset(n_samples=64,  batch_size=HP['batch_size'])
    else:
        print(f"\n[Data] Données CCLE réelles chargées ({n_real:,} triplets).")

    # ── Train
    print("\n[Training] Bi-Int Digital Twin on IC50 prediction (QSAR sur CCLE)...")
    trainer = BiIntTrainer(model, HP)
    history = trainer.fit(train_ds, val_ds, epochs=20)

    # ── RL Drug Generation
    print("\n[RL] Initializing PPO Drug Generator...")
    policy = DrugGeneratorPolicy(vocab_size=vocab.vocab_size)

    # Dummy condition: get omics latent z from a cell line
    dummy_gex = tf.random.normal([16, HP['gex_dim']])
    dummy_mut = tf.random.uniform([16, HP['mut_dim']], 0, 2, dtype=tf.float32)
    dummy_cnv = tf.random.normal([16, HP['cnv_dim']])
    z_sample, _, _ = model.omics_vae((dummy_gex, dummy_mut, dummy_cnv), training=False)

    ppo = PPODrugGenerator(policy, model, vocab, featurizer, HP)
    
    # Pre-train on valid SMILES (loaded from file or defaults)
    valid_smiles_examples = PRETRAIN_SMILES
    print(f"\n[PreTrain] Loaded {len(valid_smiles_examples)} valid SMILES for pre-training")
    ppo.pretrain_on_valid_smiles(valid_smiles_examples, z_sample[:len(valid_smiles_examples)], epochs=100)
    
    final_stats = ppo.optimize(z_sample, dummy_gex, dummy_mut, dummy_cnv, episodes=200)

    # ── Digital Twin Inference Demo
    print("\n[Inference] Digital Twin virtual screening demo...")
    inference = DigitalTwinInference(model, featurizer)
    test_smiles = [
        "CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1",  # Top GraphGA candidate QED=0.872 SA=0.794
        "COC(=O)OCC(=O)OCC(=O)Nc1ccccc1N(C)C",  # QED=0.784 SA=0.873
        "COC(=O)OCC(=O)OCC(=O)Nc1ccccc1C(=O)O",  # QED=0.733 SA=0.891
        "O=C(COC(=O)COC(=O)OC1CC1)Nc1ccccc1C(=O)O",  # QED=0.710 SA=0.876
        "CC(=O)Nc1ccccc1-c1ccccc1COC=O",  # QED=0.849 SA=0.877
    ]
    gex_demo = np.random.randn(HP['gex_dim']).astype(np.float32)
    mut_demo = np.random.randint(0, 2, HP['mut_dim']).astype(np.float32)
    cnv_demo = np.random.randn(HP['cnv_dim']).astype(np.float32)

    results = inference.screen_drug_library(test_smiles, gex_demo, mut_demo, cnv_demo)
    print("\n  Virtual Drug Screen Results (sorted by IC50):")
    for smiles, ic50 in results.items():
        print(f"    {smiles[:45]:45s} → IC50: {ic50:+.3f} log µM")

    ko_result = inference.virtual_gene_ko(
        test_smiles[0], gex_demo, mut_demo, cnv_demo,
        gene_indices=[0, 1, 5, 42]
    )
    print(f"\n  Virtual Gene KO Result: {ko_result}")

    print("\n[Done] Pipeline complete.")
    return model, history, final_stats


def load_pretrained_drug_encoder(model, weight_path='pretrained_weights/chembl_drug_encoder.weights.h5'):
    if os.path.exists(weight_path):
        print("\n[Pre-trained] Loading ChEMBL pre-trained weights for drug encoder...")
        import h5py
        with h5py.File(weight_path, 'r') as f:
            for layer_name in ['node_embed', 'gcn_proj_1', 'ln1', 'node_proj', 'ln2']:
                if layer_name in [l.name for l in model.drug_gnn.layers] and layer_name in f:
                    weights = [np.array(f[layer_name][w]) for w in f[layer_name]]
                    model.drug_gnn.get_layer(layer_name).set_weights(weights)
                    print(f"  Loaded weights for layer: {layer_name}")
        print("[Pre-trained] Drug encoder initialized with ChEMBL pre-training.")
        return True
    print("\n[Pre-trained] No pre-trained weights found, using random initialization.")
    return False


def run_pipeline(use_pretrained=True, epochs=20, run_ppo=True, rl_episodes=None,
                 loss_mode='kl'):
    print("=" * 72)
    print("  Bi-Int Digital Twin  |  Cell Line Drug Screening  |  IC50 Prediction")
    print("=" * 72)

    vocab      = SMILESVocabulary()
    HP['vocab_size'] = vocab.vocab_size
    model      = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()

    if use_pretrained:
        load_pretrained_drug_encoder(model)
    else:
        print("\n[Baseline] Running without ChEMBL pre-training.")

    vocab.save("smiles_tokenizer.json")
    print("\n[Tokenizer] Saved parallel SMILES tokenizer to smiles_tokenizer.json")
    vocab = SMILESVocabulary.load("smiles_tokenizer.json")
    print("[Tokenizer] Successfully loaded parallel SMILES tokenizer.")

    dummy_batch = generate_synthetic_ccle_batch(batch_size=2)
    ic50_out, kl_out = model(dummy_batch[:-1], training=False)
    print(f"\n[Model Built] IC50 output shape: {ic50_out.shape} | KL: {kl_out:.4f}")
    print(f"  Trainable parameters: {model.count_params():,}")

    # ── Données : vraies données CCLE en priorité, sinon synthétiques
    train_ds, val_ds, n_real = load_ccle_real_data(
        ccle_dir='Dataset/ccle_broad_2019',
        batch_size=HP['batch_size'],
    )
    if train_ds is None:
        print("\n[Data] CCLE non disponible — données synthétiques utilisées.")
        train_ds = make_tf_dataset(n_samples=256, batch_size=HP['batch_size'])
        val_ds   = make_tf_dataset(n_samples=64,  batch_size=HP['batch_size'])
    else:
        print(f"\n[Data] Données CCLE réelles chargées ({n_real:,} triplets).")

    print("\n[Training] Bi-Int Digital Twin on IC50 prediction (QSAR)...")
    trainer = BiIntTrainer(model, HP, loss_mode=loss_mode)
    history = trainer.fit(train_ds, val_ds, epochs=epochs)

    final_stats = None
    if run_ppo:
        if rl_episodes is None:
            rl_episodes = HP['rl_episodes']
        print("\n[RL] Initializing PPO Drug Generator...")
        policy = DrugGeneratorPolicy(vocab_size=vocab.vocab_size)

        dummy_gex = tf.random.normal([16, HP['gex_dim']])
        dummy_mut = tf.random.uniform([16, HP['mut_dim']], 0, 2, dtype=tf.float32)
        dummy_cnv = tf.random.normal([16, HP['cnv_dim']])
        z_sample, _, _ = model.omics_vae((dummy_gex, dummy_mut, dummy_cnv), training=False)

        ppo = PPODrugGenerator(policy, model, vocab, featurizer, HP)
        valid_smiles_examples = PRETRAIN_SMILES
        print(f"\n[PreTrain] Loaded {len(valid_smiles_examples)} valid SMILES for pre-training")
        ppo.pretrain_on_valid_smiles(valid_smiles_examples, z_sample[:len(valid_smiles_examples)], epochs=100)
        final_stats = ppo.optimize(z_sample, dummy_gex, dummy_mut, dummy_cnv, episodes=rl_episodes)

        # Save RL-generated SMILES for GraphGA initialization.
        rl_smiles = []
        max_rl_smiles = 40
        for _ in range(5):
            token_ids = ppo.policy.generate(z_sample, temperature=0.85)
            smiles_batch = ppo.vocab.batch_decode(token_ids.numpy())
            for smi in smiles_batch:
                smi = smi.strip()
                if smi and smi not in rl_smiles:
                    rl_smiles.append(smi)
                if len(rl_smiles) >= max_rl_smiles:
                    break
            if len(rl_smiles) >= max_rl_smiles:
                break

        if rl_smiles:
            with open("rl_generated_smiles.txt", "w") as f:
                for smi in rl_smiles:
                    f.write(f"{smi}\n")
            with open("smiles_data.txt", "w") as f:
                for smi in rl_smiles:
                    f.write(f"{smi}\n")
            print(f"\n[RL] Saved {len(rl_smiles)} RL-generated SMILES to rl_generated_smiles.txt and smiles_data.txt")

        print("\n[Inference] Digital Twin virtual screening demo...")
        inference = DigitalTwinInference(model, featurizer)
        test_smiles = [
            "CN1CCCN(C2CC2)CC(c2ccccc2NCCO)C1",
            "COC(=O)OCC(=O)OCC(=O)Nc1ccccc1N(C)C",
            "COC(=O)OCC(=O)OCC(=O)Nc1ccccc1C(=O)O",
            "O=C(COC(=O)COC(=O)OC1CC1)Nc1ccccc1C(=O)O",
            "CC(=O)Nc1ccccc1-c1ccccc1COC=O",
        ]
        gex_demo = np.random.randn(HP['gex_dim']).astype(np.float32)
        mut_demo = np.random.randint(0, 2, HP['mut_dim']).astype(np.float32)
        cnv_demo = np.random.randn(HP['cnv_dim']).astype(np.float32)

        results = inference.screen_drug_library(test_smiles, gex_demo, mut_demo, cnv_demo)
        print("\n  Virtual Drug Screen Results (sorted by IC50):")
        for smiles, ic50 in results.items():
            print(f"    {smiles[:45]:45s} → IC50: {ic50:+.3f} log µM")

        ko_result = inference.virtual_gene_ko(
            test_smiles[0], gex_demo, mut_demo, cnv_demo,
            gene_indices=[0, 1, 5, 42]
        )
        print(f"\n  Virtual Gene KO Result: {ko_result}")

    print("\n[Done] Pipeline complete.")
    return model, history, final_stats


def compare_pretraining(epochs=20):
    print("\n=== Comparison: Baseline vs ChEMBL Pre-trained Drug Encoder ===\n")
    print("[1/2] Running baseline model without pre-training...")
    _, baseline_history, _ = run_pipeline(use_pretrained=False, epochs=epochs, run_ppo=False)
    print("\n[2/2] Running model with ChEMBL pre-training...")
    _, pretrained_history, _ = run_pipeline(use_pretrained=True, epochs=epochs, run_ppo=False)

    baseline_val = baseline_history['val'][-1]
    pretrained_val = pretrained_history['val'][-1]
    baseline_rmse = baseline_val
    pretrained_rmse = pretrained_val

    print("\n=== Comparison Results ===")
    print(f"Baseline final Val RMSE   : {baseline_rmse:.4f} (val_loss={baseline_val:.4f})")
    print(f"Pre-trained final Val RMSE: {pretrained_rmse:.4f} (val_loss={pretrained_val:.4f})")
    delta = baseline_rmse - pretrained_rmse
    print(f"Delta (baseline - pre-trained) : {delta:+.4f}")
    if delta > 0:
        print("[INFO] Pre-training improved validation RMSE.")
    elif delta < 0:
        print("[WARN] Pre-training did not improve validation RMSE in this run.")
    else:
        print("[INFO] No change detected between baseline and pre-trained runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bi-Int full pipeline with optional ChEMBL pre-training comparison.')
    parser.add_argument('--mode', choices=['pretrained', 'baseline', 'compare'], default='pretrained', help='Pipeline mode to execute')
    parser.add_argument('--epochs', type=int, default=20, help='Number of IC50 training epochs')
    parser.add_argument('--rl-episodes', type=int, default=HP['rl_episodes'], help='Number of PPO episodes for RL drug generation')
    parser.add_argument('--no-ppo', action='store_true', help='Skip PPO/RL drug generation, only train IC50 model')
    parser.add_argument('--loss-mode', choices=['kl', 'cross_entropy', 'both'],
                        default='kl',
                        help='VAE loss mode: kl (original) | cross_entropy | both (KL+CE)')
    args = parser.parse_args()

    if args.mode == 'baseline':
        run_pipeline(use_pretrained=False, epochs=args.epochs, run_ppo=False,
                     loss_mode=args.loss_mode)
    elif args.mode == 'compare':
        compare_pretraining(epochs=args.epochs)
    else:
        run_pipeline(use_pretrained=True, epochs=args.epochs, rl_episodes=args.rl_episodes,
                     run_ppo=not args.no_ppo, loss_mode=args.loss_mode)