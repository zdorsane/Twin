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
import os, math, json, warnings
import concurrent.futures
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
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
    vae_beta        = 1.0,

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

    def call(self, inputs, training=False):
        gex, mut, cnv = inputs
        mu, log_var   = self.encode(gex, mut, cnv, training)
        z             = self.reparameterize(mu, log_var)
        recon         = self.decode(z, training)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
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
    Multi-layer GAT encoder on the atom-level feature matrix.
    Global attentive pooling → fixed-size Drug Embedding D.
    """
    def __init__(self, out_dim=HP['drug_node_dim'], **kwargs):
        super().__init__(**kwargs)
        self.input_proj = layers.Dense(128, activation='gelu')
        self.gat_layers = [GATLayer(128) for _ in range(HP['gnn_layers'])]
        self.pool_attn  = layers.Dense(1)       # attentive pooling
        self.out_proj   = layers.Dense(out_dim)

    def call(self, atom_feat, adj_mask=None, training=False):
        """
        atom_feat : [B, MAX_ATOMS, atom_feat_dim]
        Returns   : [B, MAX_ATOMS, out_dim]  — node-level drug embeddings D
        """
        x = self.input_proj(atom_feat)
        for gat in self.gat_layers:
            x = gat(x, adj_mask, training=training)
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

    def call(self, inputs, training=False):
        drug_atoms, adj_mask, gex, mut, cnv = inputs

        # ── Drug: GNN → node embeddings [B, N_atoms, drug_node_dim]
        D_nodes = self.drug_gnn(drug_atoms, adj_mask, training=training)
        D = self.drug_proj(D_nodes)   # [B, N_atoms, dm]

        # ── Omics: VAE → latent [B, latent_dim]
        z, _, kl_loss = self.omics_vae((gex, mut, cnv), training=training)
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
    def __init__(self, model: BiIntDigitalTwin, hp=HP):
        self.model = model
        self.hp    = hp
        self.opt   = keras.optimizers.AdamW(hp['learning_rate'])
        self.mse   = keras.losses.MeanSquaredError()

    @tf.function
    def train_step(self, batch):
        drug_atoms, adj_mask, gex, mut, cnv, ic50_true = batch
        with tf.GradientTape() as tape:
            ic50_pred, kl_loss = self.model(
                (drug_atoms, adj_mask, gex, mut, cnv), training=True)
            regression_loss = self.mse(ic50_true, ic50_pred)
            beta = self.hp['vae_beta']
            total_loss = regression_loss + beta * kl_loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        rmse = tf.sqrt(regression_loss)
        return {'total_loss': total_loss, 'regression_loss': regression_loss,
                'kl_loss': kl_loss, 'rmse': rmse}

    @tf.function
    def val_step(self, batch):
        drug_atoms, adj_mask, gex, mut, cnv, ic50_true = batch
        ic50_pred, kl_loss = self.model(
            (drug_atoms, adj_mask, gex, mut, cnv), training=False)
        regression_loss = self.mse(ic50_true, ic50_pred)
        rmse = tf.sqrt(regression_loss)
        return {'val_loss': regression_loss + self.hp['vae_beta']*kl_loss,
                'val_rmse': rmse}

    def fit(self, train_ds, val_ds, epochs=50):
        history = {'train': [], 'val': []}
        for epoch in range(1, epochs + 1):
            train_metrics = {}
            for batch in train_ds:
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    train_metrics.setdefault(k, []).append(v.numpy())

            val_metrics = {}
            for batch in val_ds:
                metrics = self.val_step(batch)
                for k, v in metrics.items():
                    val_metrics.setdefault(k, []).append(v.numpy())

            t_rmse = np.mean(train_metrics['rmse'])
            v_rmse = np.mean(val_metrics['val_rmse'])
            history['train'].append(t_rmse)
            history['val'].append(v_rmse)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train RMSE: {t_rmse:.4f} | "
                      f"Val RMSE: {v_rmse:.4f} | KL: {np.mean(train_metrics['kl_loss']):.4f}")
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
        chars = []
        for i in idxs:
            if i == 1: break  # EOS
            if i > 1: chars.append(self.idx2char.get(i, '?'))
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
        """Autoregressive SMILES sampling with annealed temperature."""
        # Annealed temperature: high at start (exploration), low at end (exploitation)
        annealed_temp = temperature * (0.1 + 0.9 * (1 - step / max(max(total_steps, 1), 1)))
        B = tf.shape(z)[0]
        token = tf.fill([B, 1], 2)   # start with 'C'
        generated = [token]
        states = None
        
        for i in range(max_len):
            logits, _, states = self(token, z, states, training=False, conditional=True)
            logits = logits[:, -1, :] / annealed_temp
            if i == 0:
                logits = tf.concat([tf.fill([B, 2], -1e9), logits[:, 2:]], axis=-1)
            else:
                logits = tf.concat([tf.fill([B, 1], -1e9), logits[:, 1:]], axis=-1)
            token = tf.random.categorical(logits, 1, dtype=tf.int32)
            generated.append(token)
            if tf.reduce_all(token == 1):   # all EOS
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
        self.entropy_coef  = 0.01
        self.gae_lambda   = 0.95
        self.gamma        = hp['rl_gamma']

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

    def compute_reward(self, smiles_batch: list, z_batch, gex, mut, cnv) -> np.ndarray:
        """
        Reward = +1 for valid SMILES, -1 for invalid.
        Focus on validity first.
        """
        rewards = []
        for smiles in smiles_batch:
            if HAS_RDKIT:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    rewards.append(1.0)  # Positive reward for valid SMILES
                else:
                    rewards.append(-1.0)  # Negative for invalid
            else:
                # Fallback: assume valid if no RDKit
                rewards.append(1.0)
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

            total_loss  = policy_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy.trainable_variables))
        return total_loss.numpy(), entropy.numpy()

    def train_episode(self, z, gex, mut, cnv, n_samples=64, episode=1, total_episodes=100):
        """One PPO episode: sample SMILES → compute rewards → update policy."""
        # Sample with annealed temperature
        token_ids   = self.policy.generate(z[:n_samples], step=episode, total_steps=total_episodes)    # [n, max_len]
        smiles_list = self.vocab.batch_decode(token_ids.numpy()) # Using parallel decode

        # Compute old log probs
        logits, values, _ = self.policy(token_ids, z[:n_samples], training=False, conditional=True)
        dist = tfd.Categorical(logits=logits)
        old_log_probs = tf.reduce_mean(dist.log_prob(token_ids), axis=1)

        # Rewards
        rewards = self.compute_reward(smiles_list, z[:n_samples], gex, mut, cnv)
        rewards_t = tf.constant(rewards)

        # GAE advantages
        vals_mean  = tf.reduce_mean(values, axis=1).numpy()
        advantages = rewards - vals_mean
        returns    = rewards

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = tf.constant(advantages, dtype=tf.float32)
        returns_t  = tf.constant(returns, dtype=tf.float32)

        # PPO update
        loss, entropy = self._ppo_update(token_ids, old_log_probs,
                                          advantages, returns_t, z[:n_samples])

        best_idx = np.argmax(rewards)
        return {
            'loss'       : loss,
            'entropy'    : entropy,
            'mean_reward': rewards.mean(),
            'best_smiles': smiles_list[best_idx],
            'best_reward': rewards[best_idx],
        }

    def optimize(self, z, gex, mut, cnv, episodes=HP['rl_episodes']):
        print("\n[PPO Drug Generator] Starting optimization...")
        for ep in range(1, episodes + 1):
            stats = self.train_episode(z, gex, mut, cnv, episode=ep, total_episodes=episodes)
            if ep % 10 == 0 or ep == 1:
                print(f"  Episode {ep:4d} | Reward: {stats['mean_reward']:+.3f} | "
                      f"Best: {stats['best_smiles'][:30]} | "
                      f"Entropy: {stats['entropy']:.3f}")
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
    In production: replace with actual CCLE/GDSC data loaders.
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
        atom_feat = self.featurizer.featurize(smiles)[np.newaxis]
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

def main():
    print("=" * 72)
    print("  Bi-Int Digital Twin  |  Cell Line Drug Screening  |  IC50 Prediction")
    print("=" * 72)

    # ── Build model
    model      = BiIntDigitalTwin(HP)
    featurizer = BRICSMolecularFeaturizer()
    vocab      = SMILESVocabulary()

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

    # ── Datasets
    print("\n[Data] Generating synthetic CCLE-style datasets...")
    train_ds = make_tf_dataset(n_samples=256, batch_size=HP['batch_size'])
    val_ds   = make_tf_dataset(n_samples=64,  batch_size=HP['batch_size'])

    # ── Train
    print("\n[Training] Bi-Int Digital Twin on IC50 prediction...")
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
    valid_smiles_examples = load_smiles_from_file("smiles_data.txt")
    print(f"\n[PreTrain] Loaded {len(valid_smiles_examples)} valid SMILES for pre-training")
    ppo.pretrain_on_valid_smiles(valid_smiles_examples, z_sample[:len(valid_smiles_examples)], epochs=100)
    
    final_stats = ppo.optimize(z_sample, dummy_gex, dummy_mut, dummy_cnv, episodes=200)

    # ── Digital Twin Inference Demo
    print("\n[Inference] Digital Twin virtual screening demo...")
    inference = DigitalTwinInference(model, featurizer)
    test_smiles = [
        "CC1=CC=C(C=C1)NC2=NC=CC(=N2)N3CCN(CC3)C4=CC=CC=C4",  # Imatinib-like
        "COC1=CC2=C(C=C1OC)NC(=O)C2=CC3=CC=CC=N3",              # Gefitinib-like
        "C1=CN=CC=C1",                                            # Pyridine
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


if __name__ == "__main__":
    main()