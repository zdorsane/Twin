"""
Transformer-based SMILES generator to replace LSTM
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class TransformerBlock(layers.Layer):
    """Transformer encoder/decoder block with multi-head attention"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class CrossAttentionBlock(layers.Layer):
    """Cross-attention between SMILES and omics conditioning"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, query, key_value, training):
        # Cross-attention: query from SMILES, key/value from conditioning
        attn_output = self.att(query, key_value)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerSMILESGenerator(keras.Model):
    """
    Transformer-based SMILES generator conditioned on omics latent z.
    Architecture:
      - Token embedding
      - Positional encoding
      - Transformer encoder blocks (self-attention on SMILES)
      - Cross-attention blocks (attend to z)
      - Output projection to vocabulary
    """
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, ff_dim=512, 
                 num_encoder_layers=4, max_len=80, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Embeddings
        self.token_embedding = layers.Embedding(vocab_size, embed_dim)
        self.positional_embedding = layers.Embedding(max_len, embed_dim)
        
        # Condition projection (z → embed_dim)
        self.cond_proj = layers.Dense(embed_dim)
        self.cond_cross_attn = CrossAttentionBlock(embed_dim, num_heads, ff_dim)
        
        # Transformer encoder blocks (self-attention on SMILES)
        self.encoder_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_encoder_layers)
        ]
        
        # Output heads
        self.logits_head = layers.Dense(vocab_size)
        self.value_head = layers.Dense(1)
        
    def call(self, token_ids, z, training=False, conditional=True):
        """
        token_ids: [B, T]
        z: [B, latent_dim]
        Returns: logits [B, T, vocab], value [B, T]
        """
        B = tf.shape(token_ids)[0]
        T = tf.shape(token_ids)[1]
        
        # Token embedding + positional encoding
        x = self.token_embedding(token_ids)  # [B, T, embed_dim]
        positions = tf.range(T)
        pos_embed = self.positional_embedding(positions)  # [T, embed_dim]
        x = x + tf.expand_dims(pos_embed, 0)  # Broadcast to [B, T, embed_dim]
        
        # Condition via cross-attention
        if conditional:
            z_proj = self.cond_proj(z)  # [B, embed_dim]
            z_expanded = tf.expand_dims(z_proj, 1)  # [B, 1, embed_dim]
            x = self.cond_cross_attn(x, z_expanded, training=training)  # [B, T, embed_dim]
        
        # Transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x, training=training)
        
        # Output
        logits = self.logits_head(x)  # [B, T, vocab_size]
        value = tf.squeeze(self.value_head(x), -1)  # [B, T]
        
        return logits, value
    
    def generate(self, z, max_len=80, temperature=1.0, step=0, total_steps=100):
        """Autoregressive generation with Transformer"""
        annealed_temp = temperature * (0.1 + 0.9 * (1 - step / max(total_steps, 1)))
        B = tf.shape(z)[0]
        
        token = tf.fill([B, 1], 2)  # Start with 'C'
        generated = [token]
        
        for _ in range(max_len - 1):
            logits, _ = self(tf.concat(generated, axis=1), z, training=False, conditional=True)
            next_logits = logits[:, -1, :] / annealed_temp
            next_token = tf.random.categorical(next_logits, 1, dtype=tf.int32)
            generated.append(next_token)
            
            if tf.reduce_all(next_token == 1):  # EOS token
                break
        
        return tf.concat(generated, axis=1)


# Test
if __name__ == "__main__":
    generator = TransformerSMILESGenerator(vocab_size=60, embed_dim=256, num_heads=8)
    
    # Dummy inputs
    token_ids = tf.random.uniform([4, 40], maxval=60, dtype=tf.int32)
    z = tf.random.normal([4, 128])
    
    logits, value = generator(token_ids, z, training=True, conditional=True)
    print(f"Logits shape: {logits.shape}")  # [4, 40, 60]
    print(f"Value shape: {value.shape}")    # [4, 40]
    
    # Generation
    generated = generator.generate(z, max_len=40)
    print(f"Generated shape: {generated.shape}")  # [4, 40]
    print("Transformer SMILES Generator ✅")
