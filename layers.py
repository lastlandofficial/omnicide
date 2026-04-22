"""
================================================================================
 TRANSFORMER LAYERS — Differentiable Building Blocks
================================================================================
Positional Encoding, Feed-Forward Networks, Encoder Blocks, Decoder Blocks,
and the new Decoder-Only Block (for GPT-style pretraining).

All operations use the Tensor autograd engine, so gradients flow automatically.
================================================================================
"""
import numpy as np
from math_ops import Linear, LayerNorm
from attention import MultiHeadAttention
from tensor import Tensor, tensor_relu


class PositionwiseFeedForward:
    """
    Position-wise Feed-Forward Network: FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂

    Applied independently to each position (token) in the sequence.
    This is where the model does its "thinking" — the attention layer
    gathers information, and the FFN processes it.

    Architecture:
        d_model → d_ff (expand) → ReLU → d_ff → d_model (compress)

    The expansion factor (d_ff / d_model) is typically 4x, allowing the
    model to map through a higher-dimensional space.
    """
    def __init__(self, config):
        self.linear1 = Linear(config.d_model, config.d_ff)
        self.linear2 = Linear(config.d_ff, config.d_model)

    def forward(self, x):
        """
        Forward pass: Linear → ReLU → Linear

        ReLU backward: gradient passes through where input > 0, blocked where ≤ 0.
        This creates "sparse gradients" — only a subset of neurons get updated.
        """
        return self.linear2.forward(tensor_relu(self.linear1.forward(x)))

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding — Gives the model a sense of word ORDER.

    Since attention is permutation-invariant (it doesn't care about order),
    we add unique position signals using sine and cosine functions at
    different frequencies.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This is NOT learnable — it's a fixed constant. No gradients needed.
    The gradient simply passes through the addition unchanged.
    """
    def __init__(self, config):
        pe = np.zeros((config.max_seq_length, config.d_model))
        position = np.arange(0, config.max_seq_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, config.d_model, 2) * -(np.log(10000.0) / config.d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]  # (1, max_seq, d_model)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Since PE is a constant (not a Tensor), we use Tensor.__add__ with
        a raw numpy array. The gradient flows through to x unchanged.
        """
        seq_len = x.data.shape[1]
        return x + self.pe[:, :seq_len, :]

    def parameters(self):
        return []  # No learnable parameters


class EncoderBlock:
    """
    Transformer Encoder Block (Pre-LayerNorm variant):

        x → LayerNorm → Self-Attention → + (residual) → LayerNorm → FFN → + (residual)

    Pre-LN (normalizing BEFORE attention/FFN) is more stable for training
    than Post-LN (original Transformer paper) because it keeps the residual
    stream's magnitude stable.

    The residual connection (x + sublayer(x)) allows gradients to flow
    directly through the "skip connection", preventing vanishing gradients
    in deep models. This is why Transformers can be stacked 100+ layers deep.
    """
    def __init__(self, config):
        self.self_attn = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)

    def forward(self, x, src_mask):
        # Pre-LN Self-Attention + Residual
        normed = self.norm1.forward(x)
        attn_out = self.self_attn.forward(normed, normed, normed, src_mask)
        x = x + attn_out  # Residual connection — Tensor.__add__

        # Pre-LN Feed-Forward + Residual
        normed = self.norm2.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = x + ffn_out   # Residual connection

        return x

    def parameters(self):
        return (self.self_attn.parameters() + self.ffn.parameters() +
                self.norm1.parameters() + self.norm2.parameters())


class DecoderBlock:
    """
    Transformer Decoder Block (for encoder-decoder models):

        x → Norm → Masked Self-Attention → + →
        x → Norm → Cross-Attention(q=x, k=enc, v=enc) → + →
        x → Norm → FFN → +

    Cross-attention allows the decoder to "look at" the encoder output
    while generating tokens sequentially.
    """
    def __init__(self, config):
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.norm3 = LayerNorm(config.d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked Self-Attention
        normed = self.norm1.forward(x)
        self_attn_out = self.self_attn.forward(normed, normed, normed, tgt_mask)
        x = x + self_attn_out

        # Cross-Attention
        normed_x = self.norm2.forward(x)
        cross_attn_out = self.cross_attn.forward(normed_x, enc_output, enc_output, src_mask)
        x = x + cross_attn_out

        # Feed-Forward
        normed = self.norm3.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = x + ffn_out

        return x

    def parameters(self):
        return (self.self_attn.parameters() + self.cross_attn.parameters() +
                self.ffn.parameters() + self.norm1.parameters() +
                self.norm2.parameters() + self.norm3.parameters())


class DecoderOnlyBlock:
    """
    GPT-style Decoder-Only Block (for pretraining):

        x → LayerNorm → Causal Self-Attention → + (residual) →
        x → LayerNorm → FFN → + (residual)

    This is the same as EncoderBlock but with a CAUSAL MASK that prevents
    each position from attending to future positions. This is what makes
    the model "autoregressive" — it can only use past context to predict
    the next token.

    The causal mask is a lower-triangular matrix:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Position i can attend to positions 0..i but not i+1..n.
    """
    def __init__(self, config):
        self.self_attn = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)

    def forward(self, x, causal_mask):
        # Pre-LN Causal Self-Attention + Residual
        normed = self.norm1.forward(x)
        attn_out = self.self_attn.forward(normed, normed, normed, causal_mask)
        x = x + attn_out

        # Pre-LN Feed-Forward + Residual
        normed = self.norm2.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = x + ffn_out

        return x

    def parameters(self):
        return (self.self_attn.parameters() + self.ffn.parameters() +
                self.norm1.parameters() + self.norm2.parameters())