"""
================================================================================
 TRANSFORMER MODELS — Full Architecture with Autograd Support
================================================================================
Two model architectures:

1. Transformer (encoder-decoder): For seq-to-seq tasks (translation, etc.)
   - Kept from the original codebase, now using Tensor autograd.

2. DecoderOnlyTransformer (GPT-style): For next-token prediction pretraining.
   - New! This is what we use for pretraining on a text corpus.
   - Decoder-only because we don't need an encoder for language modeling.

Both models expose a parameters() method that returns all learnable Tensors,
which the optimizer uses to update weights during training.
================================================================================
"""
import numpy as np
from math_ops import Embedding, LayerNorm, Linear
from layers import (
    PositionalEncoding, EncoderBlock, DecoderBlock, DecoderOnlyBlock
)
from tensor import tensor_softmax


# ==============================================================================
# ENCODER-DECODER TRANSFORMER (Original Architecture)
# ==============================================================================
class Transformer:
    """
    Full Encoder-Decoder Transformer (Vaswani et al., 2017).

    Architecture:
        Encoder: src → Embedding → PosEnc → N × EncoderBlock → LayerNorm
        Decoder: tgt → Embedding → PosEnc → N × DecoderBlock → LayerNorm → Linear

    Now fully differentiable — calling loss.backward() after a forward pass
    will compute gradients for all parameters.
    """
    def __init__(self, config):
        self.config = config

        self.src_embedding = Embedding(config.src_vocab_size, config.d_model)
        self.tgt_embedding = Embedding(config.tgt_vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)

        self.encoder_blocks = [EncoderBlock(config) for _ in range(config.num_layers)]
        self.decoder_blocks = [DecoderBlock(config) for _ in range(config.num_layers)]

        self.fc_out = Linear(config.d_model, config.tgt_vocab_size)
        self.norm = LayerNorm(config.d_model)

    def generate_causal_mask(self, seq_len):
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask[np.newaxis, np.newaxis, :, :]

    def encode(self, src, src_mask=None):
        x = self.src_embedding.forward(src)
        x = self.pos_encoding.forward(x)

        for block in self.encoder_blocks:
            x = block.forward(x, src_mask)

        return self.norm.forward(x)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        seq_len = tgt.shape[1]
        causal_mask = self.generate_causal_mask(seq_len)

        if tgt_mask is not None:
            tgt_mask = tgt_mask * causal_mask
        else:
            tgt_mask = causal_mask

        x = self.tgt_embedding.forward(tgt)
        x = self.pos_encoding.forward(x)

        for block in self.decoder_blocks:
            x = block.forward(x, enc_output, src_mask, tgt_mask)

        return self.norm.forward(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        out = self.fc_out.forward(dec_output)
        return out

    def parameters(self):
        """Collect all learnable parameters from every sub-module."""
        params = []
        params += self.src_embedding.parameters()
        params += self.tgt_embedding.parameters()
        for block in self.encoder_blocks:
            params += block.parameters()
        for block in self.decoder_blocks:
            params += block.parameters()
        params += self.fc_out.parameters()
        params += self.norm.parameters()
        return params


# ==============================================================================
# DECODER-ONLY TRANSFORMER (GPT-Style — For Pretraining)
# ==============================================================================
class DecoderOnlyTransformer:
    """
    GPT-style Decoder-Only Transformer for autoregressive language modeling.

    =========================================================================
    WHY DECODER-ONLY?
    =========================================================================
    For next-token prediction (language modeling), we only need a decoder:
    - Input:  sequence of tokens [t₁, t₂, ..., tₙ]
    - Output: predicted next token at each position [t₂, t₃, ..., tₙ₊₁]

    Each position can only attend to itself and previous positions (causal mask),
    making the model autoregressive — it generates one token at a time,
    conditioning on all previously generated tokens.

    This is the architecture behind GPT-1, GPT-2, GPT-3, LLaMA, etc.
    =========================================================================

    Architecture:
        tokens → Embedding → PosEnc → N × DecoderOnlyBlock → LayerNorm → Linear → logits

    The final Linear layer projects from d_model to vocab_size, producing
    a score (logit) for each token in the vocabulary. These logits are then
    passed through softmax (in the loss function) to get probabilities.
    """
    def __init__(self, config):
        self.config = config

        # Token embedding: integer IDs → dense vectors
        self.embedding = Embedding(config.vocab_size, config.d_model)

        # Positional encoding: adds position information
        self.pos_encoding = PositionalEncoding(config)

        # Stack of decoder-only blocks (the core of the model)
        self.blocks = [DecoderOnlyBlock(config) for _ in range(config.num_layers)]

        # Final layer norm (Pre-LN architecture)
        self.norm = LayerNorm(config.d_model)

        # Output projection: d_model → vocab_size (one logit per vocab token)
        self.fc_out = Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        """
        Forward pass for next-token prediction.

        Args:
            x (np.ndarray): Integer token IDs, shape (batch, seq_len)

        Returns:
            Tensor: Logits of shape (batch, seq_len, vocab_size)
                    logits[b, t, v] = score for token v at position t in batch b
        """
        seq_len = x.shape[1]

        # --- Causal Mask ---
        # Lower triangular: position i can attend to positions 0..i
        # Shape: (1, 1, seq_len, seq_len) — broadcasts over batch and heads
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]

        # --- Embedding + Positional Encoding ---
        h = self.embedding.forward(x)       # (batch, seq, d_model)
        h = self.pos_encoding.forward(h)    # (batch, seq, d_model)

        # --- Decoder Blocks ---
        for block in self.blocks:
            h = block.forward(h, causal_mask)

        # --- Final Norm + Linear Projection ---
        h = self.norm.forward(h)            # (batch, seq, d_model)
        logits = self.fc_out.forward(h)     # (batch, seq, vocab_size)

        return logits

    def parameters(self):
        """Collect ALL learnable parameters for the optimizer."""
        params = []
        params += self.embedding.parameters()
        # Positional encoding has no learnable params
        for block in self.blocks:
            params += block.parameters()
        params += self.norm.parameters()
        params += self.fc_out.parameters()
        return params

    def generate(self, start_ids, max_new_tokens, temperature=1.0):
        """
        Autoregressive text generation (greedy or with temperature sampling).

        Args:
            start_ids (np.ndarray): Seed tokens, shape (1, prompt_len)
            max_new_tokens (int):   How many tokens to generate
            temperature (float):    Controls randomness. 1.0 = normal, <1 = more greedy

        Returns:
            np.ndarray: Generated token IDs, shape (1, prompt_len + max_new_tokens)
        """
        ids = start_ids.copy()

        for _ in range(max_new_tokens):
            # Crop to max sequence length (sliding window)
            context = ids[:, -self.config.max_seq_length:]

            # Forward pass — get logits
            logits = self.forward(context)

            # Take logits at the LAST position only
            last_logits = logits.data[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            last_logits = last_logits / temperature

            # Convert to probabilities
            exp_logits = np.exp(last_logits - np.max(last_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Sample from the distribution
            next_id = np.array([[np.random.choice(probs.shape[-1], p=probs[0])]])

            # Append to sequence
            ids = np.concatenate([ids, next_id], axis=1)

        return ids