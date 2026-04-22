"""
================================================================================
 TRANSFORMER CONFIGURATION
================================================================================
Centralizes all hyperparameters for both Encoder-Decoder and Decoder-Only models.
================================================================================
"""
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # --- Vocabulary ---
    src_vocab_size: int = 1000   # For encoder-decoder models (source language)
    tgt_vocab_size: int = 1000   # For encoder-decoder models (target language)
    vocab_size: int = 1000       # For decoder-only models (single vocabulary)

    # --- Model Architecture ---
    d_model: int = 64            # Embedding dimension (width of the model)
    num_heads: int = 4           # Number of attention heads
    num_layers: int = 2          # Number of transformer blocks
    d_ff: int = 256              # Feed-forward hidden dimension (typically 4 × d_model)
    max_seq_length: int = 100    # Maximum sequence length for positional encoding