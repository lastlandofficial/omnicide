"""
Example: Test the forward pass with the Tensor-based Transformer.
"""
import numpy as np
from config import TransformerConfig
from model import Transformer, DecoderOnlyTransformer


def test_encoder_decoder():
    """Test the original encoder-decoder Transformer with Tensor autograd."""
    print("=== Encoder-Decoder Transformer (Forward Pass) ===")
    config = TransformerConfig(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_layers=2,
    )

    model = Transformer(config)

    np.random.seed(42)
    src_data = np.random.randint(0, config.src_vocab_size, (2, 5))
    tgt_data = np.random.randint(0, config.tgt_vocab_size, (2, 5))

    print("Running forward pass...")
    output = model.forward(src=src_data, tgt=tgt_data)

    print(f"Source shape: {src_data.shape}")
    print(f"Target shape: {tgt_data.shape}")
    print(f"Output shape: {output.shape}")  # Uses Tensor.shape property
    print(f"Output type:  {type(output)}")
    print(f"Parameters:   {sum(p.data.size for p in model.parameters()):,}")
    print()


def test_decoder_only():
    """Test the new GPT-style decoder-only Transformer."""
    print("=== Decoder-Only Transformer (Forward Pass) ===")
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=128,
        max_seq_length=64,
    )

    model = DecoderOnlyTransformer(config)

    np.random.seed(42)
    input_ids = np.random.randint(0, config.vocab_size, (2, 10))

    print("Running forward pass...")
    logits = model.forward(input_ids)

    print(f"Input shape:  {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")  # Expected: (2, 10, 50)
    print(f"Parameters:   {sum(p.data.size for p in model.parameters()):,}")
    print()


if __name__ == "__main__":
    test_encoder_decoder()
    test_decoder_only()