"""
================================================================================
 MATH OPERATIONS — Differentiable Building Blocks
================================================================================
Linear layers, Layer Normalization, and Embeddings — now backed by the Tensor
autograd engine so gradients flow automatically through backward().

Each class stores its learnable parameters as Tensor objects with
requires_grad=True, and exposes a parameters() method for the optimizer.
================================================================================
"""
import numpy as np
from tensor import Tensor, tensor_layer_norm, tensor_embedding


class Linear:
    """
    Fully-connected linear layer: y = x @ W + b

    This is the fundamental building block of neural networks.
    The weight matrix W learns to project from d_in to d_out dimensions.

    Parameters:
        W (Tensor): Weight matrix of shape (d_in, d_out), requires_grad=True
        b (Tensor): Bias vector of shape (d_out,), requires_grad=True

    Initialization:
        Kaiming (He) initialization: W ~ N(0, √(2/d_in))
        This keeps the variance of activations stable across layers,
        preventing the signal from exploding or vanishing.
    """
    def __init__(self, d_in, d_out):
        self.W = Tensor(
            np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in),
            requires_grad=True
        )
        self.b = Tensor(
            np.zeros(d_out),
            requires_grad=True
        )

    def forward(self, x):
        """
        Forward pass: y = x @ W + b

        The matmul (x @ W) and addition (+ b) are both Tensor operations,
        so the backward pass is handled automatically by the autograd engine.

        Shapes:
            x:      (batch, seq, d_in)   or (batch, d_in)
            W:      (d_in, d_out)
            x @ W:  (batch, seq, d_out)  — np.matmul broadcasts W across batch
            b:      (d_out,)             — NumPy broadcasts across batch and seq
            output: (batch, seq, d_out)
        """
        return (x @ self.W) + self.b

    def parameters(self):
        """Return all learnable parameters."""
        return [self.W, self.b]


class LayerNorm:
    """
    Layer Normalization: y = γ · (x - μ) / √(σ² + ε) + β

    Normalizes activations across the feature (last) dimension.
    This stabilizes training by ensuring each layer receives inputs
    with zero mean and unit variance, regardless of the scale of
    previous layers' outputs.

    Parameters:
        gamma (Tensor): Learnable scale, shape (d_model,), initialized to 1
        beta (Tensor):  Learnable shift, shape (d_model,), initialized to 0
    """
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True)

    def forward(self, x):
        """
        Forward pass: delegates to tensor_layer_norm which handles both
        the forward computation and the backward gradient computation.
        """
        return tensor_layer_norm(x, self.gamma, self.beta, self.eps)

    def parameters(self):
        return [self.gamma, self.beta]


class Embedding:
    """
    Token Embedding Table: converts integer token IDs to dense vectors.

    This is just a lookup table — each token ID selects a row from the
    weight matrix. During training, these vectors are learned to capture
    semantic meaning of each token.

    Parameters:
        weights (Tensor): Shape (vocab_size, d_model), requires_grad=True
    """
    def __init__(self, vocab_size, d_model):
        self.weights = Tensor(
            np.random.randn(vocab_size, d_model) * 0.02,
            requires_grad=True
        )

    def forward(self, x):
        """
        x is a numpy array of integer token IDs (NOT a Tensor).
        Returns a Tensor of shape (*x.shape, d_model).

        Example:
            x = [[4, 7, 2], [1, 5, 9]]  — batch of 2 sequences, length 3
            output shape: (2, 3, d_model) — each ID replaced by its d_model vector
        """
        return tensor_embedding(self.weights, x)

    def parameters(self):
        return [self.weights]