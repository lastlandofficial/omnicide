"""
================================================================================
 MULTI-HEAD ATTENTION — The Core of the Transformer
================================================================================
Implements Scaled Dot-Product Attention with multiple heads, now fully
differentiable through the Tensor autograd engine.

Every operation (linear projections, reshape, transpose, matmul, masking,
softmax) is a tracked Tensor operation, so calling loss.backward() will
propagate gradients all the way back through attention to the input embeddings.
================================================================================
"""
import numpy as np
from math_ops import Linear
from tensor import tensor_softmax, tensor_masked_fill


class MultiHeadAttention:
    """
    Multi-Head Attention: Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    Instead of performing a single attention function, we:
    1. Project Q, K, V into num_heads separate subspaces
    2. Run attention independently in each subspace (head)
    3. Concatenate results and project back

    This allows the model to jointly attend to information from
    different representation subspaces at different positions.
    """
    def __init__(self, config):
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # Linear projections for Q, K, V and the output
        self.w_q = Linear(self.d_model, self.d_model)
        self.w_k = Linear(self.d_model, self.d_model)
        self.w_v = Linear(self.d_model, self.d_model)
        self.w_o = Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        """
        Compute multi-head attention.

        Args:
            q, k, v (Tensor): Query, Key, Value — shape (batch, seq, d_model)
            mask (np.ndarray): Attention mask, broadcastable to (batch, heads, seq_q, seq_k)

        Returns:
            Tensor of shape (batch, seq_q, d_model)

        =====================================================================
        STEP-BY-STEP WITH SHAPES:
        =====================================================================
        1. Linear projections: (batch, seq, d_model) → (batch, seq, d_model)
        2. Split into heads:   (batch, seq, d_model) → (batch, seq, heads, head_dim)
        3. Transpose:          (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        4. Scaled dot-product: QKᵀ/√d_k → (batch, heads, seq_q, seq_k)
        5. Masking + Softmax:  (batch, heads, seq_q, seq_k)
        6. Weighted values:    Attn @ V → (batch, heads, seq_q, head_dim)
        7. Concat heads:       (batch, seq_q, heads, head_dim) → (batch, seq_q, d_model)
        8. Final projection:   (batch, seq_q, d_model) → (batch, seq_q, d_model)
        =====================================================================
        """
        batch_size = q.data.shape[0]
        seq_len_q = q.data.shape[1]
        seq_len_k = k.data.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Linear projections — each creates its own autograd node
        # ------------------------------------------------------------------
        Q = self.w_q.forward(q)  # (batch, seq_q, d_model)
        K = self.w_k.forward(k)  # (batch, seq_k, d_model)
        V = self.w_v.forward(v)  # (batch, seq_k, d_model)

        # ------------------------------------------------------------------
        # Step 2 & 3: Reshape into heads and transpose
        #   (batch, seq, d_model) → (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        # ------------------------------------------------------------------
        Q = Q.reshape((batch_size, seq_len_q, self.num_heads, self.head_dim))
        Q = Q.transpose((0, 2, 1, 3))  # (batch, heads, seq_q, head_dim)

        K = K.reshape((batch_size, seq_len_k, self.num_heads, self.head_dim))
        K = K.transpose((0, 2, 1, 3))  # (batch, heads, seq_k, head_dim)

        V = V.reshape((batch_size, seq_len_k, self.num_heads, self.head_dim))
        V = V.transpose((0, 2, 1, 3))  # (batch, heads, seq_k, head_dim)

        # ------------------------------------------------------------------
        # Step 4: Scaled Dot-Product Attention
        #   scores = Q @ Kᵀ / √d_k
        #   Scaling prevents dot products from getting too large, which would
        #   push softmax into regions with tiny gradients (saturation).
        # ------------------------------------------------------------------
        K_T = K.transpose((0, 1, 3, 2))   # (batch, heads, head_dim, seq_k)
        scores = (Q @ K_T) / np.sqrt(self.head_dim)  # (batch, heads, seq_q, seq_k)

        # ------------------------------------------------------------------
        # Step 5: Apply causal mask (if provided) and softmax
        #   Masked positions get -1e9 so softmax gives them ~0 probability
        # ------------------------------------------------------------------
        if mask is not None:
            scores = tensor_masked_fill(scores, mask == 0, -1e9)

        attention_weights = tensor_softmax(scores)  # (batch, heads, seq_q, seq_k)

        # ------------------------------------------------------------------
        # Step 6: Multiply attention weights by values
        #   (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, head_dim)
        #   = (batch, heads, seq_q, head_dim)
        # ------------------------------------------------------------------
        out = attention_weights @ V

        # ------------------------------------------------------------------
        # Step 7: Concatenate heads
        #   (batch, heads, seq_q, head_dim) → (batch, seq_q, heads, head_dim) → (batch, seq_q, d_model)
        # ------------------------------------------------------------------
        out = out.transpose((0, 2, 1, 3))  # (batch, seq_q, heads, head_dim)
        out = out.reshape((batch_size, seq_len_q, self.d_model))

        # ------------------------------------------------------------------
        # Step 8: Final linear projection
        # ------------------------------------------------------------------
        return self.w_o.forward(out)

    def parameters(self):
        """All learnable parameters: 4 linear layers × (W + b) = 8 tensors."""
        return (self.w_q.parameters() + self.w_k.parameters() +
                self.w_v.parameters() + self.w_o.parameters())