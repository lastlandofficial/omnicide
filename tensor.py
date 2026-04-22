"""
================================================================================
 TENSOR AUTOGRAD ENGINE — The Heart of Backpropagation
================================================================================
This module implements a Tensor class that wraps NumPy arrays and automatically
tracks the computation graph for reverse-mode automatic differentiation (backprop).

Inspired by Andrej Karpathy's micrograd, but scaled from scalars to N-dimensional
NumPy arrays. Every operation records how to compute ∂Loss/∂(inputs) given 
∂Loss/∂(output), which is the chain rule applied recursively.

Key Concept — The Chain Rule in Matrix Form:
    If L = f(g(x)), then ∂L/∂x = ∂L/∂g · ∂g/∂x
    Each operation's _backward() computes this "local gradient" multiplication.
================================================================================
"""
import numpy as np


# ==============================================================================
# HELPER: Reverse Broadcasting
# ==============================================================================
def _unbroadcast(grad, target_shape):
    """
    When NumPy broadcasts during the forward pass (e.g., adding a bias of shape
    (d_out,) to a tensor of shape (batch, seq, d_out)), the gradient has the
    BROADCASTED shape. We need to sum over the broadcast dimensions to get the
    gradient back to the original (smaller) shape.

    Example:
        forward:  (batch, seq, d_out) + (d_out,) → (batch, seq, d_out)
        backward: grad shape (batch, seq, d_out) → bias grad shape (d_out,)
                  We sum over axes 0 and 1.
    """
    # Step 1: Sum over leading dimensions that were added by broadcasting
    # e.g., grad (batch, seq, d_out) → target (d_out,): sum axes 0, 1
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Step 2: Sum over dimensions where target had size 1 (keepdims broadcasting)
    # e.g., grad (batch, seq, d_out) → target (1, 1, d_out): sum axes 0, 1
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ==============================================================================
# THE TENSOR CLASS
# ==============================================================================
class Tensor:
    """
    A wrapper around a NumPy array that tracks the computation graph for
    automatic differentiation. Analogous to PyTorch's torch.Tensor.

    Attributes:
        data (np.ndarray):        The actual numerical values (forward pass result)
        grad (np.ndarray):        ∂Loss/∂(this tensor), same shape as data
        requires_grad (bool):     True for learnable parameters (weights, biases)
        _backward (callable):     Closure that computes gradients for parent tensors
        _prev (set[Tensor]):      Parent tensors in the computation graph
        _op (str):                Name of the operation that created this tensor
    """

    def __init__(self, data, _children=(), _op='', requires_grad=False):
        # Convert to float64 NumPy array for numerical stability
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        # Gradient accumulator — initialized to zeros, same shape as data
        self.grad = np.zeros_like(self.data, dtype=np.float64)

        # Whether this tensor is a learnable parameter
        self.requires_grad = requires_grad

        # Backpropagation function — set by each operation
        self._backward = lambda: None

        # Parents in the computation graph (for topological sort)
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}', requires_grad={self.requires_grad})"

    # ==================================================================
    # BACKWARD PASS — Reverse-Mode Automatic Differentiation
    # ==================================================================
    def backward(self):
        """
        Compute gradients for all tensors in the computation graph using
        reverse-mode autodiff (backpropagation).

        Algorithm:
        1. Build a topological ordering of all tensor nodes (parents before children)
        2. Set self.grad = 1 (∂L/∂L = 1, the seed gradient)
        3. Walk the graph in REVERSE topological order, calling each node's _backward()
           to propagate gradients to its parents via the chain rule

        This is identical to engine.py's Value.backward(), but for N-D arrays.
        """
        # --- Step 1: Topological sort via DFS ---
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # --- Step 2: Seed gradient ---
        # The derivative of the loss with respect to itself is 1
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # --- Step 3: Reverse-order chain rule ---
        for node in reversed(topo):
            node._backward()

    # ==================================================================
    # OPERATOR OVERLOADS — Natural Python Syntax for Tensor Math
    # ==================================================================

    def __add__(self, other):
        """
        Element-wise addition: C = A + B

        Backward (Chain Rule):
            ∂L/∂A = ∂L/∂C · ∂C/∂A = ∂L/∂C · 1 = ∂L/∂C
            ∂L/∂B = ∂L/∂C · ∂C/∂B = ∂L/∂C · 1 = ∂L/∂C

            With broadcasting, we sum over the broadcast dims.
        """
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, (self, other), '+')
            a, b = self, other

            def _backward():
                a.grad = a.grad + _unbroadcast(out.grad, a.data.shape)
                b.grad = b.grad + _unbroadcast(out.grad, b.data.shape)

            out._backward = _backward
            return out
        else:
            # Adding a constant (numpy array or scalar) — no gradient for the constant
            const = np.asarray(other, dtype=np.float64)
            out = Tensor(self.data + const, (self,), '+const')
            a = self

            def _backward():
                a.grad = a.grad + _unbroadcast(out.grad, a.data.shape)

            out._backward = _backward
            return out

    def __radd__(self, other):
        """Handle: constant + Tensor → Tensor + constant"""
        return self.__add__(other)

    def __matmul__(self, other):
        """
        Matrix multiplication: C = A @ B   (using np.matmul for batched support)

        =====================================================================
        THE CORE LINEAR ALGEBRA OF BACKPROP:
        =====================================================================
        If C = A @ B where A ∈ ℝ^(n×m) and B ∈ ℝ^(m×k), then C ∈ ℝ^(n×k).

        Using the chain rule:
            ∂L/∂A = ∂L/∂C @ Bᵀ       (n×k) @ (k×m) = (n×m) ✓ matches A
            ∂L/∂B = Aᵀ @ ∂L/∂C       (m×n) @ (n×k) = (m×k) ✓ matches B

        For BATCHED matmul (e.g., in multi-head attention):
            A ∈ ℝ^(batch, heads, seq_q, d_k)
            B ∈ ℝ^(batch, heads, d_k, seq_k)
            The same formulas apply — "transpose" means swapping the last two axes.
            np.matmul handles the batch dims automatically.

        If B was broadcast (e.g., a 2D weight matrix used with 3D input),
        we sum the gradient over the extra batch dimensions.
        =====================================================================
        """
        out = Tensor(np.matmul(self.data, other.data), (self, other), '@')
        a, b = self, other

        def _backward():
            # ∂L/∂A = ∂L/∂C @ Bᵀ
            a_grad = np.matmul(out.grad, np.swapaxes(b.data, -2, -1))
            # ∂L/∂B = Aᵀ @ ∂L/∂C
            b_grad = np.matmul(np.swapaxes(a.data, -2, -1), out.grad)

            # Handle broadcasting (e.g., 2D weight matrix used with 3D batched input)
            a.grad = a.grad + _unbroadcast(a_grad, a.data.shape)
            b.grad = b.grad + _unbroadcast(b_grad, b.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Element-wise multiplication: C = A * B

        Backward (Product Rule):
            ∂L/∂A = ∂L/∂C * B      (element-wise)
            ∂L/∂B = ∂L/∂C * A      (element-wise)
        """
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, (self, other), '*')
            a, b = self, other

            def _backward():
                a_grad = out.grad * b.data
                b_grad = out.grad * a.data
                a.grad = a.grad + _unbroadcast(a_grad, a.data.shape)
                b.grad = b.grad + _unbroadcast(b_grad, b.data.shape)

            out._backward = _backward
            return out
        else:
            # Scalar or array constant multiplication
            const = np.float64(other)
            out = Tensor(self.data * const, (self,), '*const')
            a = self

            def _backward():
                a.grad = a.grad + out.grad * const

            out._backward = _backward
            return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        """
        Division by a scalar: C = A / s

        Backward:
            ∂L/∂A = ∂L/∂C / s
        """
        s = np.float64(scalar)
        out = Tensor(self.data / s, (self,), '/')
        a = self

        def _backward():
            a.grad = a.grad + out.grad / s

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation: -A"""
        return self * (-1.0)

    def __sub__(self, other):
        """Subtraction: A - B = A + (-B)"""
        if isinstance(other, Tensor):
            return self + (other * (-1.0))
        else:
            return self + (-np.asarray(other, dtype=np.float64))

    # ==================================================================
    # SHAPE OPERATIONS — Reshape and Transpose with Gradient Tracking
    # ==================================================================

    def reshape(self, new_shape):
        """
        Reshape tensor to a new shape.

        Backward:
            Reshape the gradient back to the original shape.
            Reshaping doesn't change the data, it just reinterprets it.
        """
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(new_shape), (self,), 'reshape')
        a = self

        def _backward():
            a.grad = a.grad + out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def transpose(self, axes):
        """
        Transpose (permute) tensor axes.

        Backward:
            Apply the INVERSE permutation to the gradient.
            If forward permutes [0,2,1,3], backward permutes [0,2,1,3]
            (which happens to be its own inverse in this case).

        Math:
            If Y = permute(X, axes), then to undo:
            inv_axes[axes[i]] = i  for all i
            ∂L/∂X = permute(∂L/∂Y, inv_axes)
        """
        out = Tensor(self.data.transpose(axes), (self,), 'transpose')
        a = self

        # Compute inverse permutation
        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i

        def _backward():
            a.grad = a.grad + out.grad.transpose(inv_axes)

        out._backward = _backward
        return out


# ==============================================================================
# STANDALONE OPERATIONS — Complex Functions with Careful Backward Passes
# ==============================================================================

def tensor_softmax(x, axis=-1):
    """
    Softmax activation: s_i = exp(x_i) / Σ_j exp(x_j)

    =========================================================================
    FORWARD PASS:
        1. Subtract max for numerical stability (prevents exp overflow)
        2. Compute exp(x - max)
        3. Normalize by the sum
    =========================================================================

    =========================================================================
    BACKWARD PASS — The Softmax Jacobian:
    =========================================================================
    The full Jacobian of softmax is:
        ∂s_i/∂x_j = s_i(δ_ij - s_j)

    where δ_ij is the Kronecker delta. This is an N×N matrix, which is
    expensive to compute. But we can avoid materializing it:

        ∂L/∂x_i = Σ_j (∂L/∂s_j) · (∂s_j/∂x_i)
                 = Σ_j (∂L/∂s_j) · s_j · (δ_ji - s_i)
                 = (∂L/∂s_i) · s_i - s_i · Σ_j (∂L/∂s_j · s_j)
                 = s_i · (∂L/∂s_i - Σ_j (∂L/∂s_j · s_j))

    In vector form:
        ∂L/∂x = s ⊙ (∂L/∂s - <∂L/∂s, s>)

    where ⊙ is element-wise multiplication and <·,·> is the dot product
    (summed over the softmax axis).
    =========================================================================
    """
    # --- Forward ---
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    softmax_out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    out = Tensor(softmax_out, (x,), 'softmax')

    def _backward():
        # <∂L/∂s, s> — dot product along the softmax axis
        dot = np.sum(out.grad * softmax_out, axis=axis, keepdims=True)
        # s ⊙ (∂L/∂s - <∂L/∂s, s>)
        x.grad = x.grad + softmax_out * (out.grad - dot)

    out._backward = _backward
    return out


def tensor_layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization: y = γ · (x - μ) / √(σ² + ε) + β

    Normalizes across the LAST dimension (feature dimension) of x.

    =========================================================================
    FORWARD PASS:
        μ = mean(x, axis=-1)           — mean over features
        σ² = var(x, axis=-1)           — variance over features
        x̂ = (x - μ) / √(σ² + ε)       — normalized
        y = γ · x̂ + β                  — scale and shift
    =========================================================================

    =========================================================================
    BACKWARD PASS — Three Gradients to Compute:
    =========================================================================
    Let N = feature dimension size, dx̂ = ∂L/∂ŷ · γ

    1) ∂L/∂γ = Σ(∂L/∂y · x̂)          summed over batch dimensions
    2) ∂L/∂β = Σ(∂L/∂y)              summed over batch dimensions
    3) ∂L/∂x:
       This is the tricky one because x appears in μ and σ² too.

       Using the efficient combined formula (derived in LayerNorm papers):

       ∂L/∂x = (1/N) · (1/σ) · [N·dx̂ - Σ(dx̂) - x̂ · Σ(dx̂ · x̂)]

       where all sums are over the last axis (feature dimension).

       DERIVATION SKETCH:
       ∂L/∂x_i = ∂L/∂x̂_i · (1/σ)                             [direct path]
               + ∂L/∂σ² · ∂σ²/∂x_i                            [through variance]
               + ∂L/∂μ · ∂μ/∂x_i                               [through mean]

       After expanding and simplifying (noting Σ(x_i - μ) = 0):
       ∂L/∂x = (1/(Nσ)) · [N·dx̂ - Σ(dx̂) - x̂·Σ(dx̂·x̂)]
    =========================================================================
    """
    # --- Forward ---
    N = x.data.shape[-1]  # Feature dimension size
    mean = np.mean(x.data, axis=-1, keepdims=True)
    var = np.var(x.data, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)             # 1/σ
    x_norm = (x.data - mean) * std_inv              # x̂ = (x - μ) / σ
    out_data = gamma.data * x_norm + beta.data      # y = γ · x̂ + β

    out = Tensor(out_data, (x, gamma, beta), 'layer_norm')

    def _backward():
        # Intermediate: dx̂ = ∂L/∂y · γ
        dx_norm = out.grad * gamma.data

        # Sum axes for γ and β gradients (all dims except the last)
        batch_axes = tuple(range(out.grad.ndim - 1))

        # ∂L/∂γ = Σ(∂L/∂y · x̂) over batch dims
        gamma.grad = gamma.grad + np.sum(out.grad * x_norm, axis=batch_axes)

        # ∂L/∂β = Σ(∂L/∂y) over batch dims
        beta.grad = beta.grad + np.sum(out.grad, axis=batch_axes)

        # ∂L/∂x = (1/(N·σ)) · [N·dx̂ - Σ(dx̂) - x̂·Σ(dx̂·x̂)]
        dx = (1.0 / N) * std_inv * (
            N * dx_norm
            - np.sum(dx_norm, axis=-1, keepdims=True)
            - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        x.grad = x.grad + dx

    out._backward = _backward
    return out


def tensor_relu(x):
    """
    ReLU activation: f(x) = max(0, x)

    Backward:
        ∂L/∂x = ∂L/∂f · ∂f/∂x
        ∂f/∂x = 1 if x > 0, else 0

        The gradient passes through unchanged where x > 0,
        and is zeroed out where x ≤ 0. This is called the
        "gradient gate" — ReLU acts as an on/off switch for gradients.
    """
    out = Tensor(np.maximum(0, x.data), (x,), 'relu')

    def _backward():
        x.grad = x.grad + out.grad * (x.data > 0).astype(np.float64)

    out._backward = _backward
    return out


def tensor_embedding(weight, indices):
    """
    Embedding table lookup: output = W[indices]

    This is NOT a matrix multiply — it's a fancy indexing operation.
    Each integer in `indices` selects a row from the weight matrix.

    Args:
        weight (Tensor): Embedding matrix of shape (vocab_size, d_model)
        indices (np.ndarray): Integer token IDs, any shape (e.g., (batch, seq))

    Returns:
        Tensor of shape (*indices.shape, d_model)

    =========================================================================
    BACKWARD PASS — Scatter-Add:
    =========================================================================
    If token ID 5 appeared 3 times in the batch, the gradient for row 5
    of the embedding matrix is the SUM of the 3 output gradients.

    This is the "scatter-add" pattern:
        ∂L/∂W[i] = Σ (∂L/∂output[j])  for all j where indices[j] == i

    We use np.add.at() which performs unbuffered in-place addition at
    specified indices, correctly handling duplicate indices.
    =========================================================================
    """
    out = Tensor(weight.data[indices], (weight,), 'embedding')

    def _backward():
        np.add.at(weight.grad, indices, out.grad)

    out._backward = _backward
    return out


def tensor_masked_fill(x, mask_zero, fill_value=-1e9):
    """
    Masked fill for causal attention masking.

    Where mask_zero is True (i.e., where the attention mask is 0),
    replace the value with fill_value (a very large negative number).
    This makes softmax assign ~0 probability to these positions.

    Args:
        x (Tensor):                  Attention scores
        mask_zero (np.ndarray bool): True where mask == 0 (positions to mask out)
        fill_value (float):          Value to fill (default: -1e9 ≈ -∞)

    Backward:
        Gradient is zeroed at masked positions (since those values are
        replaced by a constant and don't depend on the input).
        ∂L/∂x = ∂L/∂out * (1 where not masked, 0 where masked)
    """
    out = Tensor(np.where(mask_zero, fill_value, x.data), (x,), 'masked_fill')
    # Pre-compute the mask for the backward pass
    pass_mask = (~mask_zero).astype(np.float64)

    def _backward():
        x.grad = x.grad + out.grad * pass_mask

    out._backward = _backward
    return out
