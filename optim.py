"""
================================================================================
 ADAM OPTIMIZER — Adaptive Moment Estimation
================================================================================
The Adam optimizer (Kingma & Ba, 2014) is the workhorse of deep learning.
It maintains per-parameter running averages of the gradient (1st moment)
and the squared gradient (2nd moment) to adapt the learning rate for each
parameter individually.

================================================================================
THE MATH:
================================================================================
At each step t, for each parameter θ:

    1. m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          [1st moment: gradient mean]
    2. v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          [2nd moment: gradient variance]
    3. m̂_t = m_t / (1 - β₁ᵗ)                          [bias correction for m]
    4. v̂_t = v_t / (1 - β₂ᵗ)                          [bias correction for v]
    5. θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)          [parameter update]

WHY ADAM WORKS SO WELL:
    - m (1st moment): Provides momentum — smooths out noisy gradients
    - v (2nd moment): Normalizes by gradient magnitude — handles sparse
      gradients and parameters at different scales
    - Bias correction: Compensates for m and v being initialized at 0,
      which would otherwise underestimate the true moments early in training

TYPICAL HYPERPARAMETERS:
    - lr = 1e-3 to 3e-4 for Transformers
    - β₁ = 0.9 (gradient momentum)
    - β₂ = 0.999 (squared gradient momentum)
    - ε = 1e-8 (prevents division by zero)

================================================================================
"""
import numpy as np


class Adam:
    """
    Adam Optimizer — the standard choice for training Transformers.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        Args:
            params: List of Tensor objects (model.parameters())
            lr:     Learning rate — controls step size
            betas:  (β₁, β₂) — exponential decay rates for moment estimates
            eps:    Small constant for numerical stability in division
        """
        # Only optimize parameters that need gradients
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # Time step counter (for bias correction)
        self.t = 0

        # Initialize moment estimates to zero for each parameter
        # m = 1st moment (mean of gradients)
        # v = 2nd moment (mean of squared gradients)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """
        Perform a single optimization step — update all parameters.

        This is called AFTER loss.backward() has computed all gradients.
        Each parameter is updated using its accumulated gradient and the
        Adam moving averages.
        """
        self.t += 1  # Increment timestep

        for i, param in enumerate(self.params):
            g = param.grad  # The gradient computed by backward()

            # ============================================================
            # Step 1: Update biased first moment estimate (gradient mean)
            #   m = β₁ · m + (1 - β₁) · g
            #   This is an exponential moving average of the gradient.
            #   β₁ = 0.9 means the last ~10 gradients contribute most.
            # ============================================================
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # ============================================================
            # Step 2: Update biased second moment estimate (gradient variance)
            #   v = β₂ · v + (1 - β₂) · g²
            #   This tracks how "large" the gradients typically are.
            #   Parameters with consistently large gradients get smaller updates.
            # ============================================================
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # ============================================================
            # Step 3: Bias correction
            #   Since m and v are initialized to 0, they're biased toward 0
            #   in early steps. This correction compensates:
            #   m̂ = m / (1 - β₁ᵗ)     — for t=1: m̂ = m/0.1 = 10·m
            #   v̂ = v / (1 - β₂ᵗ)     — for t=1: v̂ = v/0.001 = 1000·v
            # ============================================================
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # ============================================================
            # Step 4: Update parameter
            #   θ -= lr · m̂ / (√v̂ + ε)
            #
            #   Intuition: m̂ provides the direction (with momentum),
            #   √v̂ provides adaptive scaling (large gradients → smaller steps).
            #   ε prevents division by zero when v̂ ≈ 0.
            # ============================================================
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Reset all parameter gradients to zero.

        MUST be called before each backward pass, otherwise gradients
        accumulate across iterations (which is sometimes useful for
        gradient accumulation, but not for standard training).
        """
        for param in self.params:
            param.grad = np.zeros_like(param.data)


def clip_grad_norm(params, max_norm):
    """
    Gradient clipping by global norm — prevents exploding gradients.

    Computes the L2 norm of all gradients concatenated together.
    If this total norm exceeds max_norm, all gradients are scaled down
    proportionally so the total norm equals max_norm.

    This is critical for Transformer training stability, especially
    in the early steps when gradients can be erratic.

    Args:
        params: List of Tensor objects
        max_norm: Maximum allowed gradient norm (typically 1.0)

    Returns:
        float: The actual gradient norm (before clipping)
    """
    # Compute global gradient norm: ||g||₂ = √(Σ gᵢ²)
    total_norm_sq = 0.0
    grad_params = [p for p in params if p.requires_grad]

    for p in grad_params:
        total_norm_sq += np.sum(p.grad ** 2)

    total_norm = np.sqrt(total_norm_sq)

    # If norm exceeds threshold, scale all gradients down
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in grad_params:
            p.grad *= scale

    return total_norm
