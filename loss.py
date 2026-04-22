"""
================================================================================
 CROSS-ENTROPY LOSS — Measuring How Wrong the Model Is
================================================================================
Implements the standard loss function for classification / next-token prediction:
    Loss = -mean( log P(correct_token) )

This is "fused" softmax + cross-entropy: we combine the softmax probability
computation with the negative log-likelihood loss into one numerically stable
operation. This also gives us a beautifully simple gradient.

================================================================================
THE MATH IN DETAIL:
================================================================================

FORWARD PASS:
    Given raw logits z ∈ ℝ^V (one score per vocab token) and target y (integer):

    1. Log-Softmax (numerically stable):
       log P(i) = z_i - log(Σ_j exp(z_j))
                = z_i - max(z) - log(Σ_j exp(z_j - max(z)))

    2. Negative Log-Likelihood:
       Loss = -log P(y) = -(z_y - log(Σ_j exp(z_j)))

    3. Mean over all tokens in the batch.

BACKWARD PASS:
    The beauty of fusing softmax + cross-entropy:

    ∂Loss/∂z_i = P(i) - 𝟙(i = y)

    That is:  softmax(logits) - one_hot(target)

    This is one of the most elegant results in deep learning:
    - For the correct class:     gradient = P(correct) - 1   (negative, pushes logit UP)
    - For incorrect classes:     gradient = P(incorrect)      (positive, pushes logit DOWN)

    The magnitude is proportional to the model's confidence error.
    If P(correct) ≈ 1, gradient ≈ 0 (already correct, tiny update).
    If P(correct) ≈ 0, gradient ≈ -1 (very wrong, large update).

================================================================================
"""
import numpy as np
from tensor import Tensor


class CrossEntropyLoss:
    """
    Cross-Entropy Loss with fused Softmax.

    Numerically stable implementation that avoids computing softmax
    separately (which can overflow/underflow with float64).
    """

    def __call__(self, logits, targets):
        """
        Compute cross-entropy loss.

        Args:
            logits (Tensor):      Raw scores, shape (batch, seq_len, vocab_size)
            targets (np.ndarray): Target token IDs, shape (batch, seq_len)

        Returns:
            Tensor: Scalar loss value (with backward function set for backprop)
        """
        batch_size, seq_len, vocab_size = logits.data.shape
        N = batch_size * seq_len  # Total number of predictions

        # Flatten for easier indexing
        logits_2d = logits.data.reshape(N, vocab_size)    # (N, V)
        targets_1d = targets.reshape(N)                    # (N,)

        # ================================================================
        # FORWARD: Numerically Stable Log-Softmax
        # ================================================================
        # Step 1: Subtract max for numerical stability
        #   This doesn't change the softmax output (it cancels out),
        #   but prevents exp() from overflowing to inf.
        logits_max = np.max(logits_2d, axis=-1, keepdims=True)   # (N, 1)
        logits_shifted = logits_2d - logits_max                   # (N, V)

        # Step 2: Compute softmax probabilities
        exp_logits = np.exp(logits_shifted)                       # (N, V)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)      # (N, 1)
        probs = exp_logits / sum_exp                               # (N, V)

        # Step 3: Log-softmax = logits_shifted - log(sum_exp)
        log_probs = logits_shifted - np.log(sum_exp)              # (N, V)

        # Step 4: Pick the log-probability of the correct target token
        #   loss_per_token[i] = -log P(target[i])
        loss_per_token = -log_probs[np.arange(N), targets_1d]     # (N,)

        # Step 5: Mean reduction over all tokens
        loss_value = np.mean(loss_per_token)

        # ================================================================
        # Create loss Tensor with backward function
        # ================================================================
        loss = Tensor(np.array(loss_value), (logits,), 'cross_entropy')

        def _backward():
            """
            ================================================================
            BACKWARD: The Elegant Softmax-CrossEntropy Gradient
            ================================================================
            ∂L/∂z_i = (1/N) · (softmax(z)_i - 𝟙(i = target))

            In matrix form:
                grad = (probs - one_hot(targets)) / N

            This single line encapsulates:
            - The softmax Jacobian
            - The cross-entropy derivative
            - The mean reduction
            All fused together for numerical stability and simplicity.
            ================================================================
            """
            # Start with softmax probabilities
            grad = probs.copy()                                    # (N, V)

            # Subtract 1 at the target indices (one-hot subtraction)
            grad[np.arange(N), targets_1d] -= 1.0

            # Divide by N for mean reduction
            grad /= N

            # Reshape back to (batch, seq, vocab) and accumulate
            # Multiply by loss.grad (which is 1.0 since loss is the root)
            logits.grad = logits.grad + grad.reshape(
                batch_size, seq_len, vocab_size
            ) * loss.grad

        loss._backward = _backward
        return loss
