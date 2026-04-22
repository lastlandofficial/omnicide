<![CDATA[<div align="center">

# 🔥 OMNICIDE

### A Transformer Built From Absolute Zero

**Pure Python · Pure NumPy · No PyTorch · No TensorFlow · No Shortcuts**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![From Scratch](https://img.shields.io/badge/Built-From%20Scratch-FF6B6B?style=for-the-badge)]()

---

*Every matrix multiply. Every gradient. Every backprop step. Written by hand.*

</div>

---

## 🧠 What Is This?

**Omnicide** is a complete Transformer implementation — autograd engine, attention mechanism, training loop, and text generation — built entirely from scratch using **only Python and NumPy**. No deep learning frameworks. No magic `.backward()` calls you don't understand.

This project answers the question:

> *"What if you had to build GPT from the ground up, understanding every single line?"*

```
                    ┌─────────────────────────────────────────┐
                    │         OMNICIDE ARCHITECTURE           │
                    ├─────────────────────────────────────────┤
                    │                                         │
                    │   Text ──→ Tokenizer ──→ Token IDs      │
                    │                  │                       │
                    │                  ▼                       │
                    │         ┌──────────────┐                │
                    │         │  Embedding   │                │
                    │         │  + PosEnc    │                │
                    │         └──────┬───────┘                │
                    │                │                         │
                    │                ▼                         │
                    │   ┌────────────────────────┐            │
                    │   │  Decoder-Only Block ×N │            │
                    │   │  ┌──────────────────┐  │            │
                    │   │  │ LayerNorm        │  │            │
                    │   │  │ Multi-Head Attn  │  │            │
                    │   │  │ + Residual       │  │            │
                    │   │  ├──────────────────┤  │            │
                    │   │  │ LayerNorm        │  │            │
                    │   │  │ Feed-Forward     │  │            │
                    │   │  │ + Residual       │  │            │
                    │   │  └──────────────────┘  │            │
                    │   └────────────────────────┘            │
                    │                │                         │
                    │                ▼                         │
                    │         ┌──────────────┐                │
                    │         │  LayerNorm   │                │
                    │         │  + Linear    │                │
                    │         │  → Logits    │                │
                    │         └──────┬───────┘                │
                    │                │                         │
                    │                ▼                         │
                    │   Loss ← CrossEntropy(logits, targets)  │
                    │                │                         │
                    │                ▼                         │
                    │          loss.backward()                 │
                    │     (autograd through EVERYTHING)        │
                    │                │                         │
                    │                ▼                         │
                    │       Adam optimizer.step()              │
                    │                                         │
                    └─────────────────────────────────────────┘
```

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/lastland/omnicide.git
cd omnicide

# Only dependency: NumPy
pip install numpy

# Run the training pipeline
python train.py

# Run forward pass tests
python example.py

# Test the scalar autograd engine
python test_engine.py
```

---

## 📁 Project Structure

```
omnicide/
│
├── tensor.py          # 🧬 Tensor Autograd Engine — the heart of backpropagation
│                      #    N-dimensional arrays with automatic gradient tracking
│                      #    Operations: +, -, *, @, /, reshape, transpose
│                      #    Functions: softmax, layer_norm, relu, embedding, masked_fill
│
├── engine.py          # 🔬 Scalar Autograd Engine (micrograd-inspired)
│                      #    Where it all started — Value class for scalar backprop
│
├── math_ops.py        # 🧮 Differentiable Building Blocks
│                      #    Linear (y = xW + b), LayerNorm, Embedding lookup
│
├── attention.py       # 👁️ Multi-Head Attention
│                      #    Scaled Dot-Product Attention: softmax(QKᵀ/√d_k)V
│                      #    Head splitting, masking, concatenation — all differentiable
│
├── layers.py          # 🏗️ Transformer Layers
│                      #    PositionalEncoding (sinusoidal, fixed)
│                      #    PositionwiseFeedForward (expand → ReLU → compress)
│                      #    EncoderBlock, DecoderBlock, DecoderOnlyBlock
│
├── model.py           # 🤖 Complete Transformer Models
│                      #    Transformer (encoder-decoder, Vaswani et al.)
│                      #    DecoderOnlyTransformer (GPT-style, autoregressive)
│                      #    Text generation with temperature sampling
│
├── loss.py            # 📉 Cross-Entropy Loss
│                      #    Fused softmax + NLL for numerical stability
│                      #    Elegant gradient: softmax(logits) - one_hot(target)
│
├── optim.py           # 🚀 Adam Optimizer + Gradient Clipping
│                      #    Adaptive moment estimation (Kingma & Ba, 2014)
│                      #    Bias-corrected first & second moment estimates
│
├── data.py            # 📊 Data Pipeline
│                      #    CharTokenizer — character-level tokenization
│                      #    DataLoader — random batch sampling for next-token prediction
│
├── config.py          # ⚙️ TransformerConfig
│                      #    Centralized hyperparameters (d_model, heads, layers, etc.)
│
├── train.py           # 🏋️ Complete Pretraining Pipeline
│                      #    Tokenization → Model Init → Training Loop → Text Generation
│                      #    Trains on Shakespeare's Coriolanus
│
├── example.py         # 🧪 Forward Pass Tests
│                      #    Smoke tests for both model architectures
│
└── test_engine.py     # ✅ Scalar Autograd Test
│                      #    Verifies gradient computation on the Value engine
```

---

## 🏗️ What We Built (The Ocean We Conquered)

### ✅ Custom Autograd Engine (`tensor.py`)
- **N-dimensional Tensor class** wrapping NumPy arrays with full gradient tracking
- **Reverse-mode automatic differentiation** (backpropagation) via computation graph
- **Topological sort** for correct gradient flow ordering
- **Operator overloads**: `+`, `-`, `*`, `/`, `@` (matmul) — all differentiable
- **Shape operations**: `reshape`, `transpose` with correct inverse-permutation gradients
- **Broadcast-aware gradients**: `_unbroadcast()` handles dimension reduction for bias gradients
- **Standalone differentiable functions**: `softmax`, `layer_norm`, `relu`, `embedding`, `masked_fill`

### ✅ Scalar Autograd Engine (`engine.py`)
- **micrograd-inspired** `Value` class for scalar computation graphs
- Where the journey started — proving the chain rule works before scaling to tensors

### ✅ Multi-Head Attention (`attention.py`)
- Full **Scaled Dot-Product Attention**: `softmax(QKᵀ / √d_k) · V`
- **Multi-head splitting**: reshape → transpose → independent attention per head → concatenate
- **Causal masking** for autoregressive generation (masked fill with `-1e9`)
- 4 learned linear projections: `W_Q`, `W_K`, `W_V`, `W_O`

### ✅ Transformer Layers (`layers.py`)
- **Sinusoidal Positional Encoding** (fixed, not learned)
- **Position-wise Feed-Forward Network** (expand → ReLU → compress)
- **Pre-LayerNorm architecture** (more stable than Post-LN)
- **Residual connections** throughout (gradient highways)
- Three block types: `EncoderBlock`, `DecoderBlock`, `DecoderOnlyBlock`

### ✅ Complete Model Architectures (`model.py`)
- **Encoder-Decoder Transformer** (Vaswani et al., 2017) — for seq-to-seq tasks
- **Decoder-Only Transformer** (GPT-style) — for autoregressive language modeling
- **Text generation** with temperature-controlled sampling

### ✅ Cross-Entropy Loss (`loss.py`)
- **Fused softmax + cross-entropy** for numerical stability
- Log-softmax with max-subtraction trick (prevents `exp()` overflow)
- The elegant gradient: `∂L/∂z = (softmax(z) - one_hot(target)) / N`

### ✅ Adam Optimizer (`optim.py`)
- **Full Adam implementation** (Kingma & Ba, 2014)
- First moment (gradient momentum) and second moment (adaptive learning rate)
- **Bias correction** for accurate early-training estimates
- **Gradient clipping by global norm** — prevents exploding gradients

### ✅ Data Pipeline (`data.py`)
- **Character-level tokenizer** — builds vocab from raw text
- **Batch data loader** — random sampling with next-token prediction setup
- Input/target shifting (`Y = X shifted right by 1`)

### ✅ Training Pipeline (`train.py`)
- **End-to-end pretraining** on Shakespeare text
- Complete loop: forward → loss → zero_grad → backward → clip → step
- Training metrics logging (loss, grad norm, timing)
- **Text generation** from a prompt after training

---

## 🌊 The Ocean We Haven't Crossed (Roadmap)

### 🔴 Critical — Core Architecture Gaps

| Feature | Description | Difficulty |
|---|---|---|
| **Dropout** | Regularization during training — randomly zero out neurons to prevent overfitting | 🟡 Medium |
| **Weight Tying** | Share embedding weights with output projection (`embedding.W = fc_out.W.T`) to reduce params | 🟢 Easy |
| **Learned Positional Embeddings** | Replace fixed sinusoidal with trainable position vectors (GPT-2 style) | 🟢 Easy |
| **KV-Cache** | Cache key/value tensors during generation to avoid redundant computation | 🟡 Medium |
| **Attention Dropout** | Drop attention weights during training for regularization | 🟢 Easy |

### 🟠 Important — Training & Optimization

| Feature | Description | Difficulty |
|---|---|---|
| **Learning Rate Scheduling** | Warmup + cosine decay (critical for Transformer convergence) | 🟡 Medium |
| **BPE Tokenizer** | Byte-Pair Encoding for real subword tokenization (32K-100K vocab) | 🔴 Hard |
| **Mixed Precision (Simulated)** | float16/bfloat16 simulation for understanding quantization effects | 🔴 Hard |
| **Gradient Accumulation** | Simulate larger batch sizes by accumulating gradients over micro-batches | 🟢 Easy |
| **Weight Decay (AdamW)** | Decoupled weight decay regularization (Loshchilov & Hutter, 2017) | 🟢 Easy |
| **Model Checkpointing** | Save/load model weights to/from disk (pickle or custom format) | 🟡 Medium |
| **Training Loss Curves** | Plot loss over time using matplotlib | 🟢 Easy |

### 🟡 Advanced — Modern Architecture Features

| Feature | Description | Difficulty |
|---|---|---|
| **RoPE (Rotary Position Embeddings)** | Used by LLaMA, Mistral — encodes relative position in attention | 🔴 Hard |
| **GQA (Grouped Query Attention)** | LLaMA 2 / Mistral optimization — fewer KV heads than Q heads | 🔴 Hard |
| **SwiGLU Activation** | Replace ReLU in FFN with SwiGLU (used by LLaMA, PaLM) | 🟡 Medium |
| **RMSNorm** | Simpler & faster alternative to LayerNorm (used by LLaMA) | 🟡 Medium |
| **Flash Attention (Algorithmic)** | Memory-efficient attention via tiling (concept demonstration, not GPU) | 🔴 Hard |
| **Sliding Window Attention** | Limit attention span for efficiency (Mistral-style) | 🟡 Medium |
| **ALiBi (Attention with Linear Biases)** | Extrapolates to longer sequences without position embeddings | 🟡 Medium |

### 🔵 Frontier — Post-Training & Applications

| Feature | Description | Difficulty |
|---|---|---|
| **LoRA (Low-Rank Adaptation)** | Efficient fine-tuning by injecting trainable low-rank matrices | 🔴 Hard |
| **RLHF Pipeline** | Reward model + PPO for alignment (conceptual implementation) | 🔴 Hard |
| **Beam Search** | Better-than-greedy decoding with beam width parameter | 🟡 Medium |
| **Top-k / Top-p Sampling** | Nucleus sampling for more coherent text generation | 🟢 Easy |
| **Perplexity Evaluation** | Standard metric for language model quality | 🟢 Easy |
| **Attention Visualization** | Render attention weight matrices as heatmaps | 🟡 Medium |
| **Multi-GPU Simulation** | Demonstrate data parallelism concepts (split batches across "devices") | 🔴 Hard |

### ⚪ Infrastructure & DevOps

| Feature | Description | Difficulty |
|---|---|---|
| **Unit Test Suite** | Comprehensive tests with `pytest` for every module | 🟡 Medium |
| **Gradient Checking** | Numerical gradient verification (finite differences) | 🟡 Medium |
| **CI/CD Pipeline** | GitHub Actions for automated testing | 🟢 Easy |
| **Type Hints** | Full type annotations across the codebase | 🟢 Easy |
| **Documentation Site** | Sphinx or MkDocs with mathematical explanations | 🟡 Medium |
| **Interactive Notebook** | Jupyter notebook walkthrough of the entire architecture | 🟡 Medium |
| **Benchmarking Suite** | Performance profiling and comparison with PyTorch | 🟡 Medium |

---

## 🧮 The Math Under the Hood

Every line of code in Omnicide implements real mathematics. Here are the key equations:

### Attention
```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

### Backprop through MatMul
```
If C = A @ B:
    ∂L/∂A = ∂L/∂C @ Bᵀ
    ∂L/∂B = Aᵀ @ ∂L/∂C
```

### Softmax Backward
```
∂L/∂x = s ⊙ (∂L/∂s − ⟨∂L/∂s, s⟩)
```

### Layer Normalization Backward
```
∂L/∂x = (1/Nσ) · [N · dx̂ − Σ(dx̂) − x̂ · Σ(dx̂ · x̂)]
```

### Cross-Entropy Gradient (The Beautiful One)
```
∂L/∂z = softmax(z) − one_hot(target)
```

### Adam Update Rule
```
m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
```

---

## 📐 Model Configurations

### Default (what we train with):
| Hyperparameter | Value | GPT-2 Equivalent |
|---|---|---|
| `d_model` | 64 | 768 |
| `num_heads` | 4 | 12 |
| `num_layers` | 2 | 12 |
| `d_ff` | 256 | 3072 |
| `max_seq_length` | 64 | 1024 |
| `vocab_size` | ~55 (char-level) | 50,257 (BPE) |
| **Total Params** | **~60K** | **~117M** |

---

## 🎯 Design Philosophy

1. **Zero Dependencies** (beyond NumPy) — every algorithm is implemented from scratch
2. **Exhaustive Documentation** — every function has docstrings explaining the math
3. **Educational First** — code is written to teach, not to be fast
4. **Real Gradients** — no `torch.autograd`, no finite differences — real chain rule
5. **Both Architectures** — encoder-decoder (original) AND decoder-only (GPT-style)

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2014
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016
- [micrograd](https://github.com/karpathy/micrograd) — Andrej Karpathy (inspiration for `engine.py`)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019

---

<div align="center">

**Built with 🧠 and NumPy. No frameworks were harmed in the making of this project.**

*From scalar autograd to full Transformer pretraining — every gradient earned.*

</div>
]]>
