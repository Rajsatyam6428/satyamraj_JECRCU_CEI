
# nanoGPT Architecture Breakdown

## Core Components

### 1. **LayerNorm** (`LayerNorm`)
- Normalizes inputs across features
- Optional bias (nanoGPT allows disabling for speed)

### 2. **Causal Self-Attention** (`CausalSelfAttention`)
- Computes Query, Key, Value in one linear layer for efficiency
- Uses Flash Attention if available (PyTorch ≥ 2.0)
- Applies causal mask (can't look into the future)
- Splits into heads, computes attention, then merges

### 3. **MLP** (`MLP`)
- Two linear layers with 4x expansion
- GELU activation
- Dropout regularization

### 4. **Transformer Block** (`Block`)
- Pre-normalization ("pre-norm"): LayerNorm → Attention → Residual
- Another pre-norm: LayerNorm → MLP → Residual

### 5. **Full GPT Model** (`GPT`)
- Token Embeddings (`wte`)
- Position Embeddings (`wpe`)
- Stack of Transformer Blocks (`h`)
- Final LayerNorm (`ln_f`)
- Language Model Head (`lm_head`)
- Weight tying between `wte` and `lm_head` (saves parameters!)

## Key Optimizations from nanoGPT
- Weight tying (embedding layer = final linear layer)
- Flash Attention
- Fused AdamW (if available)
- Scaled initialization for residual projections
