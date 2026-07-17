
# nanoGPT-like Model Guide (Improved!)

## Files Created:
1. **`nano_gpt_model.py`**: The full GPT model (with all nanoGPT-like features)
2. **`nano_gpt_config.py`**: Hyperparameters
3. **`nano_gpt_train.py`**: Training script
4. **`nano_gpt_generate.py`**: Text generation script
5. **`nano_gpt_test.py`**: Quick test script
6. **`NANO_GPT_ARCHITECTURE.md`**: Architecture breakdown

## What We Improved:
1. **Bigger model**: 6 layers, 6 heads, 384 embedding dim
2. **Proper weight decay separation** for optimizer
3. **Fused AdamW** (if available on CUDA)
4. **Cosine learning rate decay with warmup**
5. **Gradient clipping**
6. **Better initialization** for residual connections
7. **Dropout** for regularization

## How to Use:

### 0. Test the model (optional but recommended):
```bash
python nano_gpt_test.py
```

### 1. Train the model:
```bash
python nano_gpt_train.py
```

### 2. Generate text:
```bash
python nano_gpt_generate.py
```

## Key nanoGPT-like Features:
- Weight tying between token embeddings and language model head
- Flash Attention (if available)
- Pre-normalization transformer blocks
- GELU activation in MLP
- Scaled initialization for residual projections
- Weight decay optimization
- Learning rate scheduling
- Gradient clipping
