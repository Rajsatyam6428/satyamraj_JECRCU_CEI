
"""
Config for our nanoGPT-like model (improved!)
"""
import torch
from nano_gpt_model import NanoGPTConfig

# Model config (a bit bigger!)
model_config = NanoGPTConfig(
    block_size=256,
    vocab_size=65,  # Shakespeare char-level vocab size
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True
)

# Training config (more like real nanoGPT!)
batch_size = 32
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 200
lr_decay_iters = 10000
min_lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
