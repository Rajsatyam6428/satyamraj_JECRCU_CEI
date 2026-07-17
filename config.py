
import torch

batch_size = 32
block_size = 64
max_iters = 500
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0
