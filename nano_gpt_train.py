
"""
Training script for our nanoGPT-like model (full nanoGPT features!)
"""
import math
import torch
from nano_gpt_config import (
    model_config, batch_size, max_iters, eval_interval, eval_iters,
    learning_rate, weight_decay, beta1, beta2, grad_clip,
    decay_lr, warmup_iters, lr_decay_iters, min_lr, device
)
from nano_gpt_model import NanoGPT
from dataset import download_dataset, load_dataset, get_batch

torch.manual_seed(1337)

# Download and load data
download_dataset()
train_data, val_data, vocab_size, encode, decode = load_dataset()

# Create model
model = NanoGPT(model_config)
model = model.to(device)

# Optimizer (using model's configure_optimizers method!)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

# Learning rate scheduler (cosine decay with warmup, like real nanoGPT!)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print("Training started!", flush=True)
for iter in range(max_iters):
    # Update learning rate
    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"\nstep {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

    # Print progress every 50 iterations
    if iter % 50 == 0:
        print(f"Step {iter:4d} / {max_iters} | LR: {lr:.6f}", end="\r", flush=True)

    # Sample batch
    xb, yb = get_batch(train_data, val_data, 'train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping (just like nanoGPT!)
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

# Save model
torch.save(model.state_dict(), "nano_gpt_model.pt")
print(f"\nTraining complete! Model saved to nano_gpt_model.pt", flush=True)
