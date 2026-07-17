
import torch
from config import max_iters, eval_interval, learning_rate, eval_iters, device
from dataset import download_dataset, load_dataset, get_batch
from model import GPTLanguageModel

torch.manual_seed(1337)

download_dataset()
train_data, val_data, vocab_size, encode, decode = load_dataset()

model = GPTLanguageModel(vocab_size)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if iter % 100 == 0:
        print(f"Training iteration {iter}/{max_iters}...")

    xb, yb = get_batch(train_data, val_data, 'train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")
print("Model saved as model.pt")
