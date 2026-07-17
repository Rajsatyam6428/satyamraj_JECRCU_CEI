
import torch
import requests
from config import batch_size, block_size, device

import os
def download_dataset(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    if not os.path.exists("input.txt"):
        print("Downloading dataset...")
        response = requests.get(url)
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        print("Dataset already exists!")

def load_dataset():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, encode, decode

def get_batch(train_data, val_data, split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
