
import torch
from config import device
from dataset import download_dataset, load_dataset
from model import GPTLanguageModel

print("Testing model compilation...")
download_dataset()
train_data, val_data, vocab_size, encode, decode = load_dataset()
model = GPTLanguageModel(vocab_size).to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters!")
print("Model test passed!")
