
import torch
from config import device
from dataset import load_dataset
from model import GPTLanguageModel

def main():
    print("Loading model...")
    train_data, val_data, vocab_size, encode, decode = load_dataset()
    model = GPTLanguageModel(vocab_size)
    model.load_state_dict(torch.load("model.pt"))
    model = model.to(device)
    model.eval()
    print("Generating English text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
    print("\n" + "-"*50)
    print("Generated English Text:")
    print("-"*50)
    print(generated_text)
    print("-"*50)

if __name__ == "__main__":
    main()
