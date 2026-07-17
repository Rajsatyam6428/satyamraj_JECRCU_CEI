
import torch
from config import device
from dataset import load_dataset
from model import GPTLanguageModel

def main():
    print("Loading model...")
    train_data, val_data, vocab_size, encode, decode = load_dataset()
    model = GPTLanguageModel(vocab_size)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    print("Generating English text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=300)[0].tolist())
    
    with open("generated_english_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
    
    print("Saved generated English text to generated_english_text.txt")
    print("\nHere's the text:")
    print(generated_text)

if __name__ == "__main__":
    main()
