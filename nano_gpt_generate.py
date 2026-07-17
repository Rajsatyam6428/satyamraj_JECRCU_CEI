
"""
Generate text with our nanoGPT-like model
"""

import torch
from nano_gpt_config import model_config, device
from nano_gpt_model import NanoGPT
from dataset import load_dataset

def main():
    print("Loading model...")
    _, _, vocab_size, encode, decode = load_dataset()

    model = NanoGPT(model_config)
    model.load_state_dict(torch.load("nano_gpt_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    print("Generating text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=500, temperature=0.6, top_k=10)[0].tolist())
    print("\n" + "-"*50)
    print("Generated Text:")
    print("-"*50)
    print(generated_text)

if __name__ == "__main__":
    main()
