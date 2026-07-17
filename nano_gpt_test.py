
"""
Quick test for our improved nanoGPT-like model
"""
import torch
from nano_gpt_config import model_config, device, learning_rate, weight_decay, beta1, beta2
from nano_gpt_model import NanoGPT
from dataset import load_dataset

def main():
    print("Testing improved nanoGPT-like model...")
    _, _, vocab_size, encode, decode = load_dataset()
    model = NanoGPT(model_config)
    model = model.to(device)
    print("Model created successfully!")

    # Test optimizer setup
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    print("Optimizer configured successfully!")

    # Test a forward pass with dummy input
    dummy_input = torch.randint(0, vocab_size, (2, 32), device=device)  # (batch_size=2, seq_len=32)
    logits, loss = model(dummy_input, dummy_input)
    print(f"Forward pass successful! Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    # Test generation
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=30)
    print(f"Generation test successful: {decode(generated[0].tolist())}")

    print("\nAll tests passed! The model is now much closer to real nanoGPT!")

if __name__ == "__main__":
    main()
