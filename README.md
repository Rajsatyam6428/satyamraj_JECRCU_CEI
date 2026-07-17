# Celebal Technologies Assignments

This repository contains the assignments completed as part of my internship/training at **Celebal Technologies**.

## Project: Mini GPT-2 and nanoGPT from Scratch

A complete implementation of a mini GPT-2 style and improved nanoGPT-like decoder-only transformer, inspired by Andrej Karpathy's educational content.

### Project Structure

- [config.py](config.py): Hyperparameters
- [dataset.py](dataset.py): Data loading and tokenization
- [model.py](model.py): Transformer architecture
- [train.py](train.py): Training loop
- [generate.py](generate.py): Text generation
- [nano_gpt_model.py](nano_gpt_model.py): Improved nanoGPT-like architecture (with Flash Attention, Weight Tying, Cosine LR Decay, etc.)
- [nano_gpt_config.py](nano_gpt_config.py): Config for nanoGPT
- [nano_gpt_train.py](nano_gpt_train.py): Training script for nanoGPT
- [nano_gpt_generate.py](nano_gpt_generate.py): Generation script (includes Top-K sampling and Temperature control)

### Getting Started

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Download Dataset (optional)
The dataset is downloaded automatically when training, but you can also download it manually:

```bash
python download_data.py
```

#### Train the model

```bash
python train.py          # Trains the basic GPT-2 model
python nano_gpt_train.py # Trains the improved nanoGPT model
```

#### Generate text

```bash
python generate.py          # Generates using the basic GPT-2 model
python nano_gpt_generate.py # Generates using the improved nanoGPT model
```

### Note

During a Git operation, I encountered an issue while pushing my local repository, which resulted in the loss of the previous Git commit history. To ensure all completed work was available, I recreated the repository and pushed the assignments as a fresh commit.

Although the original commit history is unavailable, the assignment files and solutions have been preserved and uploaded in this repository.

**Maintainer:** Satyam Raj
