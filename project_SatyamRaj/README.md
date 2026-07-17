# Mini GPT-2 and nanoGPT from Scratch

A complete implementation of a mini GPT-2 style and improved nanoGPT-like decoder-only transformer, inspired by Andrej Karpathy's educational content.

## Project Structure

- [config.py](config.py) or [nano_gpt_config.py](nano_gpt_config.py): Hyperparameters/Configuration
- [dataset.py](dataset.py): Data loading and tokenization
- [model.py](model.py) or [nano_gpt_model.py](nano_gpt_model.py): Transformer decoder architecture
- [train.py](train.py) or [nano_gpt_train.py](nano_gpt_train.py): Training loops
- [generate.py](generate.py) or [nano_gpt_generate.py](nano_gpt_generate.py): Text generation

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download Dataset (optional)
The dataset is downloaded automatically when training, but you can also download it manually:

```bash
python download_data.py
```

### Train the model

```bash
python train.py          # Trains the basic GPT-2 model
python nano_gpt_train.py # Trains the improved nanoGPT model
```

### Generate text

```bash
python generate.py          # Generates using the basic GPT-2 model
python nano_gpt_generate.py # Generates using the improved nanoGPT model
```
