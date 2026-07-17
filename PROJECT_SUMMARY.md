
# Project Summary: Mini GPT & nanoGPT

## 1. Custom Mini GPT Implementation
We built a complete, modular mini GPT-2 style model from scratch!

### Files:
- [config.py](config.py): Hyperparameters
- [dataset.py](dataset.py): Data loading & tokenization
- [model.py](model.py): Transformer decoder
- [train.py](train.py): Training loop
- [generate.py](generate.py): Text generation
- [download_data.py](download_data.py): Manual data download
- [test_model.py](test_model.py): Model test

### Usage:
1. Install dependencies: `pip install -r requirements.txt`
2. Train: `python train.py`
3. Generate text: `python generate.py`

## 2. nanoGPT-like Model (Improved!)
We built a much more capable model with full nanoGPT features!

### Files:
- [nano_gpt_model.py](nano_gpt_model.py): Full nanoGPT-style model
- [nano_gpt_config.py](nano_gpt_config.py): Config
- [nano_gpt_train.py](nano_gpt_train.py): Training script
- [nano_gpt_generate.py](nano_gpt_generate.py): Text generation
- [nano_gpt_test.py](nano_gpt_test.py): Test script
- [NANO_GPT_GUIDE.md](NANO_GPT_GUIDE.md): Guide
- [NANO_GPT_ARCHITECTURE.md](NANO_GPT_ARCHITECTURE.md): Architecture breakdown

### Key Features:
- Weight tying
- Flash Attention
- Cosine LR decay with warmup
- Gradient clipping
- Weight decay
- Fused AdamW
- And more!

### Status:
Training for 10,000 iterations is now running in the background!


## 2. nanoGPT
We cloned Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) repository!

### Status:
- ✅ Cloned repository
- ✅ Prepared Shakespeare character dataset
- 🔄 Training in progress!

### Usage (once training completes):
- Sample text: `cd nanoGPT; python sample.py --out_dir=out-shakespeare-char --device=cpu`
