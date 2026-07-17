
# Your Custom Mini GPT: Quick Guide

## Your Model is Ready!

### Files:
- [config.py](config.py): Hyperparameters
- [dataset.py](dataset.py): Data & tokenization
- [model.py](model.py): Transformer model
- [train.py](train.py): Training loop
- [generate.py](generate.py): Generate text
- [generate_and_save.py](generate_and_save.py): Generate + save text to file
- [model.pt](model.pt): Trained model
- [input.txt](input.txt): English Shakespeare dataset

### How to Use:

1. **Train (optional, to get better results)**:
   ```bash
   python train.py
   ```
   (Edit config.py to adjust model size/training length)

2. **Generate English text**:
   ```bash
   python generate_and_save.py
   ```
   This will save text to `generated_english_text.txt`

3. **Or run generate.py directly**:
   ```bash
   python generate.py
   ```

The model uses Shakespeare's works as training data, so it will generate old English-style text!
