# MathGPT

A GPT-style transformer model trained to solve arithmetic expressions, built from scratch using curriculum learning.

## Overview

MathGPT is a character-level language model adapted for arithmetic reasoning. It learns to solve mathematical expressions across three levels of complexity — from single-digit operations to parenthesized expressions — achieving **89.30% overall accuracy** on a held-out test set, compared to a random baseline of 5.74%.

## Dataset

The training data consists of 100k arithmetic problems split into three stages:

| Stage | Description | Size |
|-------|-------------|------|
| Stage 1 | Single-digit: `+`, `-`, `×`, `÷` | 40k problems |
| Stage 2 | Double-digit: `+`, `-`, `×`, `÷` | 40k problems |
| Stage 3 | Parenthesized expressions: `(a±b)*c`, `(a±b)/c` | 20k problems |

10% of each stage was held out as an unseen test set. All results are non-negative integers; division produces whole numbers only.

**Vocabulary (19 tokens):** `PAD`, `ANS`, `STOP`, `LPAREN`, `RPAREN`, `PLUS`, `MINUS`, `MUL`, `DIV`, `NUM_0`–`NUM_9`

## Model Architecture

A GPT-style transformer with the following key adaptations:

- **Symbolic Tokenizer** — digits and operators are encoded as semantic tokens (e.g. `NUM_3`, `PLUS`, `LPAREN`) rather than raw characters, encouraging the model to learn arithmetic rules rather than character patterns.
- **Expression-aligned Block Size** — blocks are sized to hold complete arithmetic expressions, with `PAD` tokens filling any remaining space. This ensures no expression is ever split across batches.
- **PAD-masked Loss** — cross-entropy loss ignores `PAD` tokens so gradient updates are driven only by meaningful arithmetic tokens.
- **Argmax Decoding** — replaces softmax sampling with argmax, selecting the single most likely token at each step (appropriate since arithmetic has one correct answer).

### Final Hyperparameters

```
batch_size  = 64
block_size  = 32
max_iters   = 5000
n_embd      = 192
n_head      = 6
n_layer     = 6
dropout     = 0.1
Parameters  = 2.68M
```

## Training Strategy: Curriculum Learning

Instead of training on all data at once, the model is trained progressively:

1. Train on **Stage 1** → save weights
2. Load Stage 1 weights, train on **Stage 2** → save weights
3. Load Stage 2 weights, train on **Stage 3** → final model

This allows the model to first master simple arithmetic before tackling complex expressions.

## Results

| Operation | Accuracy |
|-----------|----------|
| Addition | 98.80% |
| Division | 100.00% |
| Subtraction | 93.44% |
| Parentheses | 83.15% |
| Multiplication | 71.58% |
| **Overall** | **89.30%** |

Random baseline accuracy: **5.74%**

## Sample Outputs

```
(16-4)/6=2
9-3=6
44/44=1
10*74=740
99+43=142
```

## Requirements

- Python 3.x
- PyTorch

## Usage

```bash
# Generate dataset
python dataset.py

# Train the model with curriculum learning
python train.py

# Evaluate on held-out test set
python evaluate.py
```