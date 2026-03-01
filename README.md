# 🧠 IkokoGPT — Optimised PyTorch GPT

> Made by Ikoko

A GPT built from scratch **optimised like the real thing**: Flash Attention, Mixed Precision,
torch.compile, cosine LR schedule, correct AdamW weight decay, and token-by-token streaming generation.

---

## Requirements

- **Python 3.11** (Python 3.12+ is not fully compatible with PyTorch — use 3.11)
- **PyTorch 2.0+** (GPU recommended — CUDA 11.8 or 12.1)

## Installation

```bash
# For RTX 4070 / 4080 / 4090 (CUDA 12.1) — recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For older GPUs (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch pyyaml

# Other dependencies
pip install pyyaml
```

> ⚠️ **Python version:** Use Python **3.11**. Python 3.12+ and 3.14 are not fully compatible with PyTorch and may cause issues with `torch.compile` and CUDA extensions.

> 💡 **RTX 4070 tip:** Use CUDA 12.1 (`cu121`). A plain `pip install torch` may install a CPU-only version and not detect your GPU.

---

## Quick Start

```bash
# Main menu (recommended)
python main.py

# Or directly:
python train.py   # train the model
python chat.py    # start chatting
```

---

## Project Structure

```
IkokoGPT/
├── main.py         ← 🎮  Main entry point (interactive menu)
├── config.yaml     ← ⚙️  Configuration (model, training, chat)
├── model.py        ← 🔑  Optimised GPT (Flash Attention, Tied Embeddings...)
├── tokenizer.py    ← 📖  CharTokenizer + lightweight BPE
├── train.py        ← 🚀  Training (Mixed Precision, cosine LR, grad accum...)
├── chat.py         ← 💬  Interactive chat with token streaming
├── data/
│   └── train.txt   ← 📂  Training data
└── checkpoints/
    └── {project}/  ← 💾  Saved models (one folder per project)
        ├── ckpt.pt
        ├── best.pt
        └── tokenizer.json
```

---

## Optimisations

### Model (`model.py`)

| Optimisation | Effect |
|---|---|
| **Flash Attention** | 2-4× faster, 10× less VRAM for attention |
| **Tied Embeddings** | Shares tok_emb ↔ lm_head → fewer parameters |
| **Weight init (GPT-2 style)** | `std=0.02`, scaled residuals → faster convergence |
| **No bias** | Linear layers without bias → slightly faster |
| **Pre-norm** | LayerNorm before op → more stable gradients |

### Training (`train.py`)

| Optimisation | Effect |
|---|---|
| **Mixed Precision** | bfloat16/float16 → 2× faster on GPU |
| **torch.compile** | JIT compilation → 30-50% faster (PyTorch 2.0+) |
| **Gradient Accumulation** | Large effective batch on small VRAM |
| **Cosine LR + Warmup** | Optimal schedule for Transformers |
| **Fused AdamW** | Fused CUDA kernel for the optimizer |
| **Gradient Clipping** | Prevents gradient explosions |
| **cudnn.benchmark** | Auto-tunes CUDA kernels |
| **pin_memory** | Non-blocking DataLoader |
| **Correct AdamW** | Weight decay only on matrices (not biases/norms) |

---

## Training Options

```bash
# More iterations (better results)
python train.py --iters 10000

# Resume an interrupted training
python train.py --resume

# torch.compile (PyTorch 2.0+ required, +30% speed)
python train.py --compile

# Train on your own data
python train.py --data my_data.txt

# Combine options
python train.py --data my_data.txt --iters 8000 --compile --resume
```

---

## Chat Options

```bash
# Load the best checkpoint
python chat.py --best

# Less creative (more precise)
python chat.py --temp 0.5

# More creative
python chat.py --temp 1.2

# Display all at once (no streaming)
python chat.py --no-stream
```

### In-chat Commands

```
/temp 0.8    → temperature
/topk 50     → top-k sampling (diversity)
/topp 0.9    → nucleus sampling
/len 300     → generation length
/rep 1.1     → repetition penalty
/info        → model info
/reset       → clear history
/quit        → quit
```

---

## Training on Your Own Data

Put any text file in `data/` and point to it in `config.yaml`:

```yaml
data:
  train_file : "data/my_data.txt"
  user_prefix: "You"
  bot_prefix : "GPT"
```

**General rule:** more data = better model.
- < 50k tokens → fast overfitting, keep the model small
- 50k – 500k tokens → sweet spot for a personal model
- > 500k tokens → increase `d_model` and `n_layers` in `config.yaml`

---

## ⚙️ Configuration — `config.yaml`

All configuration is done in `config.yaml`, **no code changes needed**:

```yaml
project: "english_bot"   # each project has its own checkpoint folder

data:
  train_file : "data/train.txt"
  user_prefix: "You"
  bot_prefix : "GPT"

model:
  d_model : 256   # embedding dimension
  n_heads : 4     # attention heads
  n_layers: 4     # transformer blocks
  seq_len : 128   # context length

train:
  batch_size    : 32
  max_iters     : 20000
  learning_rate : 3e-4

chat:
  temperature : 0.7
  max_new     : 120
```

### Recommended GPU Presets

| GPU | d_model | n_heads | n_layers | seq_len | batch_size |
|-----|---------|---------|----------|---------|------------|
| GTX 1060 / CPU (4-6 GB) | 256 | 4 | 4 | 128 | 16 |
| RTX 3060 / 2070 (8 GB)  | 384 | 6 | 6 | 128 | 32 |
| **RTX 4070 / 3080 (12 GB)** | **512** | **8** | **8** | **256** | **64** |
| RTX 4090 / A100 (24 GB+) | 768 | 12 | 12 | 512 | 128 |

---

## Understanding Training Logs

```
[███████████████       ] 2500/5000  loss=1.8234  val=1.9102↓  lr=3.0e-04  tok/s=45000  ETA=2.1min
```

- **loss** : Cross-entropy on training data (should decrease)
- **val** : Loss on validation data (↓ = improvement, ↑ = overfitting)
- **tok/s** : Tokens processed per second
- **ETA** : Estimated time remaining

Typical loss progression:
- Start : ~4.5 (random)
- After 1000 steps : ~2.5 (structure emerging)
- After 5000 steps : ~1.5 (credible generation)
- After 10000+ steps : ~1.0 (very good)
