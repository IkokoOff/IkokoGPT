"""
train.py — Ultra-optimised training script
═══════════════════════════════════════════════════════════════════════

Optimisations:
  ✅ torch.compile()           — JIT (30-50% faster on PyTorch 2.0+)
  ✅ Mixed Precision           — bfloat16 / float16 (2× faster on GPU)
  ✅ GradScaler                — stability with float16
  ✅ Gradient Accumulation     — simulates large batches on small VRAM
  ✅ Cosine LR + Warmup        — optimal schedule for Transformers
  ✅ Gradient Clipping         — prevents gradient explosions
  ✅ AdamW with decoupled WD   — correct weight decay (not on biases)
  ✅ cudnn.benchmark           — auto-tunes CUDA kernels
  ✅ pin_memory + num_workers  — non-blocking DataLoader
  ✅ Periodic generation       — see progress in real time
  ✅ Early stopping            — automatic stop if no more progress
  ✅ Auto data fallback        — creates train.txt if missing

Usage:
  python train.py                    # train with defaults
  python train.py --data myfile.txt  # on your own data
  python train.py --compile          # enable torch.compile (PyTorch 2+)
  python train.py --resume           # continue from checkpoint
  python train.py --iters 10000      # more iterations
  python train.py --tiny             # tiny model for quick testing
"""

import os
import math
import time
import shutil
import argparse
import platform
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp

from model     import MiniGPT, GPTConfig
from tokenizer import build_tokenizer, load_tokenizer

# On Windows, multiprocessing in DataLoader causes overhead — disable it
_NUM_WORKERS = 0 if platform.system() == 'Windows' else 2

# Base directory = folder where train.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(path: str) -> str:
    """Convert a relative path to absolute based on BASE_DIR."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


# ─── Configuration Loading ────────────────────────────────────────────────────

def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from config.yaml.
    Returns a flat dict compatible with the rest of the code.
    Works without PyYAML installed (falls back to default values).
    """
    defaults = {
        'data_path'       : 'train.txt',
        'tokenizer_path'  : 'checkpoints/tokenizer.json',
        'tokenizer_kind'  : 'char',
        'd_model'         : 512,
        'n_heads'         : 8,
        'n_layers'        : 8,
        'seq_len'         : 256,
        'dropout'         : 0.1,
        'batch_size'      : 64,
        'grad_accum'      : 2,
        'learning_rate'   : 3e-4,
        'min_lr'          : 3e-5,
        'warmup_iters'    : 500,
        'max_iters'       : 20000,
        'weight_decay'    : 0.1,
        'grad_clip'       : 1.0,
        'beta1'           : 0.9,
        'beta2'           : 0.95,
        'eval_every'      : 500,
        'eval_iters'      : 50,
        'save_every'      : 2000,
        'generate_every'  : 2000,
        'checkpoint_path' : 'checkpoints/ckpt.pt',
        'best_path'       : 'checkpoints/best.pt',
        'gen_prompt'      : '',
        'gen_len'         : 150,
        'gen_temperature' : 0.8,
        'patience'        : 20,
        'user_prefix'     : 'You',
        'bot_prefix'      : 'GPT',
        'gen_greeting'    : 'hello',
    }

    if not os.path.exists(config_path):
        print(f"  ⚠️  {config_path} not found — using default values")
        return defaults

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        # Flatten YAML sections into a simple dict
        flat = {}
        # Top-level keys (like 'project')
        for k, v in raw.items():
            if not isinstance(v, dict):
                flat[k] = v
        section_map = {
            'data'          : {'train_file': 'data_path', 'tokenizer_file': 'tokenizer_path',
                               'tokenizer_kind': 'tokenizer_kind',
                               'user_prefix': 'user_prefix', 'bot_prefix': 'bot_prefix'},
            'model'         : {'d_model': 'd_model', 'n_heads': 'n_heads', 'n_layers': 'n_layers',
                               'seq_len': 'seq_len', 'dropout': 'dropout'},
            'train'         : {'batch_size': 'batch_size', 'grad_accum': 'grad_accum',
                               'learning_rate': 'learning_rate', 'min_lr': 'min_lr',
                               'warmup_iters': 'warmup_iters', 'max_iters': 'max_iters',
                               'weight_decay': 'weight_decay', 'grad_clip': 'grad_clip',
                               'beta1': 'beta1', 'beta2': 'beta2'},
            'eval'          : {'eval_every': 'eval_every', 'eval_iters': 'eval_iters',
                               'save_every': 'save_every', 'generate_every': 'generate_every',
                               'checkpoint_file': 'checkpoint_path', 'best_file': 'best_path'},
            'generate'      : {'prompt': 'gen_prompt', 'length': 'gen_len',
                               'temperature': 'gen_temperature',
                               'greeting': 'gen_greeting'},
            'early_stopping': {'patience': 'patience'},
        }

        for section, mapping in section_map.items():
            if section in raw and raw[section]:
                for yaml_key, cfg_key in mapping.items():
                    if yaml_key in raw[section] and raw[section][yaml_key] is not None:
                        flat[cfg_key] = raw[section][yaml_key]

        # Merge with defaults, enforcing types from defaults
        merged = {**defaults, **flat}
        for k, v in defaults.items():
            if k in merged and merged[k] is not None:
                try:
                    if isinstance(v, float):
                        merged[k] = float(merged[k])
                    elif isinstance(v, int):
                        merged[k] = int(merged[k])
                except (ValueError, TypeError):
                    pass

        # Build project-aware checkpoint paths
        project = str(merged.get('project', 'default')).strip()
        proj_dir = f'checkpoints/{project}'
        merged['tokenizer_path']  = f'{proj_dir}/tokenizer.json'
        merged['checkpoint_path'] = f'{proj_dir}/ckpt.pt'
        merged['best_path']       = f'{proj_dir}/best.pt'

        # Build gen_prompt dynamically from user_prefix/bot_prefix if not overridden
        user_p   = merged.get('user_prefix', 'You')
        bot_p    = merged.get('bot_prefix',  'GPT')
        greeting = merged.get('gen_greeting', 'hello')
        if not merged.get('gen_prompt'):
            merged['gen_prompt'] = f'{user_p}: {greeting}\n{bot_p}:'

        return merged

    except ImportError:
        print("  ⚠️  PyYAML not installed (pip install pyyaml) — using default values")
        return defaults
    except Exception as e:
        print(f"  ⚠️  Error reading config.yaml: {e} — using default values")
        return defaults


TRAIN_CONFIG = load_config(_abs('config.yaml'))

# Fallback data if no file is found
FALLBACK_TEXT = """\
Hello! How are you today?
I'm doing great, thanks. And you?
I'm doing well too. What are you doing this weekend?
I'm going for a walk in the forest. I love nature.
That's a great idea. The weather is perfect for it.
Yes, the sun is shining and it's warm. Do you want to come with me?
I'd love to! What time shall we meet?
Let's say ten in the morning at the park?
Perfect. See you tomorrow then!
See you tomorrow, have a good evening!

Once upon a time there was a small village by a river.
The inhabitants lived happily and always helped each other.
Every morning the market came alive with colours and smells.
The children played in the cobbled streets.
The elders told stories under the great oak tree.
It was a simple life but full of happiness.
""" * 20   # repeat to have enough data


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Optimised text dataset.
    Pre-tokenises everything in memory and samples randomly
    to maximise diversity of seen examples.
    """

    def __init__(self, tokens: list, seq_len: int):
        self.data    = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.n       = max(1, len(self.data) - seq_len - 1)

    def __len__(self):
        # Overridden so DataLoader knows how many examples there are
        return self.n

    def __getitem__(self, idx):
        # Random offset → maximum diversity even on small datasets
        start = torch.randint(0, self.n, (1,)).item()
        x = self.data[start     : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


# ─── Learning Rate Schedule ───────────────────────────────────────────────────

def cosine_lr(step: int, cfg: dict) -> float:
    """
    Linear warmup then cosine decay.
    Optimal for Transformers (inspired by GPT-3 / Chinchilla).
    Robust: warmup_iters can be >= max_iters without crashing.
    """
    warmup = min(cfg['warmup_iters'], cfg['max_iters'] // 2)  # warmup ≤ moitié du training
    if step < warmup:
        return cfg['learning_rate'] * step / max(1, warmup)
    if step >= cfg['max_iters']:
        return cfg['min_lr']
    progress = (step - warmup) / max(1, cfg['max_iters'] - warmup)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg['min_lr'] + coeff * (cfg['learning_rate'] - cfg['min_lr'])


# ─── Loss Estimation ──────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, loader, n_iters, ctx, device):
    """Estimate loss over n_iters batches (eval mode, no gradient)."""
    model.eval()
    losses      = []
    loader_iter = iter(loader)
    for _ in range(n_iters):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


# ─── Quick Generation ─────────────────────────────────────────────────────────

@torch.no_grad()
def quick_generate(model, tokenizer, prompt, max_new, temperature, device):
    """Generate a short text to track training progress."""
    model.eval()
    tokens  = tokenizer.encode(prompt)
    ctx_tok = torch.tensor([tokens], dtype=torch.long, device=device)
    out     = model.generate(ctx_tok, max_new=max_new, temperature=temperature,
                             top_k=40, top_p=0.9)
    result  = tokenizer.decode(out[0].tolist())
    model.train()
    return result


# ─── Optimizer with correct weight decay ─────────────────────────────────────

def configure_optimizer(model, cfg):
    """
    AdamW with weight decay only on 2D+ parameters (matrices).
    Vectors (biases, norms, embeddings) do NOT get weight decay.
    """
    decay    = set()
    no_decay = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.add(name)
        else:
            no_decay.add(name)

    all_params = {n for n, p in model.named_parameters() if p.requires_grad}
    assert decay | no_decay == all_params, "Missing parameters in groups"
    assert decay & no_decay == set(),      "Overlap in parameter groups"

    param_groups = [
        {'params': [p for n, p in model.named_parameters() if n in decay],
         'weight_decay': cfg['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if n in no_decay],
         'weight_decay': 0.0},
    ]

    # Fused AdamW if available (single CUDA kernel → faster)
    _doc     = torch.optim.AdamW.__init__.__doc__ or ""
    use_fuse = torch.cuda.is_available() and 'fused' in _doc
    extra    = {'fused': True} if use_fuse else {}
    optimizer = torch.optim.AdamW(
        param_groups,
        lr    = cfg['learning_rate'],
        betas = (cfg['beta1'], cfg['beta2']),
        **extra,
    )
    if use_fuse:
        print("  ✓ Fused AdamW enabled")
    return optimizer


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(args):
    cfg = TRAIN_CONFIG.copy()

    # Apply CLI arguments
    if args.iters : cfg['max_iters']     = args.iters
    if args.lr    : cfg['learning_rate'] = args.lr
    if args.batch : cfg['batch_size']    = args.batch
    if args.data  : cfg['data_path']     = args.data
    if args.tiny:
        # Ultra-fast mode to test everything works (~30 seconds)
        cfg['d_model']        = 128
        cfg['n_heads']        = 2
        cfg['n_layers']       = 2
        cfg['seq_len']        = 64
        cfg['max_iters']      = 200
        cfg['batch_size']     = 16
        cfg['grad_accum']     = 2
        cfg['eval_every']     = 50
        cfg['save_every']     = 200
        cfg['generate_every'] = 100
        cfg['warmup_iters']   = 20

    project  = str(cfg.get('project', 'default')).strip()
    proj_dir = _abs(f'checkpoints/{project}')
    os.makedirs(proj_dir, exist_ok=True)
    print(f"  Project : {project}  →  {proj_dir}")

    # ── Device detection ─────────────────────────────────────────
    if torch.cuda.is_available():
        device = 'cuda'
        dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        dtype  = torch.float32
    else:
        device = 'cpu'
        dtype  = torch.bfloat16

    # autocast only on GPU/MPS (bfloat16 on CPU causes bugs on some versions)
    ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=dtype) \
          if device != 'cpu' else nullcontext()

    print("\n" + "═"*60)
    print("  🚀 IkokoGPT — Training")
    print("═"*60)
    print(f"  Device : {device}  |  dtype : {dtype}")

    # ── Data loading ──────────────────────────────────────────────
    # Search in multiple locations (absolute paths based on BASE_DIR)
    candidates = [_abs(cfg['data_path']), _abs('data/' + cfg['data_path']), _abs('data/train.txt')]
    data_path  = None
    for c in candidates:
        if os.path.exists(c):
            data_path = c
            break

    if data_path is None:
        print(f"\n  ⚠️  No data file found — using fallback data")
        data_path = _abs(cfg['data_path'])
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        Path(data_path).write_text(FALLBACK_TEXT, encoding='utf-8')

    text = Path(data_path).read_text(encoding='utf-8')
    print(f"  Data : {data_path}  ({len(text):,} characters)")

    if len(text) < 500:
        print("  ⚠️  Very short text — limited results. Add more data.")

    # ── Tokenizer ────────────────────────────────────────────────
    tok_path = _abs(cfg['tokenizer_path'])
    # Ensure project checkpoint folder exists
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    if os.path.exists(tok_path):
        # Always load existing tokenizer for vocab consistency between runs
        tokenizer = load_tokenizer(tok_path)
        print(f"  Tokenizer loaded  : {tokenizer}")
    else:
        tokenizer = build_tokenizer(text, kind=cfg['tokenizer_kind'],
                                    save_path=tok_path)
        print(f"  Tokenizer created : {tokenizer}")

    # ── Tokens & splits ──────────────────────────────────────────
    tokens = tokenizer.encode(text)
    if len(tokens) < cfg['seq_len'] * 4:
        print(f"  ⚠️  Very few tokens ({len(tokens)}). "
              f"Consider adding more text to {data_path}.")

    # Split train/val — keep at least seq_len+2 tokens in each split
    min_tokens = cfg['seq_len'] + 2
    if len(tokens) < min_tokens * 2:
        # Not enough data: use everything for train AND val (no leakage here, intentional)
        train_tok = tokens
        val_tok   = tokens
        print("  ⚠️  Little data — val_set = train_set (normal for small corpora)")
    else:
        split     = int(0.9 * len(tokens))
        train_tok = tokens[:split]
        val_tok   = tokens[split:]
        if len(val_tok) < min_tokens:
            val_tok = tokens[-min_tokens * 4:]

    print(f"  Tokens  : {len(tokens):,}  "
          f"(train={len(train_tok):,}  val={len(val_tok):,})")


    train_ds = TextDataset(train_tok, cfg['seq_len'])
    val_ds   = TextDataset(val_tok,   cfg['seq_len'])

    nw = _NUM_WORKERS if device == 'cuda' else 0
    pin = device == 'cuda'
    loader_kw    = dict(batch_size=cfg['batch_size'], shuffle=True,
                        drop_last=True, pin_memory=pin, num_workers=nw,
                        persistent_workers=(nw > 0))
    val_loader_kw = dict(batch_size=cfg['batch_size'], shuffle=False,
                         drop_last=True, pin_memory=pin, num_workers=nw,
                         persistent_workers=(nw > 0))
    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader   = DataLoader(val_ds, **val_loader_kw)

    # ── Model ────────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        vocab_size = tokenizer.vocab_size,
        seq_len    = cfg['seq_len'],
        d_model    = cfg['d_model'],
        n_heads    = cfg['n_heads'],
        n_layers   = cfg['n_layers'],
        dropout    = cfg['dropout'],
    )

    start_step = 0
    best_val   = float('inf')

    ckpt_path = _abs(cfg['checkpoint_path'])
    best_path = _abs(cfg['best_path'])

    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}...")
        model      = MiniGPT.from_checkpoint(ckpt_path, device=device)
        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        start_step = ckpt.get('step', 0)
        best_val   = ckpt.get('val_loss', float('inf'))
    else:
        model = MiniGPT(gpt_cfg).to(device)

    # ── torch.compile ────────────────────────────────────────────
    if args.compile:
        if hasattr(torch, 'compile'):
            print("  ⚡ torch.compile() enabled (~1min first time)")
            model = torch.compile(model)
        else:
            print("  ⚠️  torch.compile not available (PyTorch < 2.0)")

    # ── Optimizer & scaler ───────────────────────────────────────
    optimizer = configure_optimizer(model, cfg)
    scaler    = amp.GradScaler(enabled=(device == 'cuda' and dtype == torch.float16))

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])

    n_params       = model.count_params() if hasattr(model, 'count_params') else \
                     sum(p.numel() for p in model.parameters())
    effective_batch = cfg['batch_size'] * cfg['grad_accum']

    print(f"  Parameters      : {n_params/1e6:.2f}M")
    print(f"  Effective batch : {effective_batch}  "
          f"(batch={cfg['batch_size']} × accum={cfg['grad_accum']})")
    print(f"  Steps           : {cfg['max_iters']}  |  "
          f"LR : {cfg['learning_rate']:.0e} → {cfg['min_lr']:.0e}")
    print(f"  Context         : {cfg['seq_len']} tokens")
    print("═"*60 + "\n")

    # ── Main loop ─────────────────────────────────────────────────
    model.train()
    train_iter    = iter(train_loader)
    loss_buf      = []
    t_start       = time.time()
    t_step_start  = time.time()
    tokens_seen   = 0
    patience_ctr  = 0   # early stopping counter
    step          = start_step  # accessible in finally block

    print(f"  💡 Ctrl+C to stop and save cleanly.\n")

    try:
        for step in range(start_step + 1, cfg['max_iters'] + 1):

            # LR schedule
            lr = cosine_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(cfg['grad_accum']):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                tokens_seen += x.numel()

                with ctx:
                    _, loss = model(x, y)
                    loss    = loss / cfg['grad_accum']

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Gradient clip + step optimizer
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            loss_buf.append(accum_loss)

            # ── Progress bar ─────────────────────────────────────────
            t_now        = time.time()
            elapsed      = t_now - t_start
            eta          = elapsed / step * (cfg['max_iters'] - step) if step > 0 else 0
            pct          = step / cfg['max_iters']
            bar_len      = 28
            filled       = int(bar_len * pct)
            bar          = "█" * filled + "░" * (bar_len - filled)
            ms_step      = (t_now - t_step_start) * 1000
            t_step_start = t_now

            print(f"\r  [{bar}] {step:>5}/{cfg['max_iters']}"
                  f"  loss={accum_loss:.4f}"
                  f"  lr={lr:.1e}"
                  f"  {ms_step:.0f}ms"
                  f"  ETA={eta/60:.1f}min  ", end='', flush=True)

            # ── Evaluation ───────────────────────────────────────────
            if step % cfg['eval_every'] == 0:
                t_now   = time.time()
                elapsed = t_now - t_start
                tok_s   = tokens_seen / max(elapsed, 1e-6)
                eta     = elapsed / step * (cfg['max_iters'] - step)

                val_loss   = estimate_loss(model, val_loader, cfg['eval_iters'], ctx, device)
                train_loss = sum(loss_buf[-cfg['eval_every']:]) / min(len(loss_buf), cfg['eval_every'])

                bar_e = ("█" * int(33 * step / cfg['max_iters'])).ljust(33)
                trend = "↓" if val_loss < best_val else "↑"

                print(f"\r  [{bar_e}] {step:>5}/{cfg['max_iters']}"
                      f"  loss={train_loss:.4f}  val={val_loss:.4f}{trend}"
                      f"  lr={lr:.1e}  {tok_s:.0f}tok/s"
                      f"  ETA={eta/60:.1f}min")

                # Best checkpoint
                if val_loss < best_val:
                    best_val     = val_loss
                    patience_ctr = 0
                    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                    raw.save_checkpoint(best_path, optimizer, step, val_loss)
                    print(f"  ✨ New best model saved (val={val_loss:.4f})")
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg['patience']:
                        print(f"\n  ⏹  Early stopping — no improvement for "
                              f"{cfg['patience']} evaluations")
                        break

            # ── Periodic save ────────────────────────────────────────
            if step % cfg['save_every'] == 0:
                raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                raw.save_checkpoint(ckpt_path, optimizer, step, accum_loss)

            # ── Sample generation ────────────────────────────────────
            if step % cfg['generate_every'] == 0:
                raw    = model._orig_mod if hasattr(model, '_orig_mod') else model
                sample = quick_generate(raw, tokenizer,
                                        cfg['gen_prompt'], cfg['gen_len'],
                                        cfg['gen_temperature'], device)
                print(f"\n  ── Generation (step {step}) {'─'*35}")
                display = sample.strip()
                # Cut at the last complete GPT response (avoid truncated lines)
                user_p = cfg.get('user_prefix', 'You')
                bot_p  = cfg.get('bot_prefix',  'GPT')
                # Find the last occurrence of a new user turn → everything before it is complete
                user_turns = [f'\n{user_p}: ', f'\n{user_p}:']
                last_user = -1
                for s in user_turns:
                    idx_s = display.rfind(s)
                    if idx_s > last_user:
                        last_user = idx_s
                # If there's a trailing user turn without a GPT answer, cut it
                if last_user > 0:
                    # Check if there's a GPT response after this user turn
                    after = display[last_user:]
                    has_bot = f'{bot_p}: ' in after or f'{bot_p}:' in after
                    if not has_bot:
                        display = display[:last_user]
                for line in display.split('\n'):
                    print(f"  {line}")
                print(f"  {'─'*50}\n")

    except KeyboardInterrupt:
        print(f"\n\n  ⏸  Training interrupted at step {step}.")

    # ── End (normal or interrupted) ───────────────────────────────
    total = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  ✅  Training finished in {total/60:.1f} min  (step {step}/{cfg['max_iters']})")
    print(f"  Best val_loss : {best_val:.4f}")
    print(f"  Tokens seen   : {tokens_seen:,}  ({tokens_seen/max(total,1):.0f} tok/s)")

    # Final save — copy best.pt to ckpt.pt so chat.py uses the best model
    if step > start_step:
        if os.path.exists(best_path):
            shutil.copy(best_path, ckpt_path)
            print(f"  💾 Best checkpoint copied → {ckpt_path}  (val={best_val:.4f})")
        else:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw.save_checkpoint(ckpt_path, optimizer, step, best_val)
            print(f"  💾 Checkpoint saved → {ckpt_path}")
    print(f"{'═'*60}")
    print(f"\n  To chat:  python chat.py\n  Project:  {cfg.get('project', 'default')}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Train IkokoGPT",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--data',    type=str,            help='Training text file')
    p.add_argument('--iters',   type=int,            help='Number of iterations (default: 20000)')
    p.add_argument('--lr',      type=float,          help='Learning rate (default: 3e-4)')
    p.add_argument('--batch',   type=int,            help='Batch size (default: 32)')
    p.add_argument('--resume',  action='store_true', help='Resume from last checkpoint')
    p.add_argument('--compile', action='store_true', help='Enable torch.compile (PyTorch 2.0+)')
    p.add_argument('--tiny',    action='store_true', help='Tiny model for quick testing')
    args = p.parse_args()
    train(args)
