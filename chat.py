"""
chat.py — Interactive chat interface
════════════════════════════════════════

Usage:
  python chat.py                  # default model (ckpt.pt)
  python chat.py --best           # best checkpoint (best.pt)
  python chat.py --temp 0.7       # less creative / more coherent
  python chat.py --temp 1.2       # more creative / more surprising
  python chat.py --no-stream      # display all at once

Chat commands:
  /temp  0.8   → change temperature
  /topk  50    → top-k sampling
  /topp  0.9   → nucleus sampling
  /len   200   → generation length
  /rep   1.1   → repetition penalty
  /info        → model info
  /reset       → clear history
  /quit        → quit
"""

import os
import re
import sys
import argparse
import time

import torch
import torch.nn.functional as F

from model     import MiniGPT
from tokenizer import load_tokenizer

# Base directory = folder where chat.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(path: str) -> str:
    """Convert a relative path to absolute based on BASE_DIR."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


# ─── Configuration Loading ────────────────────────────────────────────────────

def load_chat_config(config_path: str = None) -> dict:
    """Load chat parameters and project-aware paths from config.yaml."""
    if config_path is None:
        config_path = _abs('config.yaml')
    defaults = {
        'temperature' : 0.8,
        'top_k'       : 50,
        'top_p'       : 0.9,
        'max_new'     : 200,
        'rep_penalty' : 1.1,
        'project'     : 'default',
        'user_prefix' : 'Toi',
        'bot_prefix'  : 'GPT',
    }
    if not os.path.exists(config_path):
        return defaults
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        chat_cfg = raw.get('chat', {}) or {}
        data_cfg = raw.get('data', {}) or {}
        project  = str(raw.get('project', defaults['project'])).strip()
        return {
            'temperature' : chat_cfg.get('temperature', defaults['temperature']),
            'top_k'       : chat_cfg.get('top_k',       defaults['top_k']),
            'top_p'       : chat_cfg.get('top_p',       defaults['top_p']),
            'max_new'     : chat_cfg.get('max_new',     defaults['max_new']),
            'rep_penalty' : chat_cfg.get('rep_penalty', defaults['rep_penalty']),
            'project'     : project,
            'user_prefix' : data_cfg.get('user_prefix', defaults['user_prefix']),
            'bot_prefix'  : data_cfg.get('bot_prefix',  defaults['bot_prefix']),
        }
    except Exception:
        return defaults


# ─── Couleurs ANSI ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"

BANNER = f"""
{CYAN}╔══════════════════════════════════════════════════════╗
║          🧠  IkokoGPT — Text Generator                ║
╠══════════════════════════════════════════════════════╣
║  Type a sentence start → the model continues         ║
║                                                      ║
║  Commands:                                           ║
║    /temp  0.8   → creativity  (0.3=precise 1.5=free) ║
║    /topk  50    → top-k diversity                    ║
║    /topp  0.9   → nucleus sampling                   ║
║    /len   200   → generation length                  ║
║    /rep   1.1   → repetition penalty                 ║
║    /info        → model info                         ║
║    /reset       → reset                              ║
║    /quit        → quit                               ║
╚══════════════════════════════════════════════════════╝{RESET}
"""


# ─── Streaming Generation ─────────────────────────────────────────────────────

@torch.no_grad()
def stream_generate(model, tokenizer, prompt: str,
                    max_new     : int   = 200,
                    temperature : float = 0.8,
                    top_k       : int   = 50,
                    top_p       : float = 0.9,
                    rep_penalty : float = 1.1,
                    device      : str   = 'cpu',
                    stream      : bool  = True,
                    user_prefix : str   = 'Toi',
                    bot_prefix  : str   = 'GPT') -> str:
    """
    Generate text token by token.
    If stream=True, display each token as it is generated.
    """
    was_training = model.training
    model.eval()

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]  # fallback si le prompt est vide

    idx        = torch.tensor([tokens], dtype=torch.long, device=device)
    prompt_decoded = tokenizer.decode(tokens)
    t0        = time.time()
    n_gen     = 0

    # Stop sequences: a new speaker turn ends the response
    stop_seqs  = [f'\n{user_prefix}: ', f'\n{bot_prefix}: ',
                  f'\n{user_prefix}:', f'\n{bot_prefix}:']
    max_stop   = max(len(s) for s in stop_seqs)
    generated_tokens = []
    last_decoded     = ""
    response_started = False

    for _ in range(max_new):
        # Tronquer au contexte max
        idx_cond = idx[:, -model.config.seq_len:]

        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :]   # (1, vocab_size)

        # ── Repetition penalty (on generated tokens only) ────────────
        if rep_penalty != 1.0:
            for tid in set(generated_tokens):
                if 0 <= tid < logits.size(-1):
                    if logits[0, tid] < 0:
                        logits[0, tid] *= rep_penalty
                    else:
                        logits[0, tid] /= rep_penalty

        # ── Temperature ───────────────────────────────────────────
        logits = logits / max(temperature, 1e-6)

        # ── Top-k ─────────────────────────────────────────────────
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float('-inf')

        # ── Top-p ─────────────────────────────────────────────────
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = (cum - F.softmax(sorted_logits, dim=-1)) > top_p
            remove[:, 0] = False  # garder toujours le token le plus probable
            sorted_logits[remove] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        # ── Sampling ──────────────────────────────────────────────
        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx     = torch.cat([idx, next_id], dim=1)
        n_gen  += 1

        # Decode the ENTIRE generated sequence at once (correct with BPE)
        generated_tokens.append(next_id.item())
        full_generated = tokenizer.decode(generated_tokens)

        # ── Stop token: stop when a new turn is detected ──────────────
        stop = False
        for stop_seq in stop_seqs:
            if stop_seq in full_generated:
                cut = full_generated.index(stop_seq)
                full_generated = full_generated[:cut]
                stop = True
                break

        if stream:
            # New text since last display
            new_text = full_generated[len(last_decoded):]
            if new_text:
                if not response_started:
                    new_text = new_text.lstrip(' \n\r')
                    if new_text:
                        response_started = True

                if response_started:
                    # Hold back max_stop chars to detect stop sequences
                    safe_len  = max(0, len(full_generated) - max_stop)
                    safe_text = full_generated[:safe_len]
                    to_print  = safe_text[len(last_decoded):]
                    if to_print:
                        print(re.sub(r'  +', ' ', to_print), end='', flush=True)
                        last_decoded = safe_text

        if stop:
            # Display what remains before the stop
            remaining = full_generated[len(last_decoded):]
            if stream and remaining:
                remaining = re.sub(r'  +', ' ', remaining)
                print(remaining, end='', flush=True)
            last_decoded = full_generated
            break

    # Display what remains in the buffer (normal end, no stop)
    if stream:
        # Flush everything not yet printed (buffered window + anything remaining)
        remaining = full_generated[len(last_decoded):]
        if not response_started:
            remaining = remaining.lstrip(' \n\r')
        if remaining:
            print(re.sub(r'  +', ' ', remaining), end='', flush=True)

    elapsed = time.time() - t0
    tok_s   = n_gen / max(elapsed, 1e-6)

    if stream:
        print(f"\n\n{GRAY}  [{n_gen} tokens • {elapsed:.1f}s • {tok_s:.0f} tok/s]{RESET}")

    # Restore original mode (important if called from train.py)
    if was_training:
        model.train()

    return prompt_decoded + full_generated


# ─── Main Loop ────────────────────────────────────────────────────────────────

def chat(args):
    # ── Device detection ──────────────────────────────────────────
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # ── Model loading ─────────────────────────────────────────────
    cfg      = load_chat_config()
    project  = cfg['project']
    proj_dir = _abs(f'checkpoints/{project}')

    ckpt_path = os.path.join(proj_dir, 'best.pt') if args.best else os.path.join(proj_dir, 'ckpt.pt')
    # Default loads ckpt.pt (last checkpoint), --best loads best.pt (best val_loss)

    print(f"\n  Project : {project}  →  {proj_dir}")

    if not os.path.exists(ckpt_path):
        # Try the other checkpoint if the requested one doesn't exist
        alt = os.path.join(proj_dir, 'ckpt.pt') if args.best else os.path.join(proj_dir, 'best.pt')
        if os.path.exists(alt):
            print(f"  ⚠️  {ckpt_path} not found — using {alt}")
            ckpt_path = alt
        else:
            print(f"\n  ❌  No model found in {proj_dir}")
            print("  Run first:  python train.py")
            sys.exit(1)

    tok_path = os.path.join(proj_dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        print(f"  ❌  Tokenizer not found: {tok_path}")
        print("  Run first:  python train.py")
        sys.exit(1)

    print(f"\n  Loading {ckpt_path}...")
    model     = MiniGPT.from_checkpoint(ckpt_path, device=device)
    tokenizer = load_tokenizer(tok_path)
    print(f"  Tokenizer : {tokenizer}")

    # ── Default parameters (from config.yaml, overridden by CLI) ────
    params = {
        'temperature' : args.temp or cfg['temperature'],
        'top_k'       : args.topk or cfg['top_k'],
        'top_p'       : args.topp or cfg['top_p'],
        'max_new'     : args.len  or cfg['max_new'],
        'rep_penalty' : cfg['rep_penalty'],
        'user_prefix' : cfg['user_prefix'],
        'bot_prefix'  : cfg['bot_prefix'],
    }
    stream = not args.no_stream

    print(BANNER)
    print(f"  {GRAY}Device : {device}  |  "
          f"Temp : {params['temperature']}  |  "
          f"Top-k : {params['top_k']}  |  "
          f"Top-p : {params['top_p']}  |  "
          f"Length : {params['max_new']}{RESET}\n")

    # ── Conversation history ──────────────────────────────────────
    history = []   # list of (user_msg, gpt_response) tuples
    max_history_turns = 6  # keep the N last exchanges in context

    # ── Chat loop ─────────────────────────────────────────────────
    user_prefix = params['user_prefix']
    bot_prefix  = params['bot_prefix']

    while True:
        try:
            prompt = input(f"  {CYAN}{BOLD}{user_prefix} >{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  Goodbye! 👋\n")
            break

        if not prompt:
            continue

        # ── Slash commands ────────────────────────────────────────
        if prompt.startswith('/'):
            parts = prompt.split(maxsplit=1)
            cmd   = parts[0].lower()
            val   = parts[1].strip() if len(parts) > 1 else ''

            if cmd in ('/quit', '/q', '/exit', '/bye'):
                print(f"\n  Goodbye! 👋\n")
                break

            elif cmd in ('/temp', '/temperature'):
                try:
                    params['temperature'] = float(val)
                    print(f"  {GREEN}✓ Temperature → {params['temperature']}{RESET}")
                except ValueError:
                    print(f"  Usage: /temp 0.8   (between 0.1 and 2.0)")

            elif cmd == '/topk':
                try:
                    params['top_k'] = int(val)
                    print(f"  {GREEN}✓ Top-k → {params['top_k']}{RESET}")
                except ValueError:
                    print(f"  Usage: /topk 50")

            elif cmd == '/topp':
                try:
                    params['top_p'] = float(val)
                    print(f"  {GREEN}✓ Top-p → {params['top_p']}{RESET}")
                except ValueError:
                    print(f"  Usage: /topp 0.9")

            elif cmd in ('/len', '/length'):
                try:
                    params['max_new'] = int(val)
                    print(f"  {GREEN}✓ Length → {params['max_new']}{RESET}")
                except ValueError:
                    print(f"  Usage: /len 200")

            elif cmd in ('/rep', '/repetition'):
                try:
                    params['rep_penalty'] = float(val)
                    print(f"  {GREEN}✓ Repetition penalty → {params['rep_penalty']}{RESET}")
                except ValueError:
                    print(f"  Usage: /rep 1.1   (1.0 = disabled)")

            elif cmd == '/info':
                n = sum(p.numel() for p in model.parameters())
                c = model.config
                print(f"\n  {BOLD}── Model Information ───────────────────────{RESET}")
                print(f"  Parameters  : {n/1e6:.2f}M")
                print(f"  Architecture: {c.d_model}d × {c.n_heads}h × {c.n_layers}L")
                print(f"  Context     : {c.seq_len} tokens")
                print(f"  Vocabulary  : {c.vocab_size} tokens")
                print(f"  Flash Attn  : {'✓' if c.flash_attn else '✗'}")
                print(f"  Device      : {device}")
                print(f"  Tokenizer   : {tokenizer}")
                print(f"\n  {BOLD}── Current Parameters ──────────────────────{RESET}")
                for k, v in params.items():
                    print(f"  {k:<15} : {v}")
                print()

            elif cmd in ('/reset', '/clear'):
                history.clear()
                print(f"  {GREEN}✓ History cleared + parameters reset{RESET}")
                params.update(temperature=0.8, top_k=50, top_p=0.9,
                              max_new=200, rep_penalty=1.1)

            elif cmd == '/help':
                print(f"""
  {BOLD}Available commands:{RESET}
    /temp  <val>   → Temperature (0.1–2.0)   e.g. /temp 0.9
    /topk  <val>   → Top-k sampling           e.g. /topk 40
    /topp  <val>   → Nucleus sampling         e.g. /topp 0.95
    /len   <val>   → Max length               e.g. /len 300
    /rep   <val>   → Repetition penalty       e.g. /rep 1.2
    /info          → Model info
    /reset         → Clear history + reset parameters
    /quit          → Quit
""")
            else:
                print(f"  {YELLOW}Unknown command: {cmd}  (type /help for help){RESET}")

            continue

        # ── Build context with history ────────────────────────────
        # Rebuild the prompt with the N last exchanges
        context = ""
        for user_msg, gpt_resp in history[-max_history_turns:]:
            context += f"{user_prefix}: {user_msg}\n{bot_prefix}: {gpt_resp}\n"
        context += f"{user_prefix}: {prompt}\n{bot_prefix}: "

        print(f"\n  {GREEN}{BOLD}{bot_prefix} >{RESET} ", end='', flush=True)

        result = stream_generate(
            model, tokenizer, context,
            device=device, stream=stream,
            **params,
        )

        # Extract only the GPT response (without the context)
        # Find the last occurrence of "BOT: " in the result
        marker = f"{user_prefix}: {prompt}\n{bot_prefix}: "
        if marker in result:
            response = result[result.rfind(marker) + len(marker):]
        else:
            context_decoded = tokenizer.decode(tokenizer.encode(context))
            response = result[len(context_decoded):]

        if not stream:
            print(response)
            print(f"\n{GRAY}  [{len(tokenizer.encode(response))} tokens]{RESET}")

        # Clean double spaces and leading/trailing spaces before saving
        response = re.sub(r'  +', ' ', response).strip()
        # Save to history
        history.append((prompt, response))

        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Chat with IkokoGPT",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--best',      action='store_true',
                   help='Load the best checkpoint (best.pt)')
    p.add_argument('--temp',      type=float,
                   help='Generation temperature (default: 0.8)')
    p.add_argument('--topk',      type=int,
                   help='Top-k sampling (default: 50)')
    p.add_argument('--topp',      type=float,
                   help='Nucleus sampling top-p (default: 0.9)')
    p.add_argument('--len',       type=int,
                   help='Max generation length (default: 200)')
    p.add_argument('--no-stream', action='store_true', dest='no_stream',
                   help='Display all at once (no streaming)')
    args = p.parse_args()
    chat(args)
