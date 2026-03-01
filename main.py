"""
main.py — IkokoGPT Main Menu
═══════════════════════════════════
Single entry point for the project.

Usage:
  python main.py
"""

import os
import sys
import subprocess

# Base directory = folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(path: str) -> str:
    """Convert a relative path to absolute based on BASE_DIR."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)

# ─── ANSI Colors ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RED    = "\033[91m"

BANNER = f"""
{CYAN}╔══════════════════════════════════════════════════════╗
║             🧠  IkokoGPT — Main Menu                  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   1 → 🚀  Train the model                            ║
║   2 → 💬  Start chat                                 ║
║   3 → 💬  Start chat (best checkpoint)               ║
║   4 → ⚙️   View configuration (config.yaml)           ║
║   5 → ℹ️   Model info                                 ║
║   0 → 🚪  Quit                                       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝{RESET}
"""


# ─── Utilities ────────────────────────────────────────────────────────────────

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause():
    input(f"\n  {GRAY}Press Enter to return to menu...{RESET}")

def get_project() -> str:
    """Read the current project name from config.yaml."""
    try:
        import yaml
        with open(_abs('config.yaml'), 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        return str(raw.get('project', 'default')).strip()
    except Exception:
        return 'default'

def check_checkpoints():
    """Check if a trained model exists for the current project."""
    proj_dir = _abs(f'checkpoints/{get_project()}')
    ckpt  = os.path.exists(os.path.join(proj_dir, 'ckpt.pt'))
    best  = os.path.exists(os.path.join(proj_dir, 'best.pt'))
    tok   = os.path.exists(os.path.join(proj_dir, 'tokenizer.json'))
    return ckpt or best, best, tok

def run(cmd: list):
    """Run a command and wait for it to finish."""
    # Resolve the script path relative to main.py folder
    _base = os.path.dirname(os.path.abspath(__file__))
    resolved = [os.path.join(_base, cmd[0])] + cmd[1:]
    try:
        result = subprocess.run([sys.executable] + resolved, cwd=_base)
        if result.returncode != 0:
            print(f"\n  {RED}⚠️  Script exited with an error (code {result.returncode}).{RESET}")
            print(f"  {GRAY}Try running directly: python {cmd[0]}{RESET}")
            pause()
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Interrupted.{RESET}")


# ─── Menu Actions ─────────────────────────────────────────────────────────────

def menu_train():
    clear()
    print(f"\n{CYAN}{'═'*56}")
    print(f"  🚀  Training")
    print(f"{'═'*56}{RESET}\n")

    print(f"  {BOLD}Options:{RESET}")
    print(f"  {GREEN}1{RESET} → Standard training")
    print(f"  {GREEN}2{RESET} → Training with torch.compile (+20% speed, PyTorch 2.0+)")
    print(f"  {GREEN}3{RESET} → Resume from last checkpoint")
    print(f"  {GREEN}4{RESET} → Quick test (tiny model, ~30 seconds)")
    print(f"  {GREEN}0{RESET} → Back to menu\n")

    choice = input(f"  {CYAN}{BOLD}Choice >{RESET} ").strip()

    if choice == '1':
        run(['train.py'])
    elif choice == '2':
        run(['train.py', '--compile'])
    elif choice == '3':
        run(['train.py', '--resume'])
    elif choice == '4':
        run(['train.py', '--tiny'])
    elif choice == '0':
        return
    else:
        print(f"  {RED}Invalid choice.{RESET}")
        pause()


def menu_chat(best=False):
    clear()
    has_model, has_best, has_tok = check_checkpoints()

    if not has_model or not has_tok:
        print(f"\n  {RED}❌  No model found!{RESET}")
        print(f"  Run training first (option 1).\n")
        pause()
        return

    if best and not has_best:
        print(f"\n  {YELLOW}⚠️  No best checkpoint (best.pt) — using ckpt.pt{RESET}\n")
        best = False

    print(f"\n{CYAN}{'═'*56}")
    print(f"  💬  Chat{' (best checkpoint)' if best else ''}")
    print(f"{'═'*56}{RESET}\n")
    print(f"  {GRAY}Type /quit to return to menu.{RESET}\n")

    cmd = ['chat.py']
    if best:
        cmd.append('--best')
    run(cmd)


def menu_config():
    clear()
    print(f"\n{CYAN}{'═'*56}")
    print(f"  ⚙️   Current configuration (config.yaml)")
    print(f"{'═'*56}{RESET}\n")

    if not os.path.exists(_abs('config.yaml')):
        print(f"  {RED}❌  config.yaml not found!{RESET}\n")
        pause()
        return

    try:
        import yaml
        with open(_abs('config.yaml'), 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        sections = {
            'data'          : '📂  Data',
            'model'         : '🧠  Model',
            'train'         : '🚀  Training',
            'eval'          : '📊  Evaluation',
            'generate'      : '✍️   Generation',
            'early_stopping': '⏹   Early stopping',
            'chat'          : '💬  Chat',
        }
        for key, label in sections.items():
            if key in raw and raw[key]:
                print(f"  {BOLD}{label}{RESET}")
                for k, v in raw[key].items():
                    print(f"    {GRAY}{k:<20}{RESET} {GREEN}{v}{RESET}")
                print()

    except ImportError:
        print(f"  {YELLOW}PyYAML not installed — raw display:{RESET}\n")
        with open(_abs('config.yaml'), 'r', encoding='utf-8') as f:
            print(f.read())

    pause()


def menu_info():
    clear()
    print(f"\n{CYAN}{'═'*56}")
    print(f"  ℹ️   Model Information")
    print(f"{'═'*56}{RESET}\n")

    has_model, has_best, has_tok = check_checkpoints()

    if not has_model:
        print(f"  {RED}❌  No trained model found.{RESET}")
        print(f"  Run training first (option 1).\n")
        pause()
        return

    try:
        import torch
        from model import MiniGPT
        from tokenizer import load_tokenizer

        project   = get_project()
        proj_dir  = _abs(f'checkpoints/{project}')
        ckpt_path = os.path.join(proj_dir, 'best.pt') if has_best \
                    else os.path.join(proj_dir, 'ckpt.pt')
        device    = 'cuda' if torch.cuda.is_available() else 'cpu'

        model     = MiniGPT.from_checkpoint(ckpt_path, device=device)
        tokenizer = load_tokenizer(os.path.join(proj_dir, 'tokenizer.json'))
        c         = model.config
        n_params  = sum(p.numel() for p in model.parameters())

        print(f"  {BOLD}Checkpoint   :{RESET} {ckpt_path}")
        print(f"  {BOLD}Parameters   :{RESET} {n_params/1e6:.2f}M")
        print(f"  {BOLD}Architecture :{RESET} {c.d_model}d × {c.n_heads} heads × {c.n_layers} layers")
        print(f"  {BOLD}Context      :{RESET} {c.seq_len} tokens")
        print(f"  {BOLD}Vocabulary   :{RESET} {c.vocab_size} tokens")
        print(f"  {BOLD}Flash Attn   :{RESET} {'✓' if c.flash_attn else '✗'}")
        print(f"  {BOLD}Device       :{RESET} {device}")
        print(f"  {BOLD}Tokenizer    :{RESET} {tokenizer}")

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  {BOLD}GPU          :{RESET} {torch.cuda.get_device_name(0)} ({mem:.1f} GB)")

    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")

    print()
    pause()


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    while True:
        clear()
        print(BANNER)

        # Model status
        has_model, has_best, _ = check_checkpoints()
        project = get_project()
        if has_model:
            label = f"{GREEN}✓ Trained model available{RESET}"
            if has_best:
                label += f"  {GRAY}(best.pt){RESET}"
        else:
            label = f"{YELLOW}⚠️  No model — run training first{RESET}"
        print(f"  Project : {CYAN}{project}{RESET}  |  Status: {label}\n")

        choice = input(f"  {CYAN}{BOLD}Choice >{RESET} ").strip()

        if choice == '1':
            menu_train()
        elif choice == '2':
            menu_chat(best=False)
        elif choice == '3':
            menu_chat(best=True)
        elif choice == '4':
            menu_config()
        elif choice == '5':
            menu_info()
        elif choice in ('0', 'q', 'quit', 'exit'):
            clear()
            print(f"\n  {CYAN}Goodbye! 👋{RESET}\n")
            break
        else:
            print(f"  {RED}Invalid choice — enter 0 to 5.{RESET}")
            pause()


if __name__ == '__main__':
    main()
