"""
tokenizer.py — Robust tokenizer for natural language text
══════════════════════════════════════════════════════════

Two modes:
  1. CharTokenizer — one character = one token
     → Simple, fast to train, no information loss
     → Recommended to start with

  2. BPETokenizer  — lightweight Byte Pair Encoding
     → Better compression, model learns faster
     → Useful when you have lots of data

Usage:
  from tokenizer import build_tokenizer, load_tokenizer

  tok = build_tokenizer(text, kind='char')
  ids = tok.encode("Hello!")
  txt = tok.decode(ids)

  tok.save('checkpoints/tokenizer.json')
  tok = load_tokenizer('checkpoints/tokenizer.json')
"""

import re
import json
from collections import defaultdict
from pathlib import Path


# ─── Character Tokenizer ──────────────────────────────────────────────────────

class CharTokenizer:
    """
    Character-by-character tokenizer.

    Advantages:
      • Minimal vocabulary (often < 150 tokens)
      • No information loss — can encode any text
      • No preprocessing required

    Disadvantage:
      • Sequences are longer → context consumed faster
    """

    # Special token for unknown characters
    UNK = '<UNK>'

    def __init__(self, text: str = ""):
        if text:
            chars = sorted(set(text))
        else:
            # Base ASCII vocabulary + common accented characters
            base  = [chr(i) for i in range(32, 127)]
            extra = list("àâäéèêëîïôöùûüç°ÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ\n\t")
            chars = sorted(set(base + extra))

        # Unknown token first (index 0)
        all_tokens = [self.UNK] + [c for c in chars if c != self.UNK]
        self.stoi      = {c: i for i, c in enumerate(all_tokens)}
        self.itos      = {i: c for i, c in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

    def encode(self, text: str) -> list:
        """Encode text into a list of integers."""
        unk_id = self.stoi[self.UNK]
        return [self.stoi.get(c, unk_id) for c in text]

    def decode(self, tokens: list) -> str:
        """Decode a list of integers into text."""
        return ''.join(
            self.itos.get(t, '') for t in tokens
            if self.itos.get(t, '') != self.UNK
        )

    def save(self, path: str):
        """Save the tokenizer to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'type': 'char', 'stoi': self.stoi}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load a tokenizer from a JSON file."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        obj           = cls.__new__(cls)
        obj.stoi      = data['stoi']
        obj.itos      = {int(i): c for c, i in data['stoi'].items()}
        obj.vocab_size = len(obj.stoi)
        return obj

    def __repr__(self):
        return f"CharTokenizer(vocab={self.vocab_size})"


# ─── BPE Tokenizer ────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Lightweight Byte Pair Encoding (BPE).

    Principle:
      1. Start with all individual characters as tokens
      2. Find the most frequent consecutive token pair
      3. Merge that pair into a new token
      4. Repeat until reaching the desired vocabulary size

    Advantage: frequent words become a single token
    → fewer tokens per text → more efficient context usage
    """

    SPECIAL = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

    def __init__(self):
        self.stoi      : dict = {}
        self.itos      : dict = {}
        self.merges    : list = []
        self.vocab_size: int  = 0

    def train(self, text: str, vocab_size: int = 500, verbose: bool = True):
        """Train BPE on `text` up to `vocab_size` tokens."""
        if not text.strip():
            raise ValueError("Cannot train BPE on empty text.")

        # Initial vocabulary: unique characters + special tokens
        chars = sorted(set(text))
        vocab = dict(self.SPECIAL)
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)

        # Word representation (lists of characters)
        word_counts = defaultdict(int)
        for word in text.split():
            word_counts[' '.join(list(word)) + ' </w>'] += 1

        n_merges = max(0, vocab_size - len(vocab))
        if verbose:
            print(f"  BPE: initial vocab={len(vocab)}, target={vocab_size} ({n_merges} merges)")

        for step in range(n_merges):
            # Count all adjacent pairs
            pairs = defaultdict(int)
            for word, freq in word_counts.items():
                syms = word.split()
                for i in range(len(syms) - 1):
                    pairs[(syms[i], syms[i + 1])] += freq

            if not pairs:
                break

            # Best pair
            best   = max(pairs, key=pairs.get)
            merged = best[0] + best[1]
            self.merges.append(best)

            # Merge in the corpus
            pattern   = re.compile(
                r'(?<!\S)' + re.escape(best[0]) + r' ' + re.escape(best[1]) + r'(?!\S)'
            )
            new_words = {}
            for word in word_counts:
                new_word           = pattern.sub(merged, word)
                new_words[new_word] = word_counts[word]
            word_counts = new_words

            vocab[merged] = len(vocab)

            if verbose and step % 100 == 0:
                print(f"    step {step:>4}  merge='{best[0]}'+'{best[1]}'  vocab={len(vocab)}")

        self.stoi      = vocab
        self.itos      = {i: c for c, i in vocab.items()}
        self.vocab_size = len(vocab)
        if verbose:
            print(f"  BPE done: vocab={self.vocab_size}")

    def _apply_merges(self, word: str) -> list:
        """Apply BPE merges to a word."""
        syms = list(word) + ['</w>']
        for a, b in self.merges:
            i, new_syms = 0, []
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(a + b)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        return syms

    def encode(self, text: str) -> list:
        """Encode text into a list of integers."""
        unk = self.SPECIAL['<unk>']
        tokens = []
        # Split on spaces but keep newlines and punctuation as separate tokens
        words = re.split(r'( )', text)
        for chunk in words:
            if chunk == ' ':
                space_id = self.stoi.get(' ', unk)
                tokens.append(space_id)
            elif chunk == '':
                continue
            else:
                # Further split on newlines, keeping them
                parts = re.split(r'(\n)', chunk)
                for part in parts:
                    if part == '':
                        continue
                    elif part == '\n':
                        nl_id = self.stoi.get('\n', unk)
                        tokens.append(nl_id)
                    else:
                        for sym in self._apply_merges(part):
                            tokens.append(self.stoi.get(sym, unk))
        return tokens

    def decode(self, tokens: list) -> str:
        """Decode a list of integers into text."""
        special = set(self.SPECIAL.values())
        parts   = []
        for t in tokens:
            if t in special:
                continue          # skip <pad>, <unk>, <bos>, <eos>
            sym = self.itos.get(t, '')
            parts.append(sym)
        text = ''.join(parts)
        # </w> marks end of word — replace with space
        text = text.replace('</w>', ' ')
        # Remove double spaces introduced by BPE
        text = re.sub(r'  +', ' ', text)
        return text

    def save(self, path: str):
        """Save the tokenizer to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'type': 'bpe', 'stoi': self.stoi, 'merges': self.merges},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load a tokenizer from a JSON file."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        obj            = cls.__new__(cls)
        obj.stoi       = data['stoi']
        obj.itos       = {int(i): c for c, i in data['stoi'].items()}
        obj.merges     = [tuple(m) for m in data['merges']]
        obj.vocab_size  = len(obj.stoi)
        return obj

    def __repr__(self):
        return f"BPETokenizer(vocab={self.vocab_size}, merges={len(self.merges)})"


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_tokenizer(text: str,
                    kind      : str = 'char',
                    vocab_size: int = 500,
                    save_path : str = None):
    """
    Build a tokenizer from text.

    Args:
      text       : training text
      kind       : 'char' (simple, recommended) or 'bpe' (more advanced)
      vocab_size : target vocabulary size (BPE only)
      save_path  : path to save the tokenizer (optional)

    Returns:
      CharTokenizer or BPETokenizer
    """
    if kind == 'bpe':
        tok = BPETokenizer()
        tok.train(text, vocab_size=vocab_size)
    else:
        tok = CharTokenizer(text)

    if save_path:
        tok.save(save_path)

    return tok


def load_tokenizer(path: str):
    """Load a tokenizer from a JSON file (auto-detects the type)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if data.get('type') == 'bpe':
        return BPETokenizer.load(path)
    return CharTokenizer.load(path)
