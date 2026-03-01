"""
model.py — Optimised GPT with PyTorch
══════════════════════════════════════════════════════════════════════

Included optimisations:
  ✅ Flash Attention (F.scaled_dot_product_attention — PyTorch 2.0+)
  ✅ Mixed Precision (float16/bfloat16)
  ✅ torch.compile() — JIT compile the model (PyTorch 2.0+)
  ✅ Causal mask in shared memory
  ✅ Tied embeddings (shared token_emb ↔ lm_head)
  ✅ Weight initialisation (GPT-2 style)
  ✅ Residual scaling (1/√n_layers)
  ✅ Temperature / Top-k / Top-p / Repetition penalty
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    # Architecture
    vocab_size : int   = 65      # vocabulary size (auto-updated)
    seq_len    : int   = 128     # maximum context length
    d_model    : int   = 256     # embedding dimension
    n_heads    : int   = 4       # attention heads
    n_layers   : int   = 4       # transformer blocks
    d_ff       : int   = 0       # FFN dim (0 = auto: 4 × d_model)
    dropout    : float = 0.1     # dropout (set to 0 at inference automatically)

    # Optimisations
    bias       : bool  = False   # bias in Linear layers (False = faster)
    flash_attn : bool  = True    # Flash Attention if available

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


# ─── Core Components ──────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Uses Flash Attention (F.scaled_dot_product_attention) when available.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.flash   = config.flash_attn and hasattr(F, 'scaled_dot_product_attention')

        # Q, K, V in a single projection (3× more efficient)
        self.qkv        = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj       = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask (only if Flash Attention is unavailable)
        if not self.flash:
            self.register_buffer(
                'mask',
                torch.tril(torch.ones(config.seq_len, config.seq_len))
                      .view(1, 1, config.seq_len, config.seq_len)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V then split
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.flash:
            # ⚡ Flash Attention — optimised CUDA kernel, 2-4× faster
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask  = None,
                dropout_p  = self.attn_drop.p if self.training else 0.0,
                is_causal  = True,
            )
        else:
            # Fallback: manual attention
            scale = 1.0 / math.sqrt(self.d_head)
            attn  = (q @ k.transpose(-2, -1)) * scale
            attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            attn  = F.softmax(attn, dim=-1)
            attn  = self.attn_drop(attn)
            y     = attn @ v

        # Recombine heads: (B, H, T, D) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """2-layer MLP with GELU activation (GPT-2 style)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff,    bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff,    config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with Pre-LayerNorm (more stable than Post-LN).
    Residual connections to facilitate gradient flow.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ff   = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # residual connection after attention
        x = x + self.ff(self.ln2(x))    # residual connection after FFN
        return x


# ─── Main Model ───────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    """
    IkokoGPT — generates text token by token.

    Architecture:
      TokenEmbedding + PositionEmbedding
      → N × TransformerBlock (Attention + FFN)
      → Final LayerNorm
      → LM Head (tied to TokenEmbedding)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(config.vocab_size, config.d_model),
            pos_emb = nn.Embedding(config.seq_len,    config.d_model),
            drop    = nn.Dropout(config.dropout),
            blocks  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            ln_f    = nn.LayerNorm(config.d_model, bias=config.bias),
        ))

        # Tied weights: lm_head shares weights with tok_emb (fewer parameters)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.tok_emb.weight = self.lm_head.weight

        # GPT-2 style initialisation
        self.apply(self._init_weights)

        # Scale residual projections: std = 0.02 / √(2 × n_layers)
        for pname, p in self.named_parameters():
            if pname.endswith('proj.weight') or pname.endswith('net.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        n = self.count_params()
        flash = '✓' if config.flash_attn and hasattr(F, 'scaled_dot_product_attention') else '✗'
        print(f"  ⚡ IkokoGPT  |  {n/1e6:.2f}M params  |  "
              f"Flash={flash}  |  seq={config.seq_len}  |  vocab={config.vocab_size}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self, non_embedding: bool = True) -> int:
        """Count parameters (excluding position embeddings by default)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.pos_emb.weight.numel()
        return n

    def forward(self,
                idx    : torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        """
        idx     : (B, T) LongTensor of token ids
        targets : (B, T) LongTensor — if provided, computes cross-entropy loss

        Returns (logits, loss) — loss=None if no targets provided.
        """
        B, T = idx.shape
        assert T <= self.config.seq_len, (
            f"Sequence too long: {T} > {self.config.seq_len}"
        )

        pos = torch.arange(T, device=idx.device)

        # Token + position embeddings
        x = self.transformer.drop(
            self.transformer.tok_emb(idx) +
            self.transformer.pos_emb(pos)
        )

        # Transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training mode: loss over all tokens
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference mode: only the last token (faster)
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx              : torch.Tensor,
                 max_new          : int   = 200,
                 temperature      : float = 0.8,
                 top_k            : int   = 50,
                 top_p            : float = 0.9,
                 repetition_penalty: float = 1.1,
                 stop_token       : Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive generation with:
          • Temperature scaling   — controls creativity
          • Top-k sampling        — limits to k best tokens
          • Top-p (nucleus)       — limits to most probable tokens
          • Repetition penalty    — penalises repetition
          • Stop token            — stops if this token is generated

        idx : (1, T) initial context tensor
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new):
            # Truncate to maximum context
            idx_cond = idx if idx.size(1) <= self.config.seq_len \
                       else idx[:, -self.config.seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]   # (1, vocab_size)

            # ── Repetition penalty ────────────────────────────────
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # ── Temperature ───────────────────────────────────────
            logits = logits / max(temperature, 1e-6)

            # ── Top-k ─────────────────────────────────────────────
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # ── Top-p (nucleus sampling) ──────────────────────────
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Shift one rank right to always keep at least 1 token
                remove = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
                # Always keep the most probable token (avoids empty distribution)
                remove[:, 0] = False
                sorted_logits[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # ── Sampling ──────────────────────────────────────────
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx     = torch.cat([idx, next_id], dim=1)

            # Stop if special token encountered
            if stop_token is not None and next_id.item() == stop_token:
                break

        # Restore original mode
        if was_training:
            self.train()

        return idx

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str, optimizer=None, step: int = 0, loss: float = 0.0):
        """Save the model + optionally the optimizer state."""
        ckpt = {
            'config'    : self.config,
            'model'     : self.state_dict(),
            'step'      : step,
            'val_loss'  : loss,
        }
        if optimizer is not None:
            ckpt['optimizer'] = optimizer.state_dict()
        torch.save(ckpt, path)
        print(f"  💾 Checkpoint saved → {path}  (step={step})")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cpu') -> 'MiniGPT':
        """Load a model from a checkpoint."""
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt['config'])
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        step  = ckpt.get('step', 0)
        vloss = ckpt.get('val_loss', float('inf'))
        print(f"  ✓ Model loaded — step={step}  val_loss={vloss:.4f}")
        return model
