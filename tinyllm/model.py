"""A decoder-only Transformer (GPT-style), built from PyTorch primitives: nn.Linear, nn.Embedding, matmul."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class LayerNorm(nn.Module):
    """Normalizes each vector to mean 0 and variance 1, then rescales it with learned parameters."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


# ---------------------------------------------------------------------- RoPE
def rope_tables(head_dim, max_len, base=10000.0):
    """Rotation angles: dimension pair i is rotated by position * base^(-2i/d)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)   # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotates the pairs (x[2i], x[2i+1]); the dot product q·k then depends only on the relative distance."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


# ---------------------------------------------------------------------- KV cache
class KVCache:
    """Keeps the K and V already computed, so generation only has to process the new token."""

    def __init__(self, n_layer):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    @property
    def length(self):
        return 0 if self.k[0] is None else self.k[0].size(2)

    def update(self, layer, k, v):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


# ---------------------------------------------------------------------- blocks
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.layer_idx = layer_idx
        self.impl = cfg.attn_impl
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # Q, K, V from a single matmul
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x, rope=None, kv_cache=None, pos=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head works on its own slice of head_dim dimensions
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope[0][pos:pos + T], rope[1][pos:pos + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
        Tk = k.size(2)  # how many keys it sees: T without a cache, pos + T with one
        mask = self.mask[:, :, Tk - T:Tk, :Tk]  # row i (the query at position pos+i) sees keys 0..pos+i

        if self.impl == "manual":
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, H, T, Tk)
            att = att.masked_fill(~mask, float("-inf"))   # forbid looking into the future
            att = F.softmax(att.float(), dim=-1).type_as(v)
            att = self.attn_drop(att)
            y = att @ v                                    # weighted average of the values
        else:
            drop = self.dropout if self.training else 0.0
            if Tk == T:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)

        y = y.transpose(1, 2).contiguous().view(B, T, C)   # glue the heads back together
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm: x + Attention(LN(x)), then x + MLP(LN(x)). Attention moves information between
    tokens; the MLP processes each token on its own."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, kv_cache=None, pos=0):
        x = x + self.attn(self.ln1(x), rope, kv_cache, pos)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------- the model
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos_emb == "learned" else None
        if cfg.pos_emb == "rope":
            cos, sin = rope_tables(cfg.d_model // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Projections that write into the residual stream are scaled down by 1/sqrt(2*n_layer)
        # (as in GPT-2); otherwise the variance of the stream grows with every layer added.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_cache=None, last_only=False):
        B, T = idx.shape
        pos = kv_cache.length if kv_cache is not None else 0
        assert pos + T <= self.cfg.block_size, f"context too long: {pos + T} > {self.cfg.block_size}"

        x = self.tok_emb(idx)                                         # (B, T, d_model)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(pos, pos + T, device=idx.device))
        x = self.drop(x)
        rope = (self.rope_cos, self.rope_sin) if self.cfg.pos_emb == "rope" else None
        for block in self.blocks:
            x = block(x, rope, kv_cache, pos)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # cross-entropy = -log p(correct token); computed in fp32 for numerical stability
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            return logits, loss
        if last_only:
            x = x[:, -1:, :]   # generation only needs the prediction for the last token
        return self.lm_head(x), None

    def optim_groups(self, weight_decay):
        """Weight decay only on matrices (Linear, Embedding), not on biases or LayerNorm."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]
