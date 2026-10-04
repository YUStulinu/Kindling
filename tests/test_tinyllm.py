"""Tests that check mathematical properties of the components, not just that they "run"."""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyllm.config import ModelConfig
from tinyllm.model import GPT, KVCache
from tinyllm.optim import AdamW, clip_grad_norm, lr_at
from tinyllm.tokenizer import EOT, SPLIT_PATTERN, BPETokenizer

# Romanian sample text (the tokenizer is trained on Romanian)
TEXT = ("Era odată ca niciodată, că de n-ar fi, nu s-ar povesti. Într-o țară îndepărtată trăia un împărat "
        "și o împărăteasă. Ștefan cel Mare a domnit între 1457 și 1504. ") * 20


def tiny_cfg(**kw):
    base = dict(vocab_size=300, block_size=32, n_layer=2, n_head=4, d_model=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


# ---------------------------------------------------------------------- tokenizer
@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(TEXT, vocab_size=400)
    return t


def test_split_pattern_covers_every_character():
    s = "a_b  ș-ț!! 123456 \n\n emoji 🙂 x²\t__init__"
    assert "".join(SPLIT_PATTERN.findall(s)) == s


@pytest.mark.parametrize("text", ["", "ă", "Împăratul și-a pierdut 3 cai.", "🙂 emoji și tab\tși\n\nlinii",
                                  "cuvânt-nevăzut-vreodată_xyz", f"doc 1{EOT}doc 2"])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_merges_compress(tok):
    sample = "Era odată ca niciodată, într-o țară"
    assert len(tok.encode(sample)) < len(sample.encode("utf-8")) / 2


def test_special_token(tok):
    ids = tok.encode(f"a{EOT}b")
    assert tok.special[EOT] in ids
    assert tok.special[EOT] not in tok.encode(f"a{EOT}b", allow_special=False)


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.encode(TEXT[:300]) == tok.encode(TEXT[:300])
    assert tok2.vocab_size == tok.vocab_size <= 400   # may stop early when nothing is left to merge


# ---------------------------------------------------------------------- model
@pytest.mark.parametrize("pos_emb", ["learned", "rope", "none"])
def test_causality(pos_emb):
    """Changing the token at position t must not change the predictions for positions < t."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg(pos_emb=pos_emb)).eval()
    x = torch.randint(0, 300, (1, 20))
    x2 = x.clone()
    x2[0, 12] = (x[0, 12] + 1) % 300
    a, _ = model(x)
    b, _ = model(x2)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-6)
    assert not torch.allclose(a[0, 12:], b[0, 12:], atol=1e-6)


@pytest.mark.parametrize("pos_emb", ["learned", "rope"])
@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_kv_cache_matches_full_forward(pos_emb, impl):
    torch.manual_seed(0)
    model = GPT(tiny_cfg(pos_emb=pos_emb, attn_impl=impl)).eval()
    x = torch.randint(0, 300, (2, 24))
    full, _ = model(x)
    cache = KVCache(model.cfg.n_layer)
    step_logits = [model(x[:, :10], kv_cache=cache)[0]]          # "prefill" with the first 10
    for t in range(10, 24):                                     # then one token at a time
        step_logits.append(model(x[:, t:t + 1], kv_cache=cache)[0])
    assert torch.allclose(full, torch.cat(step_logits, dim=1), atol=1e-5)


def test_manual_attention_equals_sdpa():
    torch.manual_seed(0)
    m1 = GPT(tiny_cfg(attn_impl="manual")).eval()
    m2 = GPT(tiny_cfg(attn_impl="sdpa")).eval()
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 300, (2, 32))
    assert torch.allclose(m1(x)[0], m2(x)[0], atol=1e-5)


def test_initial_loss_is_log_vocab():
    """At initialization the model must be ~uniform: loss ≈ ln(vocab_size)."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    x, y = torch.randint(0, 300, (2, 8, 32))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(300)) < 0.15


def test_weight_tying():
    model = GPT(tiny_cfg(tie_weights=True))
    assert model.lm_head.weight is model.tok_emb.weight
    assert GPT(tiny_cfg(tie_weights=False)).lm_head.weight is not GPT(tiny_cfg()).tok_emb.weight


def test_overfit_single_batch():
    """A healthy model must be able to memorize a single batch -> backprop + optimizer work."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg())
    opt = AdamW(model.optim_groups(0.0), lr=3e-3, betas=(0.9, 0.95))
    x = torch.randint(0, 300, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(150):
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.1


# ---------------------------------------------------------------------- optimizer
def test_adamw_matches_torch():
    torch.manual_seed(0)
    w1 = torch.randn(10, 5, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)
    o1 = AdamW([w1], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    o2 = torch.optim.AdamW([w2], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    target = torch.randn(10, 5)
    for _ in range(50):
        for w, o in ((w1, o1), (w2, o2)):
            ((w - target) ** 2).sum().backward()
            o.step()
            o.zero_grad()
    assert torch.allclose(w1, w2, atol=1e-6)


def test_clip_grad_norm():
    w = torch.zeros(4, requires_grad=True)
    w.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    norm = clip_grad_norm([w], 1.0)
    assert norm.item() == pytest.approx(5.0)
    assert w.grad.norm().item() == pytest.approx(1.0, rel=1e-4)


def test_lr_schedule():
    cfg = SimpleNamespace(lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000, schedule="cosine")
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(550, cfg) == pytest.approx((1e-3 + 1e-4) / 2)
    assert lr_at(1000, cfg) == pytest.approx(1e-4)
    cfg.schedule = "constant"
    assert lr_at(800, cfg) == pytest.approx(1e-3)
