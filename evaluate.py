"""Step 5: evaluate the model.

    python evaluate.py perplexity --ckpt runs/base/best.pt --split test
    python evaluate.py curves --runs runs/base            # learning curves -> reports/
    python evaluate.py samples --ckpt runs/base/best.pt   # text samples from fixed prompts -> reports/

Metrics:
  * loss (cross-entropy, in nats/token) and perplexity = exp(loss): "how many tokens the model is
    hesitating between, on average". A model that guesses uniformly has perplexity = vocabulary size.
  * bits-per-byte (bpb): the total loss divided by the number of bytes of text. It does not depend on
    the tokenizer, so models with different vocabularies can be compared fairly.
  * top-1 accuracy: how often the most likely token is actually the correct one.
  * unigram baseline: a "model" that only knows how frequent each token is in train.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from tinyllm import utf8_console
from tinyllm.data import TokenData
from tinyllm.runtime import autocast_ctx, load_model, pick_device, pick_precision
from tinyllm.tokenizer import EOT, BPETokenizer

# Romanian prompts (the model only knows Romanian): "Once upon a time", "Romania is a country",
# "In the year 1918,", "Mihai Eminescu was", "Love"
PROMPTS = [
    "A fost odată ca niciodată",
    "România este o țară",
    "În anul 1918,",
    "Mihai Eminescu a fost",
    "Dragostea",
]


def token_byte_lengths(tok):
    lengths = np.array([len(tok.vocab[i]) for i in range(tok.vocab_size)], dtype=np.float64)
    lengths[tok.special[EOT]] = 1  # the document separator counts as one "\n", not 13 bytes
    return lengths


@torch.no_grad()
def evaluate_tokens(model, tokens, byte_len, device, ctx, stride=None, batch_size=32, max_tokens=None):
    """Sliding-window evaluation: every token is predicted exactly once, with at least
    block_size - stride tokens of context (except in the first window)."""
    T = model.cfg.block_size
    stride = stride or T // 2
    data = torch.from_numpy(np.asarray(tokens[: max_tokens or len(tokens)], dtype=np.int64))
    starts = list(range(0, len(data) - T - 1, stride))
    nll = correct = count = n_bytes = 0.0
    for b in range(0, len(starts), batch_size):
        chunk = starts[b:b + batch_size]
        x = torch.stack([data[s:s + T] for s in chunk]).to(device)
        y = torch.stack([data[s + 1:s + T + 1] for s in chunk]).to(device)
        for j, s in enumerate(chunk):
            if s > 0:
                y[j, :T - stride] = -1   # these positions were already scored by the previous window
        with ctx:
            logits, _ = model(x)
        losses = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                 ignore_index=-1, reduction="none")
        valid = y.view(-1) >= 0
        nll += losses[valid].sum().item()
        correct += (logits.view(-1, logits.size(-1)).argmax(-1)[valid] == y.view(-1)[valid]).sum().item()
        count += valid.sum().item()
        n_bytes += byte_len[y.view(-1)[valid].cpu().numpy()].sum()
    loss = nll / count
    return {"loss": loss, "perplexity": math.exp(loss), "bpb": nll / math.log(2) / n_bytes,
            "accuracy": correct / count, "tokens": int(count)}


def unigram_baseline(data_dir, split, byte_len):
    train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    target = np.fromfile(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16)
    counts = np.bincount(train, minlength=len(byte_len)) + 1.0     # +1 (Laplace smoothing) to avoid log(0)
    logp = np.log(counts / counts.sum())
    nll = -logp[target].sum()
    loss = nll / len(target)
    return {"loss": loss, "perplexity": math.exp(loss), "bpb": nll / math.log(2) / byte_len[target].sum()}


def cmd_perplexity(args):
    device = pick_device(args.device)
    model, ckpt = load_model(args.ckpt, device)
    data_dir = ckpt["train_cfg"]["data_dir"]
    tok = BPETokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    byte_len = token_byte_lengths(tok)
    ctx = autocast_ctx(device, pick_precision("auto", device))
    res = evaluate_tokens(model, TokenData(data_dir, args.split).tokens, byte_len, device, ctx,
                          stride=args.stride, max_tokens=args.max_tokens)
    base = unigram_baseline(data_dir, args.split, byte_len)
    print(f"Model:   {args.ckpt} (step {ckpt['step']}), split: {args.split}, {res['tokens']:,} tokens")
    print(f"{'':10s} {'loss':>8s} {'perplexity':>10s} {'bits/byte':>10s} {'top-1 acc':>10s}")
    print(f"{'model':10s} {res['loss']:8.4f} {res['perplexity']:10.2f} {res['bpb']:10.4f} {res['accuracy']:10.2%}")
    print(f"{'unigram':10s} {base['loss']:8.4f} {base['perplexity']:10.2f} {base['bpb']:10.4f} {'':>10s}")
    print(f"{'uniform':10s} {math.log(tok.vocab_size):8.4f} {tok.vocab_size:10.2f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"split": args.split, "model": res, "unigram": base, "ckpt": args.ckpt}, f, indent=2)
    return res


def plot_curves(run_dir, out_path):
    from tinyllm.plotting import INK_2, SERIES, direct_label, plt, setup, smooth
    setup()
    with open(os.path.join(run_dir, "log.json"), encoding="utf-8") as f:
        log = json.load(f)
    tr, ev = log["train"], log["eval"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"width_ratios": [2, 1, 1]})

    ax = axes[0]
    steps = [r["step"] for r in tr]
    ax.plot(steps, [r["loss"] for r in tr], color=SERIES[0], alpha=0.25, linewidth=1)
    ax.plot(steps, smooth([r["loss"] for r in tr], 0.7), color=SERIES[0], label="train (per batch, smoothed)")
    ax.plot([e["step"] for e in ev], [e["val_loss"] for e in ev], color=SERIES[1], label="validation")
    best = min(ev, key=lambda e: e["val_loss"])
    direct_label(ax, best["step"], best["val_loss"], f"min val {best['val_loss']:.3f}", SERIES[1])
    ax.set_title("Loss (cross-entropy, nats/token)")
    ax.set_xlabel("step")
    ax.set_ylim(top=min(max(r["loss"] for r in tr), ev[0]["val_loss"]) + 0.2)
    ax.legend(loc="upper right")

    axes[1].plot(steps, [r["lr"] for r in tr], color=SERIES[0])
    axes[1].set_title("Learning rate")
    axes[1].set_xlabel("step")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[2].plot(steps, [r["grad_norm"] for r in tr], color=SERIES[0], alpha=0.3, linewidth=1)
    axes[2].plot(steps, smooth([r["grad_norm"] for r in tr], 0.7), color=SERIES[0])
    axes[2].set_title("Gradient norm")
    axes[2].set_xlabel("step (before clipping at 1.0)")

    name = os.path.basename(os.path.normpath(run_dir))
    fig.suptitle(f"{name}: {log['params_total'] / 1e6:.1f}M parameters ({log['params'] / 1e6:.1f}M non-embedding), {log['precision']}", x=0.01, ha="left",
                 color=INK_2, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Chart saved: {out_path}")


def cmd_curves(args):
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        plot_curves(run, os.path.join(args.out_dir, f"curves_{name}.png"))


def cmd_samples(args):
    from generate import generate
    device = pick_device(args.device)
    model, ckpt = load_model(args.ckpt, device)
    tok = BPETokenizer.load(os.path.join(ckpt["train_cfg"]["data_dir"], "tokenizer.json"))
    ctx = autocast_ctx(device, pick_precision("auto", device))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    lines = [f"# Generated samples\n\nCheckpoint: `{args.ckpt}` (step {ckpt['step']}), "
             f"temperature={args.temperature}, top_k=50, top_p=0.95, seed={args.seed}\n"]
    for prompt in PROMPTS:
        idx = torch.tensor([tok.encode(prompt)], device=device)
        with ctx:
            out = list(generate(model, idx, args.max_new_tokens, args.temperature, 50, 0.95,
                                stop_id=tok.special[EOT], generator=gen))
        text = prompt + tok.decode([t for t in out if t != tok.special[EOT]])
        print(f"--- {prompt}\n{text}\n")
        lines.append(f"## «{prompt}»\n\n> " + text.replace("\n", "\n> ") + "\n")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {args.out}")


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("perplexity")
    p.add_argument("--ckpt", default="runs/base/best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=None, help="default: block_size/2")
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--out", default=None, help="save the results as JSON")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("curves")
    p.add_argument("--runs", nargs="+", default=["runs/base"])
    p.add_argument("--out_dir", default="reports")

    p = sub.add_parser("samples")
    p.add_argument("--ckpt", default="runs/base/best.pt")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/samples.md")
    p.add_argument("--device", default="auto")

    args = ap.parse_args()
    {"perplexity": cmd_perplexity, "curves": cmd_curves, "samples": cmd_samples}[args.cmd](args)


if __name__ == "__main__":
    main()
