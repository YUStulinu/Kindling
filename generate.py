"""Step 4: generate text with a trained model.

Examples:
    python generate.py --prompt "A fost odată ca niciodată"
    python generate.py --interactive --temperature 0.7
    python generate.py --prompt "România este" --no_cache     # compare the speed without the KV cache
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

from tinyllm import utf8_console
from tinyllm.model import KVCache
from tinyllm.runtime import autocast_ctx, load_model, pick_device, pick_precision
from tinyllm.tokenizer import EOT, BPETokenizer


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Pick the next token from the model's distribution.

    temperature < 1 sharpens the distribution (safer, more repetitive text), > 1 flattens it (more chaotic).
    top_k keeps only the k most likely tokens; top_p keeps the smallest set whose cumulative
    probability exceeds p ("nucleus sampling").
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p   # the token that crosses the threshold is kept
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
             use_cache=True, stop_id=None, generator=None):
    """A generator: yields tokens one at a time (for live display)."""
    block = model.cfg.block_size
    cache = KVCache(model.cfg.n_layer) if use_cache else None
    logits, _ = model(idx[:, -block:], kv_cache=cache, last_only=True)
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, nxt], dim=1)
        yield nxt.item()
        if stop_id is not None and nxt.item() == stop_id:
            return
        if not use_cache:
            logits, _ = model(idx[:, -block:], last_only=True)        # recompute the whole context
        elif cache.length < block:
            logits, _ = model(nxt, kv_cache=cache, last_only=True)    # process only the new token
        else:
            # the cache is full: rebuild it from the last half of the context (sliding window)
            cache = KVCache(model.cfg.n_layer)
            logits, _ = model(idx[:, -(block // 2):], kv_cache=cache, last_only=True)


def stream(model, tok, prompt, args, device, ctx, generator):
    ids = tok.encode(prompt) if prompt else [tok.special[EOT]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out, printed, n, t0 = [], 0, 0, time.time()
    sys.stdout.write(prompt)
    with ctx:
        for t in generate(model, idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
                          use_cache=not args.no_cache, stop_id=tok.special[EOT], generator=generator):
            n += 1
            if t == tok.special[EOT]:
                break
            out.append(t)
            text = tok.decode(out)
            if not text.endswith("�"):  # don't print half of a UTF-8 character
                sys.stdout.write(text[printed:])
                sys.stdout.flush()
                printed = len(text)
    dt = time.time() - t0
    print(f"\n\n[{n} tokens in {dt:.2f}s = {n / dt:.0f} tokens/s, cache={'off' if args.no_cache else 'on'}]")


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_model(args.ckpt, device)
    tok = BPETokenizer.load(os.path.join(ckpt["train_cfg"]["data_dir"], "tokenizer.json"))
    ctx = autocast_ctx(device, pick_precision("auto", device))
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.interactive:
        print("Type the beginning of a text in Romanian, then press Enter. Type 'exit' (or Ctrl+C) to quit.")
        while True:
            try:
                prompt = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            # Empty lines are ignored instead of quitting: when a command is pasted into the terminal
            # together with its trailing newline, that stray Enter arrives here before the user types.
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break
            stream(model, tok, prompt, args, device, ctx, generator)
    else:
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"--- sample {i + 1} ---")
            stream(model, tok, args.prompt, args, device, ctx, generator)


if __name__ == "__main__":
    main()
