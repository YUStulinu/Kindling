"""Step 6: ablations — train variants of the model with ONE thing changed and measure the effect.

    python ablations.py                     # every variant, 2000 steps each
    python ablations.py --steps 500 --only baseline rope
    python ablations.py --report_only       # just rebuild the table and chart from existing runs

Results: reports/ablations.md, reports/ablations.json, reports/ablations.png
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch

from tinyllm import utf8_console
from tinyllm.data import TokenData
from tinyllm.runtime import autocast_ctx, load_model, pick_device, pick_precision
from tinyllm.tokenizer import BPETokenizer
from evaluate import evaluate_tokens, token_byte_lengths

# name -> (description, train.py arguments that differ from the baseline)
VARIANTS = {
    "baseline":  ("6 layers, 6 heads, learned positions, warmup+cosine, fp16", []),
    "rope":      ("RoPE positions instead of a learned embedding", ["--pos_emb", "rope"]),
    "no_pos":    ("no position information at all", ["--pos_emb", "none"]),
    "one_head":  ("a single attention head (instead of 6)", ["--n_head", "1"]),
    "shallow":   ("2 layers instead of 6", ["--n_layer", "2"]),
    "no_sched":  ("no warmup, constant learning rate", ["--warmup_steps", "0", "--schedule", "constant"]),
    "fp32":      ("no mixed precision (fp32 everywhere)", ["--precision", "fp32"]),
    "sdpa":      ("PyTorch's fused attention kernel instead of the manual one", ["--attn_impl", "sdpa"]),
}


def run_variant(name, extra, args):
    out_dir = os.path.join(args.root, name)
    if os.path.exists(os.path.join(out_dir, "last.pt")):
        print(f"[{name}] already exists, skipping")
        return
    cmd = [sys.executable, "train.py", "--out_dir", out_dir, "--max_steps", str(args.steps),
           "--eval_interval", str(args.eval_interval), "--log_interval", "50", *extra]
    print(f"\n=== [{name}] {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)


def report(args):
    device = pick_device("auto")
    rows = []
    for name, (desc, _) in VARIANTS.items():
        run = os.path.join(args.root, name)
        if not os.path.exists(os.path.join(run, "best.pt")):
            continue
        with open(os.path.join(run, "log.json"), encoding="utf-8") as f:
            log = json.load(f)
        model, ckpt = load_model(os.path.join(run, "best.pt"), device)
        data_dir = ckpt["train_cfg"]["data_dir"]
        tok = BPETokenizer.load(os.path.join(data_dir, "tokenizer.json"))
        ctx = autocast_ctx(device, ckpt["log"]["precision"])
        res = evaluate_tokens(model, TokenData(data_dir, "val").tokens, token_byte_lengths(tok), device, ctx,
                              max_tokens=args.eval_tokens)
        rows.append({"name": name, "desc": desc, "params_M": log["params"] / 1e6, **res,
                     "tok_s": statistics.median(r["tok_s"] for r in log["train"][2:]),
                     "time_min": log.get("train_time", 0) / 60, "curve": log["eval"]})
        del model

    if not rows:
        print("No runs found.")
        return
    base = next((r for r in rows if r["name"] == "baseline"), rows[0])
    lines = ["| variant | what changed | params | val loss | perplexity | bits/byte | Δ loss | tokens/s | time |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| `{r['name']}` | {r['desc']} | {r['params_M']:.1f}M | {r['loss']:.3f} | {r['perplexity']:.1f} "
                     f"| {r['bpb']:.3f} | {r['loss'] - base['loss']:+.3f} | {r['tok_s'] / 1000:.0f}k | {r['time_min']:.1f} min |")
    table = "\n".join(lines)
    print("\n" + table)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ablations.md"), "w", encoding="utf-8") as f:
        f.write(f"# Ablations ({args.steps} steps each, evaluated on {rows[0]['tokens']:,} val tokens)\n\n"
                f"{table}\n\n![curves](ablations.png)\n")
    with open(os.path.join(args.out_dir, "ablations.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    plot(rows, base, os.path.join(args.out_dir, "ablations.png"))


def plot(rows, base, path):
    from tinyllm.plotting import INK_2, MUTED, SERIES, plt, setup
    setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    for r in rows:   # the color follows the variant (fixed order from VARIANTS)
        color = SERIES[list(VARIANTS).index(r["name"]) % len(SERIES)]
        ev = r["curve"][1:]  # skip step 0 (loss ~ ln(vocab)), it would wreck the scale
        ax1.plot([e["step"] for e in ev], [e["val_loss"] for e in ev], color=color, label=r["name"])
    ax1.set_title("Validation loss during training")
    ax1.set_xlabel("step")
    ax1.legend(ncol=2)

    order = sorted(rows, key=lambda r: r["loss"])
    deltas = [r["loss"] - base["loss"] for r in order]
    colors = [SERIES[list(VARIANTS).index(r["name"]) % len(SERIES)] for r in order]
    ax2.barh([r["name"] for r in order], deltas, color=colors, height=0.6)
    ax2.axvline(0, color=MUTED, linewidth=1)
    for y, d in enumerate(deltas):   # label always right of the bar / zero axis, so it never covers the name
        ax2.annotate(f"{d:+.3f}", (max(d, 0), y), xytext=(4, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=8, color=INK_2)
    ax2.invert_yaxis()
    ax2.grid(axis="y", visible=False)
    ax2.set_axisbelow(True)
    ax2.set_title("Δ loss vs. baseline (lower = better)")
    ax2.set_xlabel("difference in validation loss (nats/token)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Chart saved: {path}")


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--only", nargs="+", choices=list(VARIANTS), default=None)
    ap.add_argument("--root", default="runs/ablations")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--eval_tokens", type=int, default=None, help="limit the final evaluation (default: all of val)")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        for name in args.only or VARIANTS:
            run_variant(name, VARIANTS[name][1], args)
    report(args)


if __name__ == "__main__":
    main()
