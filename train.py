"""Step 3: train the model.

Examples:
    python train.py                                   # the main model (preset "small")
    python train.py --preset smoke --out_dir runs/smoke
    python train.py --pos_emb rope --out_dir runs/rope --max_steps 2000
"""

import argparse
import json
import math
import os
import time

import torch

from tinyllm import utf8_console
from tinyllm.config import add_config_args, asdict, configs_from_args
from tinyllm.data import TokenData, load_meta
from tinyllm.model import GPT
from tinyllm.optim import AdamW, clip_grad_norm, lr_at
from tinyllm.runtime import autocast_ctx, pick_device, pick_precision


@torch.no_grad()
def estimate_loss(model, data, mcfg, tcfg, device, ctx):
    """Mean loss over eval_iters batches. Val always uses the same batches -> comparable curves."""
    model.eval()
    out = {}
    for split, d in data.items():
        gen = torch.Generator().manual_seed(1234) if split == "val" else None
        losses = []
        for _ in range(tcfg.eval_iters):
            x, y = d.batch(tcfg.batch_size, mcfg.block_size, device, gen)
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    mcfg, tcfg = configs_from_args(ap.parse_args())

    torch.manual_seed(tcfg.seed)
    device = pick_device(tcfg.device)
    precision = pick_precision(tcfg.precision, device)
    ctx = autocast_ctx(device, precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    meta = load_meta(tcfg.data_dir)
    mcfg.vocab_size = meta["vocab_size"]
    data = {"train": TokenData(tcfg.data_dir, "train"), "val": TokenData(tcfg.data_dir, "val")}

    model = GPT(mcfg).to(device)
    groups = model.optim_groups(tcfg.weight_decay)
    betas = (tcfg.beta1, tcfg.beta2)
    if tcfg.optimizer == "custom":
        optimizer = AdamW(groups, lr=tcfg.lr, betas=betas)
    else:
        optimizer = torch.optim.AdamW(groups, lr=tcfg.lr, betas=betas, fused=device.type == "cuda")
    # GradScaler (fp16 only): multiplies the loss by a large factor before backward so that small
    # gradients don't underflow to 0 in fp16; divides them back before the optimizer step.
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))

    os.makedirs(tcfg.out_dir, exist_ok=True)
    log = {"model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg), "precision": precision,
           "params": model.num_params(), "params_total": model.num_params(non_embedding=False),
           "train": [], "eval": []}
    start_step, best_val = 0, float("inf")
    last_path, best_path = os.path.join(tcfg.out_dir, "last.pt"), os.path.join(tcfg.out_dir, "best.pt")
    if tcfg.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step, best_val, log = ckpt["step"], ckpt["best_val"], ckpt["log"]
        print(f"Resuming from step {start_step}")

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.block_size
    print(f"Device: {device} | precision: {precision} | optimizer: {tcfg.optimizer}")
    print(f"Parameters: {log['params'] / 1e6:.2f}M (non-embedding), {log['params_total'] / 1e6:.2f}M total")
    print(f"Data: {len(data['train']):,} train tokens | {tokens_per_step:,} tokens/step | "
          f"{tcfg.max_steps * tokens_per_step / len(data['train']):.1f} epochs")

    def save(path, step):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg), "step": step, "best_val": best_val,
                    "log": log}, path)

    model.train()
    t_last, train_time, steps_since = time.time(), 0.0, 0
    for step in range(start_step, tcfg.max_steps + 1):
        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ------------------------------------------------------------ evaluation + checkpoint
        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps:
            train_time += time.time() - t_last
            losses = estimate_loss(model, data, mcfg, tcfg, device, ctx)
            log["eval"].append({"step": step, "tokens": step * tokens_per_step, "time": train_time,
                                "train_loss": losses["train"], "val_loss": losses["val"]})
            mark = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(best_path, step)
                mark = " *"
            print(f"[eval] step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| val ppl {math.exp(losses['val']):7.2f}{mark}")
            with open(os.path.join(tcfg.out_dir, "log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f)
            t_last, steps_since = time.time(), 0
        if step == tcfg.max_steps:
            break

        # ------------------------------------------------------------ one training step
        for _ in range(tcfg.grad_accum):
            x, y = data["train"].batch(tcfg.batch_size, mcfg.block_size, device)
            with ctx:                                     # forward pass in mixed precision
                _, loss = model(x, y)
            scaler.scale(loss / tcfg.grad_accum).backward()   # backpropagation: the gradient of every parameter
        scaler.unscale_(optimizer)                        # the real (unscaled) gradients, for clipping
        grad_norm = clip_grad_norm(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)                            # skips the step if inf/NaN appeared in fp16
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        steps_since += 1

        if step % tcfg.log_interval == 0:
            loss_val = loss.item()                         # .item() syncs the GPU -> the timing is correct
            dt = time.time() - t_last
            tok_s = steps_since * tokens_per_step / max(dt, 1e-9)
            train_time += dt
            t_last, steps_since = time.time(), 0
            log["train"].append({"step": step, "loss": loss_val, "lr": lr, "grad_norm": grad_norm.item(),
                                 "tok_s": tok_s})
            print(f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e} | |g| {grad_norm.item():5.2f} "
                  f"| {tok_s / 1000:5.1f}k tok/s")

    log["train_time"] = train_time
    save(last_path, tcfg.max_steps)
    with open(os.path.join(tcfg.out_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f)
    print(f"Done. Best val loss: {best_val:.4f} (perplexity {math.exp(best_val):.2f}) -> {best_path}")


if __name__ == "__main__":
    main()
