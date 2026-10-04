"""Model and training configuration, plus automatic generation of the CLI arguments."""

import argparse
from dataclasses import dataclass, fields, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 256        # context length (in tokens)
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.1
    pos_emb: str = "learned"     # "learned" | "rope" | "none"
    tie_weights: bool = True     # the input embedding and the output head share weights
    attn_impl: str = "manual"    # "manual" (hand-written) | "sdpa" (PyTorch's fused kernel)


@dataclass
class TrainConfig:
    out_dir: str = "runs/base"
    data_dir: str = "data"
    batch_size: int = 32
    grad_accum: int = 1          # accumulation steps -> effective batch = batch_size * grad_accum
    max_steps: int = 6000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    schedule: str = "cosine"     # "cosine" | "constant"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"      # "auto" | "fp16" | "bf16" | "fp32"
    optimizer: str = "custom"    # "custom" (tinyllm/optim.py) | "torch"
    eval_interval: int = 250
    eval_iters: int = 40
    log_interval: int = 25
    seed: int = 1337
    device: str = "auto"
    resume: bool = False


# Presets: starting points that can be overridden from the command line.
PRESETS = {
    # a few-seconds run, just to check that the pipeline works end to end
    "smoke": dict(n_layer=2, n_head=2, d_model=64, block_size=64, batch_size=8,
                  max_steps=100, eval_interval=50, eval_iters=5, log_interval=10, warmup_steps=10),
    # the main model (~14M parameters), fits comfortably in 6 GB of VRAM
    "small": dict(),
}


def _parse_bool(s):
    if s.lower() in ("1", "true", "yes"):
        return True
    if s.lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {s}")


def add_config_args(parser):
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    for cls in (ModelConfig, TrainConfig):
        group = parser.add_argument_group(cls.__name__)
        for f in fields(cls):
            typ = _parse_bool if f.type is bool else f.type
            group.add_argument(f"--{f.name}", type=typ, default=None, help=f"default: {f.default}")


def configs_from_args(args):
    """Priority order: dataclass defaults < preset < explicit command-line arguments."""
    values = dict(PRESETS[args.preset])
    values.update({k: v for k, v in vars(args).items() if v is not None})
    pick = lambda cls: cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})
    return pick(ModelConfig), pick(TrainConfig)


__all__ = ["ModelConfig", "TrainConfig", "PRESETS", "add_config_args", "configs_from_args", "asdict"]
