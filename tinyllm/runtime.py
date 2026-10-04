"""Choosing the device and precision, and loading checkpoints."""

from contextlib import nullcontext

import torch

from .config import ModelConfig
from .model import GPT

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def pick_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def pick_precision(name, device):
    """bf16 only on Ampere+ GPUs (RTX 30xx and newer); on Turing (RTX 20xx) it is emulated and slow."""
    if name != "auto":
        return name
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def autocast_ctx(device, precision):
    """Under autocast, matrix multiplications run in 16 bits; sensitive ops (softmax, sums) stay in fp32."""
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=DTYPES[precision])


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
