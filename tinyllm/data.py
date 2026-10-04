"""Loading tokens from disk and building batches."""

import json
import os

import numpy as np
import torch


class TokenData:
    """One split (train/val/test), stored on disk as a long run of uint16 and read through memmap."""

    def __init__(self, data_dir, split):
        self.tokens = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.tokens)

    def batch(self, batch_size, block_size, device, generator=None):
        """Pick batch_size random windows; the target y is the input x shifted by one position."""
        ix = torch.randint(len(self.tokens) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.tokens[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.tokens[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)
