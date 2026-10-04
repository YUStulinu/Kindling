"""Step 2: train the BPE tokenizer on train.txt and turn every split into a .bin file (uint16)."""

import argparse
import json
import os
import time

import numpy as np

from tinyllm import utf8_console
from tinyllm.tokenizer import BPETokenizer


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=8192)
    args = ap.parse_args()
    assert args.vocab_size <= 65536, "tokens are stored as uint16"

    with open(os.path.join(args.data_dir, "train.txt"), encoding="utf-8") as f:
        train_text = f.read()

    # The tokenizer sees ONLY train, so the evaluation on test stays honest.
    print(f"Training BPE with a vocabulary of {args.vocab_size} on {len(train_text) / 1e6:.1f}M characters...")
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(train_text, args.vocab_size, verbose=True)
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    tok.save(tok_path)
    print(f"Done in {time.time() - t0:.0f}s -> {tok_path}")

    longest = sorted(range(256, tok.vocab_size), key=lambda i: -len(tok.vocab[i]))[:15]
    print("Longest tokens:", [tok.decode([i]) for i in longest])

    meta = {"vocab_size": tok.vocab_size, "tokenizer": "tokenizer.json", "splits": {}}
    for split in ("train", "val", "test"):
        with open(os.path.join(args.data_dir, f"{split}.txt"), encoding="utf-8") as f:
            text = f.read()
        ids = np.array(tok.encode(text), dtype=np.uint16)
        ids.tofile(os.path.join(args.data_dir, f"{split}.bin"))
        n_bytes = len(text.encode("utf-8"))
        meta["splits"][split] = {"tokens": len(ids), "bytes": n_bytes}
        print(f"{split:5s}: {len(ids):>10,} tokens  ({n_bytes / len(ids):.2f} bytes/token)")

    with open(os.path.join(args.data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
