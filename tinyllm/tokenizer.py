"""Byte-level BPE (Byte Pair Encoding) tokenizer, written from scratch.

Idea: start from the 256 possible bytes (so any UTF-8 text can be encoded) and repeatedly
merge the most frequent pair of adjacent tokens into a new token.
"""

import heapq
import json
import re
from collections import Counter, defaultdict

# Pre-tokenization: the text is first cut into chunks (words with their leading space, numbers,
# punctuation, whitespace), and BPE is not allowed to merge across their boundaries. That way
# " casa" and "casa." don't produce wasteful tokens like "a." or "a c".
# Every character falls into exactly one alternative: letters | digits | the rest (incl. "_") | whitespace.
SPLIT_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+")

EOT = "<|endoftext|>"   # separator between documents


def merge_pair(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}                                  # (id_a, id_b) -> new_id, in the order learned
        self.special = {}                                 # special text -> id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special)

    # ------------------------------------------------------------------ training
    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256 - 1  # -1 for <|endoftext|>
        # Work on unique words + their frequency: "și" appears 100k times, but we process it once.
        counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        pair_counts = defaultdict(int)   # pair -> how often it occurs (weighted by word frequency)
        where = defaultdict(set)         # pair -> indices of the words that (may) contain it
        for wi, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        # Heap with lazy invalidation: it holds (-count, pair); an entry is valid only if its count
        # matches the current one. Otherwise we would need a max() over the whole dict every merge.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        self.merges, self.vocab, self._cache = {}, {i: bytes([i]) for i in range(256)}, {}
        for k in range(num_merges):
            while heap:
                neg, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg:
                    break
            else:
                break  # no pairs left
            if -neg < 2:
                break
            new_id = 256 + k
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Incremental update: only the words that contain the pair change.
            changed = set()
            for wi in where.pop(pair):
                ids, f = words[wi], freqs[wi]
                if len(ids) < 2:
                    continue
                new_ids = merge_pair(ids, pair, new_id)
                if len(new_ids) == len(ids):
                    continue  # stale entry in `where`: the word no longer contains the pair
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(new_ids, new_ids[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    changed.add(p)
                words[wi] = new_ids
            for p in changed:
                c = pair_counts[p]
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    del pair_counts[p]

            if verbose and (k + 1) % 500 == 0:
                print(f"  merge {k + 1}/{num_merges}: {self.vocab[new_id]!r} (frequency {-neg})")

        self.special = {EOT: 256 + len(self.merges)}
        self.vocab[self.special[EOT]] = EOT.encode("utf-8")

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        # Apply merges in the order they were learned (lower id = learned earlier).
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_pair(ids, pair, self.merges[pair])
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        ids = []
        if allow_special and self.special:
            pattern = "(" + "|".join(re.escape(s) for s in self.special) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special and allow_special:
                ids.append(self.special[part])
            elif part:
                for chunk in SPLIT_PATTERN.findall(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        # errors="replace": a token can be just half of a UTF-8 character (e.g. the first byte of "ș")
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, i):
        return self.vocab[i]

    # ------------------------------------------------------------------ saving
    def save(self, path):
        data = {"merges": [[a, b] for (a, b) in self.merges], "special": self.special}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        for k, (a, b) in enumerate(data["merges"]):
            tok.merges[(a, b)] = 256 + k
            tok.vocab[256 + k] = tok.vocab[a] + tok.vocab[b]
        tok.special = {s: int(i) for s, i in data["special"].items()}
        for s, i in tok.special.items():
            tok.vocab[i] = s.encode("utf-8")
        return tok
