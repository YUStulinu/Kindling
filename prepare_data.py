"""Step 1: download a Romanian corpus, clean it and split it into train/val/test.

Sources:
  * Project Gutenberg — the books available in Romanian (Caragiale, Slavici, ...)
  * Romanian Wikipedia — the "featured" and "good" articles (long, well written, reviewed)

Raw downloads are kept in data/raw/, so a second run downloads nothing.
"""

import argparse
import json
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from tinyllm import utf8_console
from tinyllm.tokenizer import EOT

USER_AGENT = "kindling/0.1 (educational project; python-urllib)"
GUTENBERG_IDS = [35323, 64597, 65565, 62916, 11756]
WIKI_BATCH = 20
MIN_DIACRITICS = 0.02   # normal Romanian prose has ~7-9% letters with diacritics
WIKI_CATEGORIES = ["Categorie:Articole de calitate", "Categorie:Articole bune"]   # featured, good

NBSP, BOM, WORD_JOINER = chr(0x00A0), chr(0xFEFF), chr(0x2060)


def fetch(url, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # the network sometimes fails; retry with a growing pause
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            if isinstance(e, urllib.error.HTTPError) and e.headers.get("Retry-After", "").isdigit():
                wait = int(e.headers["Retry-After"])   # the server tells us how long to wait (429)
            print(f"  error ({e}), retrying in {wait}s...")
            time.sleep(wait)


def wiki_api(**params):
    params.update(action="query", format="json", formatversion=2)
    return json.loads(fetch("https://ro.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)))


# ---------------------------------------------------------------------- cleaning
def normalize(text):
    """NFC + the correct diacritics (ș, ț with a comma, not ş, ţ with a cedilla) + uniform whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans({"ş": "ș", "Ş": "Ș", "ţ": "ț", "Ţ": "Ț",
                                         NBSP: " ", "\r": "", "\t": " ", BOM: ""}))
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_gutenberg(text):
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    if start and end:
        text = text[start.end():end.start()]
    # Gutenberg wraps lines at ~70 characters; rebuild the paragraphs but keep short verse lines
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue
        is_prose = sum(len(l) for l in lines) / len(lines) > 55
        paragraphs.append(" ".join(lines) if is_prose else "\n".join(lines))
    return normalize("\n\n".join(paragraphs))


def clean_wiki(title, text, para_chars=700):
    """Search-index text is one long line: prose, followed by the notes/bibliography list at the end."""
    # the notes list starts with "^ a b ..." -> everything after it is references, not prose
    cut = text.find(" ^ ")
    if cut > 0:
        text = text[:cut]
    text = text.replace(WORD_JOINER, "")
    # "[traduceți]" = "[translate]" link, "[necesită citare]" = "[citation needed]"
    text = re.sub(r"\(\w{2,3}\)\[traduceți\]|\[traduceți\]|\[necesită citare\]|\{\\displaystyle[^}]*\}", "", text)
    # rebuild paragraphs by grouping sentences (otherwise the model would never see "\n\n" in Wikipedia text)
    sentences = re.split(r"(?<=[.!?…])\s+(?=[A-ZĂÂÎȘȚ„«\"(])", text.strip())
    paragraphs, current = [], ""
    for s in sentences:
        current = f"{current} {s}" if current else s
        if len(current) >= para_chars:
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return normalize(title + "\n\n" + "\n\n".join(paragraphs))


def chunk_document(text, max_chars):
    """Split long documents at paragraph boundaries so the train/val/test split stays balanced."""
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------- downloading
def download_gutenberg(raw_dir):
    docs = []
    for book_id in GUTENBERG_IDS:
        path = os.path.join(raw_dir, f"gutenberg_{book_id}.txt")
        if not os.path.exists(path):
            print(f"  downloading Gutenberg #{book_id}")
            # newline="": write the text exactly as received; otherwise on Windows "\r\n" becomes "\r\r\n"
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(fetch(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"))
        with open(path, encoding="utf-8") as f:
            text = clean_gutenberg(f.read())
        # Some editions are written entirely without diacritics ("si", "scoala"). Mixed with correct
        # text, they would teach the model inconsistent spelling, so we leave them out.
        rate = sum(text.count(c) for c in "ăâîșțĂÂÎȘȚ") / max(1, len(text))
        if rate < MIN_DIACRITICS:
            print(f"  #{book_id}: only {rate:.2%} diacritics -> excluded")
            continue
        docs.append(text)
    return docs


def download_wikipedia(raw_dir, delay):
    titles_path = os.path.join(raw_dir, "wikipedia_titles.json")
    if os.path.exists(titles_path):
        with open(titles_path, encoding="utf-8") as f:
            titles = json.load(f)
    else:
        titles = []
        for cat in WIKI_CATEGORIES:
            params = dict(list="categorymembers", cmtitle=cat, cmlimit=500, cmnamespace=0)
            while True:
                d = wiki_api(**params)
                titles += [m["title"] for m in d["query"]["categorymembers"]]
                if "continue" not in d:
                    break
                params.update(d["continue"])
        titles = sorted(set(titles))
        with open(titles_path, "w", encoding="utf-8") as f:
            json.dump(titles, f, ensure_ascii=False)

    # Save batch by batch: if the download is interrupted, it resumes where it left off.
    path = os.path.join(raw_dir, "wikipedia.jsonl")
    articles = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                art = json.loads(line)
                articles[art["title"]] = art
    todo = [t for t in titles if t not in articles]
    if todo:
        print(f"  {len(todo)} of {len(titles)} articles to download from ro.wikipedia.org")
    with open(path, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), WIKI_BATCH):
            batch = todo[b:b + WIKI_BATCH]
            # `cirrusdoc` = the search-index document: plain text for up to 20 articles in a single
            # request. (The `extracts` endpoint returns one article per request, and the server
            # quickly answers with 429 Too Many Requests.)
            pages = wiki_api(prop="cirrusdoc", titles="|".join(batch))["query"]["pages"]
            for page in pages:
                docs = page.get("cirrusdoc") or [{"source": {}}]
                art = {"title": page["title"], "text": docs[0]["source"].get("text", "")}
                f.write(json.dumps(art, ensure_ascii=False) + "\n")
                articles[art["title"]] = art
            f.flush()
            print(f"    {min(b + WIKI_BATCH, len(todo))}/{len(todo)}")
            time.sleep(delay)
    return [clean_wiki(a["title"], a["text"]) for a in articles.values() if len(a["text"]) > 500]


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--chunk_chars", type=int, default=6000)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--delay", type=float, default=2.0, help="pause (s) between requests to Wikipedia")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print("Project Gutenberg:")
    docs = download_gutenberg(raw_dir)
    print("Wikipedia:")
    docs += download_wikipedia(raw_dir, args.delay)

    chunks = [c for d in docs for c in chunk_document(d, args.chunk_chars) if len(c) > 200]
    random.Random(args.seed).shuffle(chunks)
    n_val, n_test = int(len(chunks) * args.val_frac), int(len(chunks) * args.test_frac)
    splits = {"val": chunks[:n_val], "test": chunks[n_val:n_val + n_test], "train": chunks[n_val + n_test:]}

    for name, items in splits.items():
        path = os.path.join(args.data_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"\n{EOT}\n".join(items))
        print(f"{name:5s}: {len(items):5d} documents, {sum(map(len, items)) / 1e6:6.2f}M characters -> {path}")


if __name__ == "__main__":
    main()
