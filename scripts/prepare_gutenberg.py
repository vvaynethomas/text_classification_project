"""Build data/gutenberg_passages.csv: fixed-length passages from the NLTK Project Gutenberg selection.

Each row is one passage (`content`), its author (`Author`) and `<book> #<n>` (`Title`), the same
columns as the short-story CSV, so `classify.py --data data/gutenberg_passages.csv` runs unchanged.

    python scripts/prepare_gutenberg.py [--words 1000] [--max-per-author 60]
"""
import argparse
import csv
import sys
from pathlib import Path

import nltk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from textclf.config import DATA_DIR  # noqa: E402

AUTHORS = {
    "austen": "Jane Austen",
    "blake": "William Blake",
    "bryant": "Sara Cone Bryant",
    "burgess": "Thornton Burgess",
    "carroll": "Lewis Carroll",
    "chesterton": "G. K. Chesterton",
    "edgeworth": "Maria Edgeworth",
    "melville": "Herman Melville",
    "milton": "John Milton",
    "shakespeare": "William Shakespeare",
    "whitman": "Walt Whitman",
}  # bible-kjv.txt is left out: it has no single author.


def body(raw):
    """Drop the leading "[Title by Author Year]" line so the label never appears in the text."""
    return raw.split("]", 1)[1] if raw.startswith("[") else raw


def passages(words, size):
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - size + 1, size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--words", type=int, default=1000, help="words per passage")
    parser.add_argument("--max-per-author", type=int, default=60,
                        help="cap per author, taken evenly across that author's books")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "gutenberg_passages.csv")
    args = parser.parse_args()

    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg

    rows = []
    for prefix, author in AUTHORS.items():
        chunks = []
        for fileid in sorted(f for f in gutenberg.fileids() if f.startswith(prefix + "-")):
            book = fileid.split("-", 1)[1].removesuffix(".txt")
            chunks += [(f"{book} #{i + 1}", text) for i, text in enumerate(passages(body(gutenberg.raw(fileid)).split(), args.words))]
        step = max(1, len(chunks) / args.max_per_author)
        picked = [chunks[int(i * step)] for i in range(min(args.max_per_author, len(chunks)))]
        rows += [{"content": text, "Author": author, "Title": title} for title, text in picked]
        print(f"{author:20s} {len(picked):4d} passages")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["content", "Author", "Title"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} passages to {args.out}")


if __name__ == "__main__":
    main()
