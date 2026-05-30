"""
chunker.py
----------
Splits cleaned documents into overlapping token-based chunks.

Output schema per chunk:
{
  "chunk_id":   str,   # e.g. "istanbul_00042"
  "title":      str,   # source article title
  "url":        str,   # source URL
  "heading":    str,   # section heading the chunk came from
  "text":       str,   # chunk text
  "token_count": int,
  "chunk_index": int,  # position within the document
}

Saved to: data/chunks/all_chunks.jsonl  (one JSON object per line)
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 200   # tokens
CHUNK_OVERLAP = 50    # tokens

CLEAN_DIR  = Path(__file__).parent.parent / "data" / "cleaned"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "all_chunks.jsonl"


# ---------------------------------------------------------------------------
# Tokenizer — simple whitespace split (consistent with sentence-transformers)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return text.split()


def detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping token windows."""
    tokens = tokenize(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(detokenize(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk a single cleaned document, preserving section metadata."""
    results = []
    chunk_index = 0

    for section in doc["sections"]:
        heading = section["heading"]
        text = section["text"].strip()
        if not text:
            continue

        for chunk_text_str in chunk_text(text, chunk_size, overlap):
            chunk_text_str = chunk_text_str.strip()
            if not chunk_text_str:
                continue

            # slug-based ID: title_slug + zero-padded index
            title_slug = re.sub(r"\W+", "_", doc["title"]).lower().strip("_")
            chunk_id = f"{title_slug}_{chunk_index:05d}"

            results.append({
                "chunk_id":    chunk_id,
                "title":       doc["title"],
                "url":         doc.get("url", ""),
                "heading":     heading,
                "text":        chunk_text_str,
                "token_count": len(tokenize(chunk_text_str)),
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def chunk_all(
    clean_dir: Path = CLEAN_DIR,
    out_file: Path = CHUNKS_FILE,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(clean_dir.glob("*.json"))
    if not files:
        print("No cleaned files found. Run preprocess.py first.")
        return []

    all_chunks: list[dict] = []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        print(f"  {doc['title']:45s}  → {len(chunks):3d} chunks")

    # Write JSONL
    with out_file.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {out_file.resolve()}")
    return all_chunks


if __name__ == "__main__":
    chunk_all()
