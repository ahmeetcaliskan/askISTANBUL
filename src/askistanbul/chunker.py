"""Token-window chunking with section awareness."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Chunk, CleanedDocument
from .paths import CHUNKS_FILE, CLEAN_DIR


# ---------------------------------------------------------------------------
# Tokenizer helpers — whitespace split. Note: this is coarser than the subword
# tokenizer the sentence-transformer model uses (200 whitespace tokens is
# roughly 260-320 model tokens), but it's deterministic and dependency-free.
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return text.split()


def detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping token windows."""
    tokens = tokenize(text)
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(detokenize(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Slice cleaned documents into overlapping token windows, per section."""

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 50,
        min_section_tokens: int = 0,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_section_tokens = min_section_tokens

    def chunk_document(self, doc: CleanedDocument) -> list[Chunk]:
        results: list[Chunk] = []
        chunk_index = 0
        title_slug = re.sub(r"\W+", "_", doc.title).lower().strip("_")

        for section in doc.sections:
            text = section.text.strip()
            if not text:
                continue
            if self.min_section_tokens and len(tokenize(text)) < self.min_section_tokens:
                continue

            for piece in chunk_text(text, self.chunk_size, self.overlap):
                piece = piece.strip()
                if not piece:
                    continue
                results.append(Chunk(
                    chunk_id=f"{title_slug}_{chunk_index:05d}",
                    title=doc.title,
                    url=doc.url,
                    heading=section.heading,
                    text=piece,
                    token_count=len(tokenize(piece)),
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
        return results

    def chunk_all(
        self,
        clean_dir: Path = CLEAN_DIR,
        out_file: Path = CHUNKS_FILE,
    ) -> list[Chunk]:
        clean_dir = Path(clean_dir)
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        files = sorted(clean_dir.glob("*.json"))
        if not files:
            print(f"No cleaned files in {clean_dir}. Run the preprocessor first.")
            return []

        all_chunks: list[Chunk] = []
        for f in files:
            doc = CleanedDocument.from_dict(json.loads(f.read_text(encoding="utf-8")))
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  {doc.title:45s}  → {len(chunks):3d} chunks")

        with out_file.open("w", encoding="utf-8") as fh:
            for c in all_chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

        print(f"\nTotal chunks: {len(all_chunks)}")
        print(f"Saved to: {out_file.resolve()}")
        return all_chunks


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Chunk cleaned documents into token windows.")
    p.add_argument("--chunk-size", type=int, default=200, help="Tokens per chunk.")
    p.add_argument("--overlap", type=int, default=50, help="Token overlap between adjacent chunks.")
    p.add_argument("--min-section-tokens", type=int, default=0,
                   help="Drop sections shorter than this (in whitespace tokens).")
    args = p.parse_args()

    Chunker(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_section_tokens=args.min_section_tokens,
    ).chunk_all()


if __name__ == "__main__":
    _main()
