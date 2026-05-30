"""
embedder.py
-----------
Embeds all chunks with sentence-transformers and builds a FAISS index.

Saved to:
  data/index/faiss.index     — the FAISS flat L2 index
  data/index/chunks.jsonl    — chunk metadata in the same row order as the index
  data/index/config.json     — model name and dimension, for consistency checks

Usage:
  python src/embedder.py              # build index from data/chunks/all_chunks.jsonl
  python src/embedder.py --model multilingual-e5-base   # alternative model
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Lazy imports — only needed at build time, not when index is loaded
def _import_faiss():
    import faiss
    return faiss

def _import_st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "intfloat/multilingual-e5-base"  
BATCH_SIZE    = 64

CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks" / "all_chunks.jsonl"
INDEX_DIR   = Path(__file__).parent.parent / "data" / "index"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_chunks(chunks_file: Path = CHUNKS_FILE) -> list[dict]:
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}\nRun chunker.py first.")
    chunks = []
    with chunks_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(
    chunks: list[dict],
    model_name: str = DEFAULT_MODEL,
    index_dir: Path = INDEX_DIR,
    batch_size: int = BATCH_SIZE,
):
    faiss = _import_faiss()
    SentenceTransformer = _import_st()

    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)

    # intfloat/multilingual-e5-* models require "passage: " prefix on documents
    # For other models the prefix is harmless (stripped by the tokenizer)
    is_e5 = "e5" in model_name.lower()
    prefix = "passage: " if is_e5 else ""
    texts = [prefix + c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks in batches of {batch_size} ...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine sim via inner product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]

    # Inner product index (cosine similarity, since vectors are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "config.json").write_text(
        json.dumps({"model": model_name, "dim": dim, "num_chunks": len(chunks)}, indent=2),
        encoding="utf-8",
    )
    # Save chunk metadata in index order
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nIndex built: {len(chunks)} vectors × {dim} dims")
    print(f"Saved to: {index_dir.resolve()}")
    return index, chunks


# ---------------------------------------------------------------------------
# Load (used by retriever.py)
# ---------------------------------------------------------------------------

def load_index(index_dir: Path = INDEX_DIR):
    """Load the FAISS index and chunk metadata from disk."""
    faiss = _import_faiss()

    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "faiss.index"))
    chunks = []
    with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return index, chunks, config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from chunk data.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformer model name")
    parser.add_argument("--chunks", default=str(CHUNKS_FILE), help="Path to all_chunks.jsonl")
    parser.add_argument("--index-dir", default=str(INDEX_DIR), help="Output directory for index")
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    build_index(chunks, model_name=args.model, index_dir=Path(args.index_dir))
