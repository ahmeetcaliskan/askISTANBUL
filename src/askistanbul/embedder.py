"""Embed chunks with sentence-transformers and build/load a FAISS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import Chunk
from .paths import CHUNKS_FILE, INDEX_DIR
from .config import config


def _import_faiss():
    import faiss
    return faiss


def _load_chunks_jsonl(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


class Embedder:
    """Sentence-transformer encoder + FAISS index helpers.

    Handles the e5 ``query:`` / ``passage:`` prefix convention so callers
    don't have to know which model is in use.
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-base"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        self._is_e5 = "e5" in model_name.lower()
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _prefix(self, is_query: bool) -> str:
        if not self._is_e5:
            return ""
        return "query: " if is_query else "passage: "

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        prefix = self._prefix(is_query)
        inputs = [prefix + t for t in texts] if prefix else list(texts)
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=len(inputs) > 32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build_index(self, chunks: list[Chunk], index_dir: Path = INDEX_DIR):
        faiss = _import_faiss()
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Embedding {len(chunks)} chunks in batches of {self.batch_size} ...")
        embeddings = self.encode([c.text for c in chunks], is_query=False)
        dim = embeddings.shape[1]

        # Inner-product index — cosine similarity since vectors are normalized.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, str(index_dir / "faiss.index"))
        (index_dir / "config.json").write_text(
            json.dumps(
                {"model": self.model_name, "dim": dim, "num_chunks": len(chunks)},
                indent=2,
            ),
            encoding="utf-8",
        )
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

        print(f"\nIndex built: {len(chunks)} vectors × {dim} dims")
        print(f"Saved to: {index_dir.resolve()}")
        return index, chunks

    @staticmethod
    def load_index(index_dir: Path = INDEX_DIR):
        """Load (index, chunks, config) from disk."""
        faiss = _import_faiss()
        index_dir = Path(index_dir)
        config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_dir / "faiss.index"))
        chunks: list[Chunk] = []
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(Chunk.from_dict(json.loads(line)))
        return index, chunks, config


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from chunk data.")
    parser.add_argument("--model", default=config.embedding_model or Embedder.DEFAULT_MODEL)
    parser.add_argument("--chunks", default=str(CHUNKS_FILE))
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    chunks = _load_chunks_jsonl(Path(args.chunks))
    Embedder(model_name=args.model).build_index(chunks, index_dir=Path(args.index_dir))


if __name__ == "__main__":
    _main()
