"""Offline indexing orchestration: scrape → preprocess → chunk → embed.

The online query-time pipeline (``RAGPipeline``) lives in ``rag.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chunker import Chunker
from .embedder import Embedder
from .models import Chunk
from .paths import CHUNKS_FILE, INDEX_DIR
from .preprocess import Preprocessor
from .scraper import Scraper


class Pipeline:
    """Run scrape → preprocess → chunk → embed end-to-end."""

    def __init__(
        self,
        scraper: Optional[Scraper] = None,
        preprocessor: Optional[Preprocessor] = None,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        chunks_file: Path = CHUNKS_FILE,
        index_dir: Path = INDEX_DIR,
    ):
        self.scraper = scraper or Scraper()
        self.preprocessor = preprocessor or Preprocessor()
        self.chunker = chunker or Chunker()
        # Embedder is lazily constructed — model loading is expensive and
        # callers may want to skip the embed step entirely (skip_embed=True).
        self.embedder = embedder
        self.chunks_file = Path(chunks_file)
        self.index_dir = Path(index_dir)

    def run(self, skip_scrape: bool = False, skip_embed: bool = False) -> list[Chunk]:
        if not skip_scrape:
            print("\n=== 1. Scrape ===")
            self.scraper.scrape_all()

        print("\n=== 2. Preprocess ===")
        self.preprocessor.preprocess_all()

        print("\n=== 3. Chunk ===")
        chunks = self.chunker.chunk_all(out_file=self.chunks_file)

        if skip_embed:
            return chunks

        print("\n=== 4. Embed ===")
        if self.embedder is None:
            self.embedder = Embedder()
        self.embedder.build_index(chunks, index_dir=self.index_dir)
        return chunks


def _main() -> None:
    """Run the full offline indexing pipeline: scrape → preprocess → chunk → embed."""
    import argparse

    p = argparse.ArgumentParser(description="Build the full RAG index end-to-end.")
    p.add_argument("--skip-scrape", action="store_true",
                   help="Reuse cached raw data; skip the scraping step.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Stop after chunking; don't build the FAISS index.")
    args = p.parse_args()

    Pipeline().run(skip_scrape=args.skip_scrape, skip_embed=args.skip_embed)


if __name__ == "__main__":
    _main()
