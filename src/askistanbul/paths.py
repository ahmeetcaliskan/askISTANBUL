"""Central data-directory locations.

Override the data root by setting ``ASKISTANBUL_DATA_DIR`` in your ``.env``
or environment; otherwise defaults to ``<repo>/data``.
"""

from pathlib import Path

from .config import config

# src/askistanbul/paths.py -> parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    Path(config.askistanbul_data_dir).resolve()
    if config.askistanbul_data_dir
    else PROJECT_ROOT / "data"
)

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "all_chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"
