# askISTANBUL

A retrieval-augmented question-answering system for Istanbul travel.

Ask it natural-language questions — *"How do I get from the airport to Taksim Square?"*, *"What are the best restaurants near Sultanahmet?"*, *"Where can I find rooftop bars in Beyoğlu?"* — and it produces factually-grounded, source-attributed answers retrieved from Wikivoyage's curated Istanbul travel articles.

Built for **CS 455 / CS 555 Large Language Models** (Sabancı University, Spring 2025/2026). See `CS455_555_Proposal_Caliskan_Sahinbas.pdf` for the project proposal.

---

## How it works

Two pipelines: an **offline indexing** pipeline that runs once to prepare the corpus, and an **online query** pipeline that runs per question.

```
                              OFFLINE INDEXING

   Wikivoyage  ──►  Scraper  ──►  Preprocessor  ──►  Chunker  ──►  Embedder  ──►  FAISS
   (~20 pages)       raw/          cleaned/           chunks/        vectors        index/


                              ONLINE QUERY ANSWERING

                   ┌─►  DenseRetriever (FAISS, semantic)   ─┐
       question ───┤                                         ├─►  [optional]   ─►  RAGPipeline ─►  LLM  ─►  grounded
                   └─►  BM25Retriever  (sparse, lexical)    ─┘   RerankingRetriever      │            answer + citations
                                                                  (cross-encoder)        │
                                                                                         ▼
                                                                              BaseLLMClient (Ollama, …)
```

Every stage is a standalone class and a CLI command. You can run the whole pipeline end-to-end (`askistanbul-index`) or any stage individually for experiments.

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install
pip install -U pip setuptools
pip install -e . -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env (at minimum, set ASKISTANBUL_CONTACT to your email)

# 4. Build the index (scrape → clean → chunk → embed)
askistanbul-index

# 5. Query
askistanbul-ask "How do I get to the airport from Taksim?"
askistanbul-repl                   # interactive mode
```

---

## Setup in detail

### Requirements

- Python 3.10 or later (Python 3.14 tested)
- ~500 MB disk for the embedding model and FAISS index
- (Optional) [Ollama](https://ollama.com/) for local LLM generation. The retrieval part works standalone without an LLM.

### Install the package

```bash
pip install -e . -r requirements.txt
```

`pip install -e .` puts the package in editable mode — your code edits in `src/askistanbul/` take effect immediately with no reinstall. `-r requirements.txt` installs the third-party libraries (FAISS, sentence-transformers, requests, rank-bm25, etc.).

### Pull an LLM (for generation)

```bash
ollama pull qwen2.5:7b           # default; ~4.7 GB
```

The REPL works without an LLM (retrieval-only mode); generation just stays empty.

---

## Configuration

All configuration lives in a project-root `.env` file. Copy `.env.example` to `.env` and edit. Every variable has a sensible default — only `ASKISTANBUL_CONTACT` is worth changing on first run.

| Variable | Default | Purpose |
|---|---|---|
| `ASKISTANBUL_CONTACT` | `askistanbul@example.com` | Email in the User-Agent the scraper sends to Wikivoyage |
| `ASKISTANBUL_DATA_DIR` | `<repo>/data` | Override the data directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer used by the dense retriever |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker (when enabled with `--rerank`) |
| `RERANKER_FETCH_K` | `20` | How many candidates the reranker pulls before narrowing |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon address |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Local LLM for answer generation |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | — | Optional fallback (per proposal) |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | — | Optional fallback (per proposal) |

Empty values in `.env` (e.g. `RERANKER_FETCH_K=`) fall back to the defaults — they don't crash the loader.

---

## Building the index

The offline pipeline has four stages, each runnable in isolation, each writing to `data/<stage>/`.

| Command | What it does | Reads | Writes |
|---|---|---|---|
| `askistanbul-scrape` | Fetch ~16 Istanbul-related Wikivoyage articles via the MediaWiki API. Caches on disk; reruns are idempotent. | (network) | `data/raw/*.json` |
| `askistanbul-preprocess` | Strip wikitext markup, preserve POI templates (`{{see}}`, `{{eat}}`, `{{sleep}}`, …), split into sections. | `data/raw/*.json` | `data/cleaned/*.json` |
| `askistanbul-chunk` | Split sections into overlapping token windows (default 200 tokens, 50 overlap). | `data/cleaned/*.json` | `data/chunks/all_chunks.jsonl` |
| `askistanbul-embed` | Embed each chunk with a sentence-transformer; write a FAISS inner-product index. | `data/chunks/all_chunks.jsonl` | `data/index/{faiss.index, chunks.jsonl, config.json}` |

Each accepts overrides — e.g. for Week-4 ablations:

```bash
askistanbul-chunk --chunk-size 300 --overlap 75 --min-section-tokens 80
askistanbul-embed --model intfloat/multilingual-e5-base --index-dir data/index_e5
```

Or run the whole pipeline:

```bash
askistanbul-index                  # all four stages
askistanbul-index --skip-scrape    # reuse cached raw, redo cleaning onward
askistanbul-index --skip-embed     # stop after chunking (cheap iterate)
```

---

## Querying

### One-shot

```bash
askistanbul-ask "best rooftop bars in Beyoglu"
askistanbul-ask "Hagia Sophia opening hours" --k 3 --show-text
askistanbul-ask "how to get to the airport" --method bm25
askistanbul-ask "rooftop bars" --rerank --fetch-k 30      # add cross-encoder rerank
```

### Interactive REPL

```bash
askistanbul-repl
askistanbul-repl --rerank --method bm25 --show-text
```

Inside the REPL:

```
ask> best rooftop bars in Beyoglu
ask> :k 10                 # change top-k
ask> :method bm25          # switch retrieval backend
ask> :rerank on            # toggle cross-encoder reranking
ask> :fetch-k 30           # over-fetch count for reranker
ask> :show on              # show chunk text
ask> :help                 # see all commands
ask> :q                    # exit
```

The REPL prints retrieved chunks with their scores and source attributions. If Ollama is reachable, it also prints a generated answer above the chunk list.

---

## Project structure

```
askISTANBUL/
├── README.md
├── pyproject.toml              # package metadata + [project.scripts] CLI entries
├── requirements.txt            # third-party deps (FAISS, sentence-transformers, …)
├── .env / .env.example         # local config (.env is gitignored)
├── data/                       # generated; see "Building the index" above
│   ├── raw/                    # Wikivoyage wikitext (one .json per page)
│   ├── cleaned/                # plain-text articles split into sections
│   ├── chunks/all_chunks.jsonl # token-window chunks with metadata
│   └── index/                  # FAISS index + chunk lookup + model config
└── src/askistanbul/
    ├── __init__.py             # re-exports the public API
    ├── config.py               # .env loading + Config dataclass singleton
    ├── paths.py                # canonical data/ directory locations
    ├── models.py               # core dataclasses (Chunk, Section, …)
    ├── scraper.py              # MediaWiki API scraping
    ├── preprocess.py           # wikitext cleaning + section splitting
    ├── chunker.py              # token-window chunking
    ├── embedder.py             # sentence-transformer + FAISS index
    ├── retriever.py            # DenseRetriever, BM25Retriever, Retriever ABC
    ├── reranker.py             # cross-encoder reranker (two-stage retrieval)
    ├── rag.py                  # RAGPipeline + interactive REPL
    ├── pipeline.py             # offline indexing facade (Pipeline)
    └── generator/              # LLM clients (hexagonal architecture)
        ├── port/BaseLLMClient.py        # abstract LLM client interface
        ├── adapter/OllamaClient.py      # concrete Ollama implementation
        └── factory/LLMClientFactory.py  # picks the right client at runtime
```

### What each script does

#### `config.py`
Calls `load_dotenv()` once at import, then reads every env var into a frozen `Config` dataclass. The module-level `config` singleton is imported everywhere else, so no other module ever calls `os.getenv` directly. Empty values in `.env` fall back to defaults via `_str_env` / `_int_env` helpers.

#### `paths.py`
Resolves the canonical locations of `data/raw/`, `data/cleaned/`, `data/chunks/`, `data/index/`. Override the root with `ASKISTANBUL_DATA_DIR` in `.env`.

#### `models.py`
Plain `@dataclass` types that mirror the on-disk JSON schemas: `RawPage`, `Section`, `CleanedDocument`, `Chunk`, `RetrievalResult`. Every dataclass has `to_dict()` / `from_dict()` so JSON serialization is symmetric and round-trippable.

#### `scraper.py` — `Scraper`
Fetches Wikivoyage articles via the MediaWiki API. Configurable contact email (User-Agent), rate limit, and retry. Idempotent — already-fetched pages are loaded from disk.

#### `preprocess.py` — `WikitextCleaner`, `SectionSplitter`, `Preprocessor`
- `WikitextCleaner` strips templates, wikilinks, HTML, tables, and formatting markers. Importantly, it **preserves Wikivoyage listing templates** (`{{see}}`, `{{eat}}`, `{{sleep}}`, …) by rendering their parameters inline — these are where POI names, addresses, phones, and hours live.
- `SectionSplitter` splits cleaned text on heading boundaries (`== Heading ==`, …).
- `Preprocessor` composes the two and handles disk I/O.

#### `chunker.py` — `Chunker`
Splits cleaned documents into overlapping token windows. Respects section boundaries (each section is chunked separately), so a chunk is always from one section. Default 200 tokens with 50-token overlap; both configurable.

#### `embedder.py` — `Embedder`
Loads a sentence-transformer once, encodes texts in batches, builds a FAISS inner-product index (cosine similarity on normalized vectors). Knows about the `intfloat/e5-*` family's `query:` / `passage:` prefix convention and applies it transparently.

#### `retriever.py` — `Retriever` (ABC), `DenseRetriever`, `BM25Retriever`
- `DenseRetriever` loads the FAISS index + chunk metadata, encodes the query through the same `Embedder`, returns top-k by inner product.
- `BM25Retriever` builds an in-memory BM25 index from `all_chunks.jsonl` using a custom tokenizer (lowercase + punctuation strip + English stopwords).
- Both return `list[RetrievalResult]` so they're drop-in interchangeable.

#### `reranker.py` — `Reranker`, `RerankingRetriever`
- `Reranker` wraps a HuggingFace cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) and scores `(query, text)` pairs jointly.
- `RerankingRetriever` decorates any `Retriever` with two-stage retrieval: over-fetch `fetch_k` candidates from the base, rescore with the cross-encoder, return top-`k`. Method tag is preserved as `"dense+rerank"` / `"bm25+rerank"`.

#### `rag.py` — `RAGPipeline`, `Answer`, REPL
- `RAGPipeline` ties a `Retriever` and an optional `BaseLLMClient` together. Its `answer(question, k)` retrieves, builds a chat-completion messages list via `form_the_question()`, and (if a generator is wired in) calls `.chat(messages)` for the final answer.
- The `_main()` function is the `askistanbul-repl` entry point. It loads both retrievers at startup, lazily loads the reranker if enabled, dispatches REPL commands (`:k`, `:method`, `:rerank`, `:fetch-k`, `:show`).

#### `pipeline.py` — `Pipeline`
Composes all four offline stages (`Scraper` → `Preprocessor` → `Chunker` → `Embedder`) into a single `.run()` call. The `askistanbul-index` CLI is just a thin wrapper that exposes `--skip-scrape` and `--skip-embed` flags.

#### `generator/`
Hexagonal-architecture LLM client layer.

- **`port/BaseLLMClient.py`** — abstract base. Subclasses must implement `chat(messages, temperature, max_new_tokens) -> str`; `chat_json` has a default implementation that calls `chat` and parses the response.
- **`adapter/OllamaClient.py`** — concrete adapter for Ollama (`POST /api/chat`). Reads `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from `config`. Includes a `ping()` method for connectivity checks; doesn't auto-ping at construction.
- **`factory/LLMClientFactory.py`** — picks the right client based on a `client_type` string. Easy to extend with OpenAI / Anthropic later.

---

## Programmatic API

Everything CLI-able is also usable from Python:

```python
from askistanbul import (
    Pipeline, RAGPipeline,
    DenseRetriever, BM25Retriever, RerankingRetriever, Reranker,
    OllamaClient,
)

# offline: build the index
Pipeline().run()

# online: retrieval only
rag = RAGPipeline(retriever=DenseRetriever())
result = rag.answer("best rooftop bars in Beyoglu", k=5)
for r in result.results:
    print(r.score, r.chunk.title, "—", r.chunk.heading)

# online: retrieval + reranking + LLM generation
rag = RAGPipeline(
    retriever=RerankingRetriever(DenseRetriever(), Reranker(), fetch_k=20),
    generator=OllamaClient(),
)
answer = rag.answer("how do I get from the airport to Taksim?", k=5)
print(answer.answer)         # LLM-generated, grounded text
print(answer.citations)      # ["Istanbul Airport — Get out (https://…)", …]
```

This is what the eval harness (Week 3) and any future Gradio UI (Week 5) hook into.

---

## Development notes

- **No reinstall after code edits.** `pip install -e .` was a one-time setup. Editing files under `src/askistanbul/` takes effect immediately. You only need to reinstall when `pyproject.toml` changes (new dependency, new CLI script).
- **Python version drift on macOS.** If `import askistanbul` stops working after a Python point-upgrade (e.g. 3.14.0 → 3.14.1), recreate the venv: `deactivate; rm -rf .venv; python3.14 -m venv .venv; source .venv/bin/activate; pip install -U pip setuptools; pip install -e . -r requirements.txt`.
- **Reproducing the index.** `data/raw/` is gitignored. To reproduce a teammate's setup, just `askistanbul-index` from scratch — the scraper is idempotent and the cached Wikivoyage articles come down deterministically (modulo upstream edits).
- **Ablation experiments.** Each chunker/embedder config can live in its own index directory: `askistanbul-embed --index-dir data/index_e5 --model intfloat/multilingual-e5-base`, then in code: `DenseRetriever(index_dir="data/index_e5")`. No need to clobber the primary index.

---

## License

Wikivoyage content is CC BY-SA 3.0. This repository's code is for coursework purposes.
