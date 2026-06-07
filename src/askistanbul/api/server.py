"""
api/server.py
-------------
FastAPI backend + lightweight web UI for AskIstanbul.

No authentication (per current scope). By default all models (dense + bm25
retrievers, reranker, LLM client) are warmed at startup; set ASKISTANBUL_WARM=0
(or --no-warm) to load them lazily on first request instead.

Endpoints
  GET  /                 → the single-page web UI (static/index.html)
  GET  /api/health       → liveness + which backends are ready
  GET  /api/config       → defaults and available options (for the UI)
  POST /api/retrieve     → retrieval only (no LLM)
  POST /api/ask          → retrieval + grounded generation

Run:
  askistanbul-serve                 # http://127.0.0.1:8000
  askistanbul-serve --port 9000 --reload
  uvicorn askistanbul.api.server:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import config
from ..generator.factory.LLMClientFactory import LLMClientFactory
from ..rag import Answer, RAGPipeline
from ..retriever import BM25Retriever, DenseRetriever, Retriever

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Lazily-initialised singletons (shared across requests)
# ---------------------------------------------------------------------------

_state: dict = {
    "dense": None,
    "bm25": None,
    "reranker": None,
    "generator": None,
    "generator_kind": None,   # "ollama" | "openrouter" | None
}


def get_retriever(method: str) -> Retriever:
    """Return the retriever for a method. The k values are passed at query time
    (retrieve/answer), not at construction."""
    method = method.lower()
    if method in ("dense", "rerank") and _state["dense"] is None:
        _state["dense"] = DenseRetriever()
    if method == "bm25" and _state["bm25"] is None:
        _state["bm25"] = BM25Retriever()

    if method == "dense":
        return _state["dense"]
    if method == "bm25":
        return _state["bm25"]
    if method == "rerank":
        from ..reranker import Reranker, RerankingRetriever
        if _state["reranker"] is None:
            _state["reranker"] = Reranker()
        return RerankingRetriever(base=_state["dense"], reranker=_state["reranker"])
    raise HTTPException(status_code=400, detail=f"unknown method {method!r}")


def _try_make_generator(kind: str) -> Optional[object]:
    """Build one LLM client via the factory, returning None if unusable."""
    try:
        client = LLMClientFactory.create_llm_client(config, kind)
    except Exception as exc:
        print(f"[generator] {kind!r} unavailable: {exc}")
        return None
    # Ollama is constructed without a connection check — verify it now so we can
    # fall back instead of failing at request time.
    if kind == "ollama" and hasattr(client, "ping") and not client.ping():
        return None
    return client


def get_generator() -> Optional[object]:
    """Return a cached LLM client, preferring ``config.generator_type``.

    The configured backend (``GENERATOR_TYPE``) is tried first; if it isn't
    reachable, the remaining known backends are tried as fallbacks, so the UI
    keeps working. Returns None if none are usable (retrieval-only).
    """
    if _state["generator"] is not None:
        return _state["generator"]

    preferred = (config.generator_type or "ollama").lower().strip()
    order = [preferred] + [k for k in ("ollama", "openrouter") if k != preferred]

    for kind in order:
        client = _try_make_generator(kind)
        if client is not None:
            _state["generator"] = client
            _state["generator_kind"] = kind
            if kind != preferred:
                print(f"[generator] fell back to {kind!r} (preferred {preferred!r} unavailable)")
            return client

    return None


def warmup() -> None:
    """Eagerly load every model/retriever so the first request is fast.

    Runs at server startup (see ``lifespan``). Set ``ASKISTANBUL_WARM=0`` to
    fall back to lazy, on-first-request loading instead.
    """
    print("[startup] warming up — this downloads/loads models once ...")
    print("[startup]   dense retriever (embedding model + FAISS) ...")
    _state["dense"] = DenseRetriever(fetch_k=config.biencoder_fetch_k)
    print("[startup]   bm25 retriever ...")
    _state["bm25"] = BM25Retriever()
    print("[startup]   cross-encoder reranker ...")
    try:
        from ..reranker import Reranker
        _state["reranker"] = Reranker(config.reranker_model)
    except Exception as exc:  # don't let a reranker failure block the server
        print(f"[startup]   reranker warmup skipped: {exc}")
    print("[startup]   generator (LLM client) ...")
    get_generator()
    print(f"[startup] ready — generator={_state['generator_kind'] or 'none (retrieval-only)'}")


# CLI may override the configured default (see _main); None = use config.
_warm_override: Optional[bool] = None


def _should_warm() -> bool:
    """Resolve the warmup setting: CLI override wins, else config (from .env)."""
    if _warm_override is not None:
        return _warm_override
    return config.askistanbul_warm


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    if _should_warm():
        warmup()
    else:
        print("[startup] warmup disabled (ASKISTANBUL_WARM) — "
              "models load lazily on first request")
    yield


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language travel question")
    method: str = Field("dense", description="dense | bm25 | rerank")
    generate: bool = Field(True, description="If false, retrieval only (no LLM)")
    fetch_k_biencoder: int = Field(20, ge=1, le=100,
                                   description="Candidates the bi-encoder returns (dense/bm25 result count, or rerank pool)")
    fetch_k_reranker: int = Field(5, ge=1, le=100,
                                  description="Final results after reranking (only used when method=rerank)")


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    method: str = Field("dense", description="dense | bm25 | rerank")
    fetch_k_biencoder: int = Field(20, ge=1, le=100,
                                   description="Candidates the bi-encoder returns (dense/bm25 result count, or rerank pool)")
    fetch_k_reranker: int = Field(5, ge=1, le=100,
                                  description="Final results after reranking (only used when method=rerank)")


def _serialize_results(results) -> list[dict]:
    out = []
    for rank, r in enumerate(results, start=1):
        out.append({
            "rank": rank,
            "chunk_id": r.chunk.chunk_id,
            "title": r.chunk.title,
            "heading": r.chunk.heading,
            "url": r.chunk.url,
            "text": r.chunk.text,
            "score": round(r.score, 4),
            "method": r.method,
            "cescore": round(r.cescore, 4) if r.cescore is not None else None,
        })
    return out


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AskIstanbul API",
    description="RAG-based Q&A travel guide for Istanbul (Wikivoyage-grounded).",
    version="0.1.1",
    lifespan=lifespan,
)

# No auth for now — allow any origin so the UI / external callers work freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "dense_loaded": _state["dense"] is not None,
        "bm25_loaded": _state["bm25"] is not None,
        "generator_kind": _state["generator_kind"],
    }


@app.get("/api/config")
def get_config() -> dict:
    return {
        "methods": ["dense", "bm25", "rerank"],
        "default_method": "dense",
        "default_fetch_k_biencoder": config.biencoder_fetch_k,
        "default_fetch_k_reranker": config.reranker_fetch_k,
        "embedding_model": config.embedding_model or "all-MiniLM-L6-v2",
        "generator_type": config.generator_type,
        "ollama_model": config.ollama_model,
        "openrouter_model": config.openrouter_model,
    }


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    retriever = get_retriever(req.method)
    results = retriever.retrieve(
        req.question,
        biencoder_k=req.fetch_k_biencoder,
        reranker_k=req.fetch_k_reranker,
    )
    return {
        "question": req.question,
        "method": req.method,
        "fetch_k_biencoder": req.fetch_k_biencoder,
        "fetch_k_reranker": req.fetch_k_reranker,
        "results": _serialize_results(results),
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    retriever = get_retriever(req.method)
    generator = get_generator() if req.generate else None
    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    ans = pipeline.answer(
        req.question,
        fetch_k_biencoder=req.fetch_k_biencoder,
        fetch_k_reranker=req.fetch_k_reranker,
    )
    return {
        "question": req.question,
        "method": req.method,
        "fetch_k_biencoder": req.fetch_k_biencoder,
        "fetch_k_reranker": req.fetch_k_reranker,
        "generated": generator is not None,
        "generator_kind": _state["generator_kind"] if generator is not None else None,
        "answer": ans.answer,
        "citations": ans.citations,
        "results": _serialize_results(ans.results),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Serve any other static assets (css/js) if added later.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# CLI launcher — `askistanbul-serve`
# ---------------------------------------------------------------------------

def _port_in_use(host: str, port: int) -> bool:
    """True if something is already listening on (host, port)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _main() -> None:
    import argparse
    import sys

    import uvicorn

    p = argparse.ArgumentParser(description="Serve the AskIstanbul API + web UI.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    warm_group = p.add_mutually_exclusive_group()
    warm_group.add_argument("--warm", dest="warm", action="store_true", default=None,
                            help="Force warmup at startup (overrides ASKISTANBUL_WARM)")
    warm_group.add_argument("--no-warm", dest="warm", action="store_false", default=None,
                            help="Force lazy loading on first request (overrides ASKISTANBUL_WARM)")
    args = p.parse_args()

    # Fail fast BEFORE warmup if the port is taken — otherwise uvicorn would load
    # every model and only then hit "address already in use".
    if _port_in_use(args.host, args.port):
        print(f"[error] {args.host}:{args.port} is already in use — stop the other "
              f"server (e.g. lsof -ti tcp:{args.port} | xargs kill) or pass --port <N>.",
              file=sys.stderr)
        raise SystemExit(1)

    # Default warmup comes from config (ASKISTANBUL_WARM in .env); --warm/--no-warm
    # override it. Set both the module flag (for this process) and the env var (so
    # the --reload subprocess, which re-imports config, sees the override too).
    if args.warm is not None:
        global _warm_override
        _warm_override = args.warm
        os.environ["ASKISTANBUL_WARM"] = "true" if args.warm else "false"

    print(f"AskIstanbul serving on http://{args.host}:{args.port}  (UI at /)")
    uvicorn.run(
        "askistanbul.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    _main()
