"""Centralized configuration.

Imports this module:
  1. Loads `.env` from the project root (via python-dotenv).
  2. Reads every env var the rest of the package needs.
  3. Exposes a frozen ``config`` singleton; every other module reads from it
     instead of calling ``os.getenv`` directly.

This guarantees ``.env`` is loaded exactly once, before any other module-level
env-var access happens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# python-dotenv is in requirements.txt but we degrade gracefully if missing,
# so the package still imports on a fresh venv before `pip install`.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _str_env(name: str, default: str | None = None) -> str | None:
    """Like os.getenv, but treat an empty string the same as missing.

    Common case: a user writes ``FOO=`` in ``.env`` and python-dotenv returns ``""``,
    which os.getenv would NOT replace with ``default``. This helper does.
    """
    v = os.getenv(name)
    return v if v else default


def _int_env(name: str, default: int) -> int:
    """Like _str_env but returns int. Empty/missing → default (no ValueError)."""
    v = os.getenv(name)
    return int(v) if v else default


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var. Empty/missing → default.

    Truthy: 1, true, yes, on (case-insensitive). Everything else is False.
    """
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # --- Scraper -----------------------------------------------------------
    askistanbul_contact: str
    askistanbul_data_dir: str | None

    # --- Embedding model ---------------------------------------------------
    embedding_model: str | None
    biencoder_fetch_k: int

    # --- Reranker (cross-encoder; opt-in) ---------------------------------
    reranker_model: str
    reranker_fetch_k: int

    # --- API server --------------------------------------------------------
    # Eagerly load retrievers/reranker/LLM at startup (vs. lazily on first
    # request). True = no first-query stall; False = faster boot, lazy loading.
    askistanbul_warm: bool

    # --- Judge type for proposal evaluation (openrouter, openai, anthropic) -------------
    judge_type: str | None

    # --- Generator type — which LLM backend answers queries (ollama, openrouter) -------
    generator_type: str

    # --- Local LLM (Ollama) ------------------------------------------------
    ollama_base_url: str
    ollama_model: str

    # --- Hosted API fallbacks ---------------------------------------------
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_referer: str | None
    openrouter_title: str | None
    openai_api_key: str | None
    openai_model: str
    anthropic_api_key: str | None
    anthropic_model: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            askistanbul_contact=_str_env("ASKISTANBUL_CONTACT", "askistanbul@example.com"),
            askistanbul_data_dir=_str_env("ASKISTANBUL_DATA_DIR"),
            embedding_model=_str_env("EMBEDDING_MODEL"),
            biencoder_fetch_k=_int_env("BIENCODER_FETCH_K", 20),
            reranker_model=_str_env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            reranker_fetch_k=_int_env("RERANKER_FETCH_K", 5),
            askistanbul_warm=_bool_env("ASKISTANBUL_WARM", True),
            judge_type=_str_env("JUDGE_TYPE", "openrouter"),
            generator_type=_str_env("GENERATOR_TYPE", "ollama"),
            ollama_base_url=_str_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_str_env("OLLAMA_MODEL", "qwen2.5:7b"),
            openrouter_api_key=_str_env("OPENROUTER_API_KEY"),
            openrouter_model=_str_env("OPENROUTER_LLM_MODEL", "qwen/qwen-2.5-72b-instruct"),
            openrouter_referer=_str_env("OPENROUTER_REFERER"),
            openrouter_title=_str_env("OPENROUTER_TITLE", "askistanbul"),
            openai_api_key=_str_env("OPENAI_API_KEY"),
            openai_model=_str_env("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_api_key=_str_env("ANTHROPIC_API_KEY"),
            anthropic_model=_str_env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        )


# Module-level singleton — populated once at import.
config: Config = Config.from_env()
