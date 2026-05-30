"""askistanbul — RAG-based Q&A travel guide for Istanbul."""

# Loaded first so `.env` is in place before any other module-level env reads.
from .config import Config, config

from .chunker import Chunker, chunk_text, tokenize
from .embedder import Embedder
from .models import (
    Chunk,
    CleanedDocument,
    RawPage,
    RetrievalResult,
    Section,
)
from .pipeline import Pipeline
from .rag import Answer, RAGPipeline
from .preprocess import (
    LISTING_TYPES,
    Preprocessor,
    SectionSplitter,
    WikitextCleaner,
    remove_html,
    remove_tables,
    remove_wikilinks,
)
from .reranker import Reranker, RerankingRetriever
from .retriever import (
    BM25Retriever,
    DenseRetriever,
    Retriever,
    bm25_tokenize,
)
from .scraper import DEFAULT_PAGES, Scraper, slug

from .generator.adapter.OllamaClient import OllamaClient
from .generator.adapter.OpenRouterClient import OpenRouterClient
from .generator.port.BaseLLMClient import BaseLLMClient
from .generator.factory.LLMClientFactory import LLMClientFactory

__all__ = [
    # config
    "Config", "config",
    # data
    "RawPage", "Section", "CleanedDocument", "Chunk", "RetrievalResult",
    # scraper
    "Scraper", "DEFAULT_PAGES", "slug",
    # preprocess
    "WikitextCleaner", "SectionSplitter", "Preprocessor", "LISTING_TYPES",
    "remove_wikilinks", "remove_html", "remove_tables",
    # chunker
    "Chunker", "tokenize", "chunk_text",
    # embedder
    "Embedder",
    # retriever
    "Retriever", "DenseRetriever", "BM25Retriever", "bm25_tokenize",
    # reranker
    "Reranker", "RerankingRetriever",
    # facade
    "Pipeline", "RAGPipeline", "Answer",
    # LLM clients
    "BaseLLMClient", "OllamaClient", "OpenRouterClient", "LLMClientFactory"
]
