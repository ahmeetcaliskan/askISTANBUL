"""Dataclasses for the askistanbul pipeline.

`to_dict` / `from_dict` mirror the existing on-disk JSON schema so the
refactor doesn't force a re-scrape or re-index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RawPage:
    """A page fetched from Wikivoyage with raw wikitext intact."""
    title: str
    pageid: int
    url: str
    wikitext: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawPage":
        return cls(
            title=d["title"],
            pageid=d.get("pageid", 0),
            url=d.get("url", ""),
            wikitext=d.get("wikitext", ""),
        )


@dataclass
class Section:
    """A heading + body slice of a cleaned document."""
    heading: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Section":
        return cls(heading=d["heading"], text=d["text"])


@dataclass
class CleanedDocument:
    """A scrubbed Wikivoyage article split into sections."""
    title: str
    url: str
    sections: list[Section] = field(default_factory=list)
    full_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "sections": [s.to_dict() for s in self.sections],
            "full_text": self.full_text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CleanedDocument":
        return cls(
            title=d["title"],
            url=d.get("url", ""),
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            full_text=d.get("full_text", ""),
        )


@dataclass
class Chunk:
    """A token-window slice of a section, with provenance metadata."""
    chunk_id: str
    title: str
    url: str
    heading: str
    text: str
    token_count: int
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            title=d["title"],
            url=d.get("url", ""),
            heading=d.get("heading", ""),
            text=d["text"],
            token_count=d.get("token_count", 0),
            chunk_index=d.get("chunk_index", 0),
        )


@dataclass
class RetrievalResult:
    """A chunk returned by a retriever, with score + method tag.

    ``score``    — always the base retriever's score (dense inner-product or BM25).
    ``method``   — always the base retriever's identifier ("dense" or "bm25").
    ``cescore``  — cross-encoder reranker score, or ``None`` if not reranked.

    ``cescore is not None`` is the canonical "was this result reranked?" check.
    """
    chunk: Chunk
    score: float
    method: str                       # "dense" | "bm25"
    cescore: float | None = None      # cross-encoder rerank score, or None

    @property
    def reranked(self) -> bool:
        return self.cescore is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            **self.chunk.to_dict(),
            "score": self.score,
            "retrieval_method": self.method,
        }
        if self.cescore is not None:
            d["cescore"] = self.cescore
        return d
