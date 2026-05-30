"""Online RAG pipeline + interactive REPL.

This module owns the query-time path:
  * :class:`Answer`    — return container for a single Q&A turn.
  * :class:`RAGPipeline` — retrieve (and optionally generate) for a question.
  * :func:`_main`      — the ``askistanbul-repl`` entry point.

``Pipeline`` (offline indexing) lives in ``pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import config
from .generator.factory.LLMClientFactory import LLMClientFactory
from .generator.port.BaseLLMClient import BaseLLMClient
from .models import RetrievalResult
from .retriever import BM25Retriever, DenseRetriever, Retriever


# ---------------------------------------------------------------------------
# Generator protocol + Answer container
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    question: str
    results: list[RetrievalResult]
    answer: Optional[str] = None
    citations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Retrieve chunks (and optionally generate an answer) for a question."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[BaseLLMClient] = None,
    ):
        self.retriever = retriever or DenseRetriever()
        self.generator = generator

    def answer(self, question: str, k: int = 5) -> Answer:
        results = self.retriever.retrieve(question, k=k)
        answer_text: Optional[str] = None
        if self.generator is not None:
            messages = self.form_the_question(question, results)
            answer_text = self.generator.chat(messages)
        citations = [
            f"{r.chunk.title} — {r.chunk.heading} ({r.chunk.url})"
            for r in results
        ]
        return Answer(
            question=question,
            results=results,
            answer=answer_text,
            citations=citations,
        )

    def form_the_question(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[dict[str, str]]:
        """Build a chat-completion messages list from question + retrieved chunks."""
        context = "\n\n".join(
            f"[{i}] {r.chunk.title} — {r.chunk.heading}\n{r.chunk.text}"
            for i, r in enumerate(results, 1)
        )
        system = f"""
        You are an assistant for answering questions about Istanbul. 
        Use only the provided retrieved information to answer the question. 
        If you don't know the answer, say you don't know — do not make up an answer. 
        Always cite your sources using the format [1], [2], etc. corresponding to the retrieved chunks.

        Context and question are below. Provide a concise answer, then list your citations.
        """
        user = f"Context: {context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

_HELP = """\
Commands:
  :help                    Show this message
  :k N                     Set top-k (current: {k})
  :method dense|bm25       Switch retrieval backend (current: {method})
  :rerank on|off           Toggle cross-encoder reranker (current: {rerank})
  :fetch-k N               Reranker over-fetch count (current: {fetch_k})
  :show on|off             Toggle full chunk text in output (current: {show})
  :q / :quit / exit        Exit
Anything else is treated as a question.\
"""


def _render(ans: Answer, show_text: bool) -> None:
    if ans.answer:
        print(f"\n{ans.answer}\n")
    for i, r in enumerate(ans.results, 1):
        score_str = f"{r.score:.4f}"
        if r.cescore is not None:
            score_str += f"  ce={r.cescore:.4f}"
        print(f"[{i}] {score_str}  |  {r.chunk.title} / {r.chunk.heading}  ({r.method})")
        if show_text:
            print(f"    {r.chunk.text[:400].replace(chr(10), ' ')}")
    print()


def _main() -> None:
    import argparse

    from .reranker import Reranker, RerankingRetriever

    p = argparse.ArgumentParser(description="Interactive RAG REPL over the Istanbul index.")
    p.add_argument("--k", type=int, default=5, help="Default top-k.")
    p.add_argument("--method", choices=["dense", "bm25"], default="dense",
                   help="Initial retrieval backend.")
    p.add_argument("--rerank", action="store_true",
                   help="Enable cross-encoder reranking from the start.")
    p.add_argument("--fetch-k", type=int, default=config.reranker_fetch_k,
                   help="Reranker over-fetch count (default from RERANKER_FETCH_K).")
    p.add_argument("--client-type", choices=["ollama", "openai", "anthropic"], default="ollama",
                   help="LLM client type.")
    p.add_argument("--show-text", action="store_true",
                   help="Print chunk text by default (toggle with :show).")
    args = p.parse_args()

    # readline gives arrow-key history on Mac/Linux when available.
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    k = args.k
    method = args.method
    llm_client = args.client_type
    show = args.show_text
    rerank_on = args.rerank
    fetch_k = args.fetch_k

    print("Loading retrievers... (this may take a few seconds the first time)")
    base_retrievers: dict[str, Retriever] = {
        "dense": DenseRetriever(),
        "bm25":  BM25Retriever(),
    }

    # Reranker is lazy-loaded — pays its ~90MB download cost only when needed.
    reranker: Optional[Reranker] = None
    if rerank_on:
        print("Loading reranker... (first time downloads model from HF)")
        reranker = Reranker()

    def active_retriever() -> Retriever:
        nonlocal reranker
        base = base_retrievers[method]
        if not rerank_on:
            return base
        if reranker is None:
            print("Loading reranker... (first time downloads model from HF)")
            reranker = Reranker()
        return RerankingRetriever(base=base, reranker=reranker, fetch_k=fetch_k)

    print("Loading generator... (this may take a few seconds the first time)")
    generator: BaseLLMClient = LLMClientFactory.create_llm_client(config, llm_client)

    print(
        f"Initializing RAG pipeline — method={method}, rerank={'on' if rerank_on else 'off'}, "
        f"generator={llm_client}..."
    )
    rag = RAGPipeline(retriever=active_retriever(), generator=generator)

    print(
        f"\naskistanbul REPL — method={method}, k={k}, "
        f"rerank={'on' if rerank_on else 'off'}. "
        f"Type :help for commands, :q to exit.\n"
    )

    while True:
        try:
            line = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # ----- commands ---------------------------------------------------
        if line in (":q", ":quit", "exit", "quit"):
            break
        if line == ":help":
            print(_HELP.format(
                k=k, method=method, show="on" if show else "off",
                rerank="on" if rerank_on else "off", fetch_k=fetch_k,
            ))
            continue
        if line.startswith(":k"):
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[1].isdigit():
                print("usage: :k <positive int>")
            else:
                k = int(parts[1])
                print(f"k = {k}")
            continue
        if line.startswith(":method"):
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[1] not in base_retrievers:
                print("usage: :method dense|bm25")
            else:
                method = parts[1]
                rag.retriever = active_retriever()
                print(f"method = {method}")
            continue
        if line.startswith(":rerank"):
            parts = line.split(None, 1)
            val = parts[1].strip().lower() if len(parts) == 2 else ""
            if val in ("on", "true", "1"):
                rerank_on = True
            elif val in ("off", "false", "0"):
                rerank_on = False
            else:
                print("usage: :rerank on|off")
                continue
            rag.retriever = active_retriever()
            print(f"rerank = {'on' if rerank_on else 'off'}")
            continue
        if line.startswith(":fetch-k"):
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[1].isdigit():
                print("usage: :fetch-k <positive int>")
            else:
                fetch_k = int(parts[1])
                rag.retriever = active_retriever()
                print(f"fetch-k = {fetch_k}")
            continue
        if line.startswith(":show"):
            parts = line.split(None, 1)
            val = parts[1].strip().lower() if len(parts) == 2 else ""
            if val in ("on", "true", "1"):
                show = True
            elif val in ("off", "false", "0"):
                show = False
            else:
                print("usage: :show on|off")
                continue
            print(f"show-text = {'on' if show else 'off'}")
            continue
        if line.startswith(":"):
            print(f"unknown command: {line!r}. Type :help.")
            continue

        # ----- query ------------------------------------------------------
        try:
            ans = rag.answer(line, k=k)
            _render(ans, show)
        except Exception as exc:
            print(f"[error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    _main()
