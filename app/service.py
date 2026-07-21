"""The RAG business logic — the heart of the application.

Turns a question into a grounded answer: retrieve the most relevant chunks,
wrap them in a prompt that forbids answering outside that context, and generate
the reply. Framework-independent, so the API, the UI, and the evaluation
harness all reuse the same logic instead of duplicating it.
"""
import logging
import time
from dataclasses import dataclass

from app.providers import LLMProvider, get_provider
from app.retrieval import Chunk, Retrieval, get_retrieval

log = logging.getLogger("docq")

# The grounding contract: answer only from retrieved context, refuse otherwise.
# This is what keeps answers traceable to documents instead of invented.
PROMPT = """You are a precise assistant answering questions from a document set.
Using ONLY the context below, write a clear, complete answer of 2-4 sentences.
Do not copy the context verbatim — explain it. If the context does not contain
the answer, reply exactly: "I don't have that information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class Answer:
    """A generated answer plus the evidence and metadata behind it."""

    text: str
    sources: list[Chunk]
    latency_seconds: float
    provider: str


class RagService:
    """Orchestrates retrieval and generation. Backends are injected so the
    service can be tested or reconfigured without touching this logic."""

    def __init__(
        self,
        retrieval: Retrieval | None = None,
        provider: LLMProvider | None = None,
    ):
        self.retrieval = retrieval or get_retrieval()
        self.provider = provider or get_provider()

    def answer(self, question: str, k: int = 4) -> Answer:
        start = time.perf_counter()
        chunks = self.retrieval.search(question, k=k)
        context = "\n\n".join(c.text for c in chunks)
        text = self.provider.generate(
            PROMPT.format(context=context, question=question)
        )
        elapsed = time.perf_counter() - start
        log.info("answered | chunks=%d | latency=%.2fs", len(chunks), elapsed)
        return Answer(
            text=text,
            sources=chunks,
            latency_seconds=round(elapsed, 2),
            provider=self.provider.name,
        )
