"""Vector-search backends behind a single interface.

Each backend takes a question and returns the most relevant chunks. The app
depends on the Retrieval interface, so local (Chroma) and cloud (Bedrock
Knowledge Base) are interchangeable by configuration.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import AWS_REGION, KB_ID, NUM_RESULTS, RETRIEVAL_BACKEND


@dataclass
class Chunk:
    """A retrieved passage, its origin document, and its similarity score."""

    text: str
    source: str
    score: float | None = None


class Retrieval(ABC):
    """Contract every retrieval backend implements."""

    @abstractmethod
    def search(self, question: str, k: int = NUM_RESULTS) -> list[Chunk]: ...


class ChromaRetrieval(Retrieval):
    """Similarity search over a local Chroma vector store."""

    def __init__(self, path: str = "chroma_db"):
        import chromadb

        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            "payer_docs"
        )

    def search(self, question: str, k: int = NUM_RESULTS) -> list[Chunk]:
        r = self._collection.query(query_texts=[question], n_results=k)
        # distance -> similarity so higher always means "closer in meaning"
        return [
            Chunk(text=doc, source=meta.get("source", "?"), score=1 - dist)
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]


class KBRetrieval(Retrieval):
    """Similarity search delegated to a managed Bedrock Knowledge Base."""

    def __init__(self):
        from langchain_aws import AmazonKnowledgeBasesRetriever

        self._retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=KB_ID,
            region_name=AWS_REGION,
            retrieval_config={
                "vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}
            },
        )

    def search(self, question: str, k: int = NUM_RESULTS) -> list[Chunk]:
        docs = self._retriever.invoke(question)
        return [
            Chunk(text=d.page_content, source=str(d.metadata.get("source_metadata", "KB")))
            for d in docs
        ]


def get_retrieval(name: str | None = None) -> Retrieval:
    """Return the retrieval backend named by the argument or configuration."""
    name = name or RETRIEVAL_BACKEND
    if name == "chroma":
        return ChromaRetrieval()
    if name == "kb":
        return KBRetrieval()
    raise ValueError(f"Unknown retrieval backend {name!r}. Options: chroma, kb")
