"""Vector-search backends behind one interface, selected by config."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import AWS_REGION, KB_ID, NUM_RESULTS, RETRIEVAL_BACKEND


@dataclass
class Chunk:
    text: str
    source: str
    score: float | None = None


class Retrieval(ABC):
    @abstractmethod
    def search(self, question: str, k: int = NUM_RESULTS) -> list[Chunk]: ...


class ChromaRetrieval(Retrieval):
    """Similarity search over a local Chroma store."""

    def __init__(self, path: str = "chroma_db"):
        import chromadb

        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            "payer_docs"
        )

    def search(self, question: str, k: int = NUM_RESULTS) -> list[Chunk]:
        r = self._collection.query(query_texts=[question], n_results=k)
        return [
            Chunk(text=doc, source=meta.get("source", "?"), score=1 - dist)  # distance -> similarity
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]


class KBRetrieval(Retrieval):
    """Similarity search via a managed Bedrock Knowledge Base."""

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
    """Instantiate the configured (or named) retrieval backend."""
    name = name or RETRIEVAL_BACKEND
    if name == "chroma":
        return ChromaRetrieval()
    if name == "kb":
        return KBRetrieval()
    raise ValueError(f"Unknown retrieval backend {name!r}. Options: chroma, kb")
