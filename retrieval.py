"""Retrieval layer — vector search backends behind one interface.

ChromaRetrieval: local persistent vector store (dev / offline).
KBRetrieval:    Bedrock Knowledge Base (AWS, managed).
Selected via RETRIEVAL_BACKEND env var.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    score: float | None = None


class Retrieval(ABC):
    @abstractmethod
    def search(self, question: str, k: int = 4) -> list[Chunk]: ...


class ChromaRetrieval(Retrieval):
    def __init__(self, path: str = "chroma_db"):
        import chromadb

        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            "payer_docs"
        )

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        r = self._collection.query(query_texts=[question], n_results=k)
        return [
            Chunk(text=doc, source=meta.get("source", "?"), score=1 - dist)
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]


class KBRetrieval(Retrieval):
    def __init__(self):
        from langchain_aws import AmazonKnowledgeBasesRetriever

        from config import AWS_REGION, KB_ID, NUM_RESULTS

        self._retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=KB_ID,
            region_name=AWS_REGION,
            retrieval_config={
                "vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}
            },
        )

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        docs = self._retriever.invoke(question)
        return [
            Chunk(text=d.page_content, source=str(d.metadata.get("source_metadata", "KB")))
            for d in docs
        ]


def get_retrieval(name: str | None = None) -> Retrieval:
    name = name or os.environ.get("RETRIEVAL_BACKEND", "chroma")
    if name == "chroma":
        return ChromaRetrieval()
    if name == "kb":
        return KBRetrieval()
    raise ValueError(f"Unknown retrieval backend {name!r}. Options: chroma, kb")
