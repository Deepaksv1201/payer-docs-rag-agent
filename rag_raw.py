"""RAG from scratch — no frameworks, no vector database, no cloud.

The full pipeline in one readable file:
    chunk -> embed -> cosine-similarity retrieval -> grounded generation

Usage: python rag_raw.py "your question"   (reads docs/*.txt, needs Ollama running)
"""
import os
import pathlib
import sys

import numpy as np
import ollama
from fastembed import TextEmbedding

EMBED_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
GEN_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

PROMPT = """Answer the question using ONLY the context below. If the answer
is not in the context, say "I don't have that information in the provided
documents."

Context:
{context}

Question: {question}"""


def chunk(text: str, max_chars: int = 1500) -> list[str]:
    """Paragraph-packing chunker with a max size per chunk."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > max_chars and cur:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    return out + ([cur] if cur else [])


def embed(text: str) -> np.ndarray:
    return np.array(next(iter(EMBED_MODEL.embed([text]))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))


if __name__ == "__main__":
    question = sys.argv[1]

    # index (in a real system this happens once, offline — see ingest.py)
    chunks: list[str] = []
    for f in pathlib.Path("docs").glob("*.txt"):
        chunks += chunk(f.read_text(encoding="utf-8"))
    if not chunks:
        sys.exit("No .txt files found in ./docs")
    vectors = [embed(c) for c in chunks]

    # retrieve
    qv = embed(question)
    scores = [cosine(qv, v) for v in vectors]
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:3]
    context = "\n\n".join(chunks[i] for i in top)

    # generate
    resp = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
    )
    print(resp["message"]["content"])

    print("\n--- sources (cosine similarity) ---")
    for i in top:
        print(f"[{scores[i]:.3f}]", chunks[i][:120].replace("\n", " "))
