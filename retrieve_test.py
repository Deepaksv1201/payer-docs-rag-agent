"""Inspect retrieval in isolation — which chunks does vector search return?

Works against whichever backend RETRIEVAL_BACKEND selects (chroma | kb).

Usage: python retrieve_test.py "your question"
"""
import sys

from retrieval import get_retrieval

if __name__ == "__main__":
    question = sys.argv[1]
    chunks = get_retrieval().search(question)
    print(f"{len(chunks)} chunks retrieved for: {question!r}\n")
    for i, c in enumerate(chunks, 1):
        score = f" · score {c.score:.3f}" if c.score is not None else ""
        print(f"--- chunk {i} · {c.source}{score} ---")
        print(c.text[:400], "\n")
