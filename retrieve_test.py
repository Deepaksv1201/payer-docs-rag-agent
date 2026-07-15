"""Inspect retrieval in isolation — which chunks does vector search return?

Usage: python retrieve_test.py "your question"
"""
import sys

from ask import retriever

if __name__ == "__main__":
    question = sys.argv[1]
    docs = retriever.invoke(question)
    print(f"{len(docs)} chunks retrieved for: {question!r}\n")
    for i, d in enumerate(docs, 1):
        print(f"--- chunk {i} ---")
        print(d.page_content[:400])
        print("metadata:", d.metadata, "\n")
