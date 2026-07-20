"""Ingestion pipeline: docs/*.txt -> chunks -> local Chroma vector store.

Run once (and after any docs change):  python ingest.py
"""
import pathlib

import chromadb


def chunk(text: str, max_chars: int = 1500) -> list[str]:
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > max_chars and cur:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    return out + ([cur] if cur else [])


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("payer_docs")  # rebuild from scratch
    except Exception:
        pass
    collection = client.get_or_create_collection(
        "payer_docs", metadata={"hnsw:space": "cosine"}
    )

    n = 0
    for f in pathlib.Path("docs").glob("*.txt"):
        for i, c in enumerate(chunk(f.read_text(encoding="utf-8"))):
            collection.add(
                ids=[f"{f.stem}-{i}"], documents=[c], metadatas=[{"source": f.name}]
            )
            n += 1
    print(f"indexed {n} chunks from docs/ into chroma_db/")
