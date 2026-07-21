"""Offline ingestion: turn a folder of documents into a searchable vector store.

Reads PDFs and text files, splits them into overlapping chunks, and writes them
into Chroma (which embeds each chunk on insert). Runs as a batch step whenever
the corpus changes — separate from the query path, which only reads the store.
"""
import os
import pathlib
import sys


def read_document(path: pathlib.Path) -> str:
    """Extract plain text from a PDF or text file."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks up to a size limit, carrying a small overlap
    so a sentence split at a boundary survives intact in the next chunk."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for para in paras:
        if len(cur) + len(para) > max_chars and cur:
            out.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def build_store() -> int:
    """Rebuild the vector store from every document in the corpus folder."""
    docs_dir = pathlib.Path(os.environ.get("DOCS_DIR", "docs"))
    files = [p for p in docs_dir.glob("*") if p.suffix.lower() in (".txt", ".pdf")]
    if not files:
        sys.exit(f"No .txt or .pdf files in {docs_dir}/ — add documents there first.")

    import chromadb

    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("payer_docs")  # full rebuild — replace, not append
    except Exception:
        pass
    collection = client.get_or_create_collection(
        "payer_docs", metadata={"hnsw:space": "cosine"}
    )

    total = 0
    for f in sorted(files):
        chunks = chunk(read_document(f))
        if not chunks:
            print(f"  skip {f.name} (no extractable text)")
            continue
        collection.add(
            ids=[f"{f.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": f.name, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"  {f.name}: {len(chunks)} chunks")

    print(f"\nIndexed {total} chunks from {len(files)} documents into chroma_db/")
    return total


if __name__ == "__main__":
    build_store()
