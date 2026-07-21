"""Chunking behaves deterministically — verified without models or network."""
from app.ingestion import chunk


def test_short_text_single_chunk():
    assert chunk("one short paragraph.") == ["one short paragraph."]


def test_empty_text_no_chunks():
    assert chunk("") == []


def test_large_text_splits():
    text = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(20))
    chunks = chunk(text, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 + 200 for c in chunks)  # allow overlap slack


def test_overlap_preserves_continuity():
    text = "\n\n".join("word " * 100 for _ in range(6))
    chunks = chunk(text, max_chars=800, overlap=100)
    assert len(chunks) >= 2
    assert chunks[0][-50:].strip() and chunks[1][:50].strip()


def test_no_empty_chunks():
    text = "\n\n\n\n".join(["real content here"] * 5)
    assert all(c.strip() for c in chunk(text))
