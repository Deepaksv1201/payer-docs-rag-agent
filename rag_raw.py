"""RAG without frameworks — chunking, embedding, retrieval and grounded
generation using the Bedrock runtime API directly.

Usage: python rag_raw.py "your question"   (reads docs/*.txt)
"""
import json
import pathlib
import sys

import boto3
import numpy as np

from config import AWS_REGION, CHAT_MODEL_ID, EMBED_MODEL_ID

br = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def embed(text: str) -> np.ndarray:
    resp = br.invoke_model(
        modelId=EMBED_MODEL_ID, body=json.dumps({"inputText": text})
    )
    return np.array(json.loads(resp["body"].read())["embedding"])


def chunk(text: str, max_chars: int = 1500) -> list[str]:
    """Paragraph-packing chunker with a max size per chunk."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > max_chars and cur:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    return out + ([cur] if cur else [])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))


PROMPT = """Answer the question using ONLY the context below. If the answer
is not in the context, say "I don't have that information in the provided
documents."

Context:
{context}

Question: {question}"""


if __name__ == "__main__":
    question = sys.argv[1]

    # index
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
    resp = br.converse(
        modelId=CHAT_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [{"text": PROMPT.format(context=context, question=question)}],
        }],
    )
    print(resp["output"]["message"]["content"][0]["text"])

    print("\n--- sources ---")
    for i in top:
        print(f"[{scores[i]:.3f}]", chunks[i][:120].replace("\n", " "))
