"""RAG in one file, no frameworks — direct Bedrock calls plus NumPy.

Demonstrates the mechanics that LangChain and a vector database normally hide:
chunk, embed (Titan), retrieve by cosine similarity, generate (Bedrock).
Requires Bedrock quota. Usage: python minimal_rag.py "your question"
"""
import json
import pathlib
import sys

import boto3
import numpy as np

from app.config import AWS_REGION, CHAT_MODEL_ID

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

PROMPT = """Answer the question using ONLY the context below. If the answer is
not in the context, say "I don't have that information in the provided documents."

Context:
{context}

Question: {question}"""


def embed(text: str) -> np.ndarray:
    """Text to a Titan embedding vector."""
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID, body=json.dumps({"inputText": text})
    )
    return np.array(json.loads(resp["body"].read())["embedding"])


def chunk(text: str, max_chars: int = 1200) -> list[str]:
    """Pack paragraphs into chunks below a character limit."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > max_chars and cur:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    return out + ([cur.strip()] if cur.strip() else [])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — the comparison a vector database performs internally."""
    return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))


if __name__ == "__main__":
    question = sys.argv[1]

    chunks: list[str] = []
    for f in pathlib.Path("docs").glob("*.txt"):
        chunks += chunk(f.read_text(encoding="utf-8"))
    if not chunks:
        sys.exit("No .txt files in docs/")
    vectors = [embed(c) for c in chunks]

    qv = embed(question)
    scores = [cosine(qv, v) for v in vectors]
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:3]
    context = "\n\n".join(chunks[i] for i in top)

    resp = bedrock.converse(
        modelId=CHAT_MODEL_ID,
        messages=[{"role": "user", "content": [
            {"text": PROMPT.format(context=context, question=question)}
        ]}],
    )
    print(resp["output"]["message"]["content"][0]["text"])
    print("\n--- sources (cosine similarity) ---")
    for i in top:
        print(f"[{scores[i]:.3f}]", chunks[i][:120].replace("\n", " "))
