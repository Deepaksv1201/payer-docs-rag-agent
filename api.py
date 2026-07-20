"""FastAPI backend for the payer-docs RAG service.

Run:  uvicorn api:app --reload --port 8000
Try:  POST http://localhost:8000/ask   {"question": "what is an EOB?"}
"""
import logging
import time

from fastapi import FastAPI
from pydantic import BaseModel

from providers import get_provider
from retrieval import get_retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("payer-rag-api")

app = FastAPI(title="Payer Docs RAG API")

retrieval = get_retrieval()
llm = get_provider()

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say
"I don't have that information in the provided documents."

Context:
{context}

Question: {question}"""


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_seconds: float
    provider: str


@app.get("/health")
def health():
    return {"status": "ok", "provider": llm.name}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    chunks = retrieval.search(req.question)
    context = "\n\n".join(c.text for c in chunks)
    answer = llm.generate(PROMPT.format(context=context, question=req.question))
    elapsed = time.perf_counter() - start
    log.info("answered | chunks=%d | latency=%.2fs", len(chunks), elapsed)
    return AskResponse(
        answer=answer,
        sources=[
            {"source": c.source, "preview": c.text[:200], "score": c.score}
            for c in chunks
        ],
        latency_seconds=round(elapsed, 2),
        provider=llm.name,
    )
