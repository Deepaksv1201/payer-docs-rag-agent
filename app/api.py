"""HTTP interface for the RAG service — a thin FastAPI layer.

Holds no business logic: it validates requests, delegates to RagService, and
shapes the JSON response. Run with:  uvicorn app.api:app --port 8000
"""
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.service import RagService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="DocQ RAG API")
service = RagService()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_seconds: float
    provider: str


@app.get("/health")
def health():
    """Report liveness and whether the retrieval store answers a probe query."""
    try:
        service.retrieval.search("healthcheck", k=1)
        ready = True
    except Exception:
        ready = False
    return {
        "status": "ok" if ready else "degraded",
        "provider": service.provider.name,
        "retrieval_ready": ready,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question from the document corpus, with sources and latency."""
    result = service.answer(req.question)
    return AskResponse(
        answer=result.text,
        sources=[
            {"source": c.source, "preview": c.text[:200], "score": c.score}
            for c in result.sources
        ],
        latency_seconds=result.latency_seconds,
        provider=result.provider,
    )
