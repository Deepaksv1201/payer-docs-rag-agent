# DocQ — Document Intelligence RAG

Grounded question-answering over an arbitrary document corpus. Answers are
constrained to the supplied documents, returned with source citations and
similarity scores, and refused when the corpus does not contain the answer.

The corpus is domain-independent: healthcare, telecom, legal, or any set of
PDFs/text placed in `docs/`. The LLM provider and vector store are each
selected by configuration, so the same code runs fully local (Ollama + Chroma)
or on AWS (Bedrock + Knowledge Base).

## Architecture

Layered by responsibility — presentation is thin, business logic is isolated,
and the external integrations sit behind interfaces:

```
   streamlit_app.py         presentation — chat UI
        │ HTTP
   app/api.py               presentation — FastAPI routes (/ask, /health)
        │
   app/service.py           business logic — retrieve → ground → generate
        │
   ┌────┴─────────────┐
   app/retrieval.py    app/providers.py     integrations
   Chroma │ Bedrock KB  Ollama │ Bedrock
```

```
app/
  api.py         thin HTTP layer over the service
  service.py     the RAG orchestration (reused by API, UI, eval)
  retrieval.py   vector-search backends (Chroma / Bedrock KB)
  providers.py   generation backends (Ollama / Bedrock)
  ingestion.py   builds the vector store from docs/
  config.py      environment-driven settings
streamlit_app.py chat UI
evaluate.py      offline quality evaluation
tests/           unit tests
```

Two paths, deliberately decoupled:

- **Ingestion** (`app/ingestion.py`) — offline batch. Documents are chunked,
  embedded, and written to the vector store. Triggered by corpus changes, not
  by the service lifecycle.
- **Query** (`app/service.py` behind `app/api.py`) — online, stateless.
  Retrieves the nearest chunks and generates a grounded answer. Reads only;
  scales horizontally.

Nothing above the integration layer names a concrete backend — the service
depends on the `Retrieval` and `LLMProvider` interfaces, so local and cloud
differ by configuration alone.

## Running

Prerequisites: Python 3.11+, and [Ollama](https://ollama.com) with a pulled
model (`ollama pull llama3.2:3b`) for local generation.

```bash
pip install -r requirements.txt

# 1. add documents to docs/  (.pdf or .txt)
# 2. build the vector store
python -m app.ingestion

# 3. run the service and UI (separate processes)
uvicorn app.api:app --port 8000
streamlit run streamlit_app.py
```

Container alternative: `docker compose up`.

## Configuration

Copy `.env.example` to `.env`. Backends are selected by environment variable:

| Variable | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `bedrock` |
| `RETRIEVAL_BACKEND` | `chroma` | `chroma`, `kb` |
| `OLLAMA_MODEL` | `llama3.2:3b` | any local Ollama model |
| `KB_ID` / `CHAT_MODEL_ID` | — | required for the AWS backends |

Switching to AWS is configuration only — no code changes.

## Evaluation

`evaluate.py` measures retrieval and generation separately against a fixed
question set (`eval_set.json`): retrieval hit-rate, answer faithfulness, a
refusal probe for out-of-corpus questions, and median latency. The bundled
question set targets the author's healthcare corpus; a new corpus needs a
matching set.

## Testing

`pytest tests/` — unit coverage for chunking and the backend factories. CI runs
on every push (`.github/workflows/ci.yml`).

## Design notes

- **Grounding with explicit refusal** — the prompt constrains answers to
  retrieved context; the refusal probe verifies it.
- **Factory pattern** — providers and retrieval backends behind interfaces;
  adding one is a single subclass, switching one is an env var.
- **Ingestion/query separation** — batch indexing vs. stateless serving, so
  redeploys are fast and re-indexing is independent of uptime.
- **Cosine-space store** so surfaced similarity scores are interpretable.
- **No secrets in source** — configuration via gitignored `.env`.
- **Public documents only** in any shared corpus; a PHI deployment would add
  BAA-covered infrastructure (Bedrock), encryption and access control on the
  store, and redaction before indexing.

## Limitations

- Small local models follow the refusal instruction less reliably than hosted
  models; the eval refusal probe makes this measurable.
- Retrieval is dense-vector only; hybrid search and a reranker would improve
  precision on exact identifiers (e.g. codes).
- Chunking is structure-agnostic; heading-aware chunking would raise hit-rate.


