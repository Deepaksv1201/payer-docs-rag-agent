# Payer Docs RAG Agent

Grounded Q&A over healthcare payer documentation — answers come only from the
document corpus, with sources and similarity scores shown for every response.

Three-tier design with **pluggable backends**: swap the LLM provider or the
vector store with an environment variable, zero code changes.

## Architecture

```
                Streamlit chat UI  (app.py, :8501)
                        │ HTTP
                FastAPI backend    (api.py, :8000)
                        │
        ┌───────────────┴────────────────┐
   Retrieval factory              LLM provider factory
   (retrieval.py)                 (providers.py)
        │                                │
  ┌─────┴──────┐                  ┌──────┴──────┐
  │ Chroma     │ local            │ Ollama      │ local
  │ Bedrock KB │ AWS              │ Bedrock     │ AWS
  └────────────┘                  └─────────────┘
```

| File | Purpose |
|---|---|
| `rag_raw.py` | The whole idea in one file — chunking, embeddings, cosine retrieval, grounded generation. No frameworks, no vector DB. |
| `ingest.py` | Offline ingestion: `docs/*.txt` → chunks → Chroma vector store |
| `retrieval.py` | Retrieval interface + Chroma / Bedrock Knowledge Base implementations |
| `providers.py` | LLM interface + Ollama / Bedrock implementations (adding a provider = one subclass) |
| `api.py` | FastAPI service: `POST /ask`, `GET /health`, per-query latency logging |
| `app.py` | Streamlit chat with history, sources and latency per answer |
| `retrieve_test.py` | Debug tool: see raw retrieval output for any question |

## Quickstart (fully local — no cloud account needed)

```bash
pip install -r requirements.txt
# install Ollama (ollama.com), then:
ollama pull llama3.2:3b

python ingest.py                 # build the vector store from docs/
uvicorn api:app --port 8000      # terminal 1 — backend
streamlit run app.py             # terminal 2 — chat UI
```

## Switching to AWS

Same code, different config in `.env`:

```
LLM_PROVIDER=bedrock         # generation via Amazon Bedrock (Nova / Claude)
RETRIEVAL_BACKEND=kb         # retrieval via a Bedrock Knowledge Base
KB_ID=<your knowledge base id>
```

## Design decisions

- **Grounding with explicit refusal** — the prompt forbids answering outside the
  retrieved context and requires a refusal when the context doesn't contain the
  answer. Verified by testing out-of-corpus questions.
- **Factory pattern for models and retrieval** — dependency inversion keeps the
  API layer ignorant of which backend serves it; local dev and cloud prod are a
  config flip apart.
- **Offline ingestion vs online query path** — indexing is a batch job
  (`ingest.py`), the query path stays latency-bound (~seconds). They fail and
  scale independently.
- **Cosine similarity space** in Chroma so retrieval scores are interpretable
  (1.0 = identical meaning).
- **Rate-limit hardening** — developed against near-zero Bedrock quotas on a fresh
  AWS account: exponential backoff, request pacing, and disabling SDK retry
  amplification on the ingestion path.
- **No secrets in code** — configuration via `.env` (gitignored); `.env.example`
  documents required variables.
- **Public documents only** (CMS/insurance-domain reference material, no PHI).
  Production PHI handling would start with Bedrock's HIPAA eligibility (BAA),
  encryption and access control on the vector store, and redaction before indexing.

## Monitoring

The API logs retrieval count and end-to-end latency per query. On the AWS
backends, Bedrock publishes invocation metrics (latency, token counts, errors)
to CloudWatch automatically.

---
**Sai Deepak S V** · Hyderabad · saideepak.2003@gmail.com
