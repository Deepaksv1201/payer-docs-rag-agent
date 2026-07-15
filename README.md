# Payer Docs RAG Agent

RAG (Retrieval-Augmented Generation) system that answers questions over healthcare
payer documentation with **grounded, source-cited answers** — no hallucinated policy rules.

Built on the AWS stack I use in production daily (Lambda/S3/IAM background), extended
with the GenAI retrieval layer: Amazon Bedrock, Titan embeddings, and LangChain.

## Architecture

```
                  (indexing — once)                        (query time)
 payer PDFs ──► S3 ──► Bedrock Knowledge Base          Streamlit UI
                        ├─ chunking                         │ question
                        ├─ Titan embeddings                 ▼
                        └─ vector store  ◄──── LangChain retriever
                               │  top-k chunks               │
                               └─────────────────────────────┤
                                                             ▼
                                              grounding prompt template
                                                             │
                                                             ▼
                                            Claude (via Bedrock) ──► answer + sources
```

## What's in here

| File | Purpose |
|---|---|
| `rag_raw.py` | **RAG with zero frameworks** — chunking, embedding, cosine-similarity retrieval and grounded generation in ~40 lines of raw boto3. Proves nothing here is a black box. |
| `retrieve_test.py` | Retrieval in isolation — see exactly which chunks vector search returns before any generation happens |
| `ask.py` | The production-style pipeline: LangChain LCEL chain (`retriever \| prompt \| llm \| parser`) over a Bedrock Knowledge Base |
| `app.py` | Streamlit chat UI with an expandable sources panel — every answer shows the chunks it was grounded in |
| `config.py` | Single place for region / KB id / model ids (env-var driven, no secrets in code) |

## Design decisions

- **Managed Knowledge Base over hand-rolled vector store:** speed to production; the
  trade-off is less chunking control. `rag_raw.py` demonstrates I know what the KB
  replaces (raw embedding calls + cosine search) and could swap to OpenSearch/pgvector
  if retrieval quality demanded it.
- **`numberOfResults = 4`:** tested 1 / 4 / 10 — one chunk misses nuance, ten dilutes
  the prompt with noise. Four balanced precision vs. context.
- **Grounding prompt with explicit refusal:** the chain answers *only* from retrieved
  context and says so when the documents don't contain the answer. Verified by testing
  with out-of-corpus questions (it invents rules without this — try it).
- **No credentials in code:** AWS auth via environment / `aws configure` only.
- **Public documents only:** CMS and public payer manuals — no PHI. (Production PHI
  handling would start with Bedrock's HIPAA eligibility + BAA, encryption and access
  control on the vector store, and PHI redaction before indexing.)

## Monitoring (CloudWatch)

- Bedrock publishes invocation metrics (latency, token counts, errors) to CloudWatch
  automatically — no setup, covered by the free tier at this scale.
- Optional: enable **Model invocation logging** in the Bedrock console to capture full
  request/response logs in CloudWatch Logs.
- The app logs per-query retrieval count and end-to-end latency (`ask.py`), giving a
  local view of the same signals.

## Running it

```bash
# 1. Prereqs: AWS account, model access enabled in Bedrock (Claude + Titan
#    embeddings) in your region; a Knowledge Base synced to an S3 bucket of docs.
pip install -r requirements.txt
aws configure                    # or export AWS_* env vars

# 2. Point config at your resources
export AWS_REGION=us-east-1
export KB_ID=YOUR_KNOWLEDGE_BASE_ID

# 3. Try the layers in order
python rag_raw.py "What are the eligibility verification requirements?"   # zero-framework version (uses docs/*.txt, no KB needed)
python retrieve_test.py "What are the eligibility verification requirements?"
python ask.py "What are the eligibility verification requirements?"
streamlit run app.py
```

## Screenshots

_(screenshots/ — app answering, sources expanded, KB console)_

---
**Sai Deepak S V** · Hyderabad · saideepak.2003@gmail.com
