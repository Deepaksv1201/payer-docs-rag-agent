"""Settings resolved once from the environment (.env or shell)."""
import os

from dotenv import load_dotenv

load_dotenv()

# Backend selection; defaults keep the app local and offline.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")            # ollama | bedrock
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "chroma")  # chroma | kb
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# AWS settings, used only when a Bedrock/KB backend is selected.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "us.amazon.nova-lite-v1:0")
KB_ID = os.environ.get("KB_ID", "")

# Chunks returned per query.
NUM_RESULTS = int(os.environ.get("NUM_RESULTS", "4"))
