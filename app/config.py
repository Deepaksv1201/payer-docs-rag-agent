"""Settings, resolved once from the environment (a local .env or the shell).

Keeping every tunable in one place means backends, models, and credentials
are chosen by configuration rather than being hard-wired in the code.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Which backends to use. Defaults keep the app fully local and offline.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")       # ollama | bedrock
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "chroma")  # chroma | kb

# Local generation model (Ollama).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# AWS settings, consulted only when a Bedrock/KB backend is selected.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "us.amazon.nova-lite-v1:0")
KB_ID = os.environ.get("KB_ID", "")

# How many chunks retrieval returns per question.
NUM_RESULTS = int(os.environ.get("NUM_RESULTS", "4"))
