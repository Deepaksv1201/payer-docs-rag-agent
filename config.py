"""Central config — everything env-var driven, nothing secret in code.

Values come from a local .env file (gitignored) or the shell environment.
Backend selection:
    LLM_PROVIDER       ollama (default) | bedrock
    RETRIEVAL_BACKEND  chroma (default) | kb
"""
import os

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Bedrock Knowledge Base — required only when RETRIEVAL_BACKEND=kb
KB_ID = os.environ.get("KB_ID", "")

# Bedrock chat model — required only when LLM_PROVIDER=bedrock
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "us.amazon.nova-lite-v1:0")

# Retrieval tuning knob — see README "Design decisions"
NUM_RESULTS = int(os.environ.get("NUM_RESULTS", "4"))
