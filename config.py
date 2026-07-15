"""Central config — everything env-var driven, nothing secret in code.

Values come from a local .env file (gitignored) or the shell environment.
boto3 picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
automatically once load_dotenv() has run.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Bedrock Knowledge Base (create in console, Lab 1) — required by ask.py / app.py
KB_ID = os.environ.get("KB_ID", "")

# Models — any current Claude + Titan embeddings work; copy exact ids from
# Bedrock console -> Model catalog if these have rotated.
CHAT_MODEL_ID = os.environ.get(
    "CHAT_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"
)
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Retrieval tuning knob — see README "Design decisions"
NUM_RESULTS = int(os.environ.get("NUM_RESULTS", "4"))
