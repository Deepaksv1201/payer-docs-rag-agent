"""Text-generation backends behind a single interface.

Each provider knows only how to turn a prompt into text. The rest of the app
depends on the LLMProvider interface, so a new model source is one subclass
and switching between them is a configuration value.
"""
from abc import ABC, abstractmethod

from app.config import AWS_REGION, CHAT_MODEL_ID, LLM_PROVIDER, OLLAMA_MODEL


class LLMProvider(ABC):
    """Contract every generation backend implements."""

    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class OllamaProvider(LLMProvider):
    """Generation from a model running locally via Ollama — free and offline."""

    name = "ollama"

    def __init__(self):
        import ollama

        self._client = ollama
        self._model = OLLAMA_MODEL

    def generate(self, prompt: str) -> str:
        resp = self._client.chat(
            model=self._model, messages=[{"role": "user", "content": prompt}]
        )
        return resp["message"]["content"]


class BedrockProvider(LLMProvider):
    """Generation from Amazon Bedrock (Nova / Claude) via the Converse API."""

    name = "bedrock"

    def __init__(self):
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        self._model = CHAT_MODEL_ID

    def generate(self, prompt: str) -> str:
        resp = self._client.converse(
            modelId=self._model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        return resp["output"]["message"]["content"][0]["text"]


_PROVIDERS = {"ollama": OllamaProvider, "bedrock": BedrockProvider}


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the generation backend named by the argument or configuration."""
    name = name or LLM_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Options: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()
