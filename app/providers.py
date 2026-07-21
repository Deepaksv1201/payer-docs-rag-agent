"""Text-generation backends behind one interface, selected by config."""
from abc import ABC, abstractmethod

from app.config import AWS_REGION, CHAT_MODEL_ID, LLM_PROVIDER, OLLAMA_MODEL


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class OllamaProvider(LLMProvider):
    """Local generation via Ollama."""

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
    """Generation via Amazon Bedrock (Converse API)."""

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
    """Instantiate the configured (or named) generation backend."""
    name = name or LLM_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Options: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()
