"""LLM provider factory — generation backends behind one interface.

Adding a provider = one subclass with a generate() method.
Selected via LLM_PROVIDER env var (see config.py).
"""
import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self):
        import ollama

        self._client = ollama
        self._model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    def generate(self, prompt: str) -> str:
        resp = self._client.chat(
            model=self._model, messages=[{"role": "user", "content": prompt}]
        )
        return resp["message"]["content"]


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self):
        import boto3

        from config import AWS_REGION, CHAT_MODEL_ID

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
    name = name or os.environ.get("LLM_PROVIDER", "ollama")
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Options: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()
