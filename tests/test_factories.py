"""Factory selection and error handling — no live backends required."""
import pytest

from app import providers, retrieval


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        providers.get_provider("does-not-exist")


def test_unknown_retrieval_raises():
    with pytest.raises(ValueError, match="Unknown retrieval"):
        retrieval.get_retrieval("does-not-exist")


def test_provider_registry_has_both_backends():
    assert set(providers._PROVIDERS) == {"ollama", "bedrock"}


def test_chunk_dataclass_defaults():
    c = retrieval.Chunk(text="hi", source="a.txt")
    assert c.score is None
    assert c.text == "hi"
