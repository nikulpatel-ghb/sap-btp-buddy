from types import SimpleNamespace

import pytest

from sap_btp_buddy.embeddings import EmbeddingConfig, LocalEmbeddingAdapter
from sap_btp_buddy.generation_adapter import GenerationConfig, OpenAICompatibleGenerationAdapter, SecretValue


class FakeEmbeddingModel:
    max_seq_length = 512
    tokenizer = SimpleNamespace(model_max_length=512, init_kwargs={"_commit_hash": "abc123"})

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_e5_prefixes_and_smoke_metadata_are_model_card_compatible():
    adapter = LocalEmbeddingAdapter(
        EmbeddingConfig("intfloat/e5-small-v2", "abc123", "cpu", "passage: ", "query: ", None),
        FakeEmbeddingModel(),
    )
    result = adapter.smoke_call()
    assert adapter.format_document("text").startswith("passage: ")
    assert adapter.format_query("question").startswith("query: ")
    assert result.dimensions == 3
    assert result.revision == "abc123"
    assert result.usage is None


def test_e5_rejects_wrong_prefix_configuration():
    with pytest.raises(ValueError, match="requires"):
        LocalEmbeddingAdapter(
            EmbeddingConfig("intfloat/e5-small-v2", None, "cpu", "", "", None),
            FakeEmbeddingModel(),
        )


class FakeCompletions:
    def create(self, **kwargs):
        assert kwargs["model"] == "test/model"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
            usage=None,
        )


class FakeClient:
    chat = SimpleNamespace(completions=FakeCompletions())


def test_generation_smoke_does_not_turn_unknown_usage_into_zero():
    adapter = OpenAICompatibleGenerationAdapter(
        GenerationConfig("secret-not-printed", "https://example.invalid/v1/", "test/model", 1, 0, 0),
        FakeClient(),
    )
    result = adapter.smoke_call()
    assert result.model_id == "test/model"
    assert result.usage is None


def test_generation_config_redacts_api_key_from_string_representations():
    config = GenerationConfig("super-secret-value", "https://example.invalid/v1/", "test/model", 1, 0, 0)
    assert isinstance(config.api_key, SecretValue)
    assert "super-secret-value" not in repr(config)
    assert "super-secret-value" not in str(config.api_key)
    assert repr(config.api_key) == "SecretValue(<redacted>)"
