import hashlib

import pytest

from sap_btp_buddy.indexing import build_index, load_index, rrf_fuse, tokenize_bm25
from sap_btp_buddy.models import Chunk


class DeterministicEmbedder:
    revision = "fixture-revision"
    input_limit = 512

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vector(text)

    @staticmethod
    def _vector(text):
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[0] / 255, digest[1] / 255, digest[2] / 255]


def _chunks():
    return [
        Chunk("chunk-a", "account hierarchy and global account", {"repository": "SAP-docs/test", "commit": "abc", "path": "a.md", "topic": "account"}),
        Chunk("chunk-b", "btp login and cloud foundry", {"repository": "SAP-docs/test", "commit": "abc", "path": "b.md", "topic": "cli"}),
    ]


def test_identical_build_is_idempotent_and_fresh_load_retrieves_same_ids(tmp_path):
    embedder = DeterministicEmbedder()
    first = build_index(_chunks(), tmp_path / "indexes", embedding_model="fixture/model", embedding_function=embedder)
    before = load_index(first, embedder)
    first_ids = [result.chunk.chunk_id for result in before.retrieve("account", mode="hybrid", top_k=2)]
    second = build_index(_chunks(), tmp_path / "indexes", embedding_model="fixture/model", embedding_function=embedder)
    assert first == second
    assert embedder.document_calls == 1
    fresh = load_index(second, DeterministicEmbedder())
    assert fresh.manifest["chunk_count"] == 2
    assert [result.chunk.chunk_id for result in fresh.retrieve("account", mode="hybrid", top_k=2)] == first_ids


def test_incompatible_model_does_not_mix_with_existing_collection(tmp_path):
    root = tmp_path / "indexes"
    build_index(_chunks(), root, embedding_model="fixture/model", embedding_function=DeterministicEmbedder())
    with pytest.raises(ValueError, match="incompatible"):
        build_index(_chunks(), root / next(root.iterdir()).name, embedding_model="other/model", embedding_function=DeterministicEmbedder())


def test_metadata_filter_is_applied_to_loaded_index(tmp_path):
    index = build_index(_chunks(), tmp_path / "indexes", embedding_model="fixture/model", embedding_function=DeterministicEmbedder())
    loaded = load_index(index, DeterministicEmbedder())
    results = loaded.retrieve("btp", mode="bm25", top_k=5, where={"topic": "cli"})
    assert results
    assert all(result.chunk.metadata["topic"] == "cli" for result in results)


def test_bm25_tokenizer_preserves_technical_terms():
    tokens = tokenize_bm25("btp create --subaccount /org/path --format=json")
    assert "btp" in tokens
    assert "--subaccount" in tokens
    assert "/org/path" in tokens
    assert "--format=json" in tokens


def test_rrf_fuses_ranked_ids_deduplicates_and_tie_breaks():
    fused = rrf_fuse(["b", "a", "a"], ["a", "c"], top_k=3, rrf_k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] > fused[1][1]
