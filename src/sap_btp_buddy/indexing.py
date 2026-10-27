"""Persistent Chroma and BM25 indexing with compatibility validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from rank_bm25 import BM25Okapi

from .models import Chunk, SearchResult


class EmbeddingLike(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


SCHEMA_VERSION = "index-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _chunk_records(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            "metadata": chunk.metadata,
        }
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    ]


def _config_fingerprint(chunks: list[Chunk]) -> dict[str, Any]:
    keys = {"chunk_strategy", "chunker_version", "max_input_tokens", "requested_overlap_tokens", "token_limit_status", "embedding_text_has_context"}
    return {key: sorted({chunk.metadata.get(key) for chunk in chunks}, key=str) for key in sorted(keys)}


def _model_fingerprint(embedding_model: str, embedding_function: EmbeddingLike, dimension: int) -> dict[str, Any]:
    config = getattr(embedding_function, "config", None)
    return {
        "provider": type(embedding_function).__module__,
        "model_id": embedding_model,
        "revision": getattr(embedding_function, "revision", None) or getattr(config, "revision", None),
        "input_limit": getattr(embedding_function, "input_limit", None),
        "document_prefix": getattr(config, "document_prefix", None),
        "query_prefix": getattr(config, "query_prefix", None),
        "dimensions": dimension,
        "normalized": True,
        "distance": "cosine",
    }


def _index_identity(corpus_fingerprint: str, config_fingerprint: str, model_fingerprint: str, retrieval: str) -> str:
    return hashlib.sha256(_canonical({"schema_version": SCHEMA_VERSION, "corpus_fingerprint": corpus_fingerprint, "config_fingerprint": config_fingerprint, "model_fingerprint": model_fingerprint, "retrieval": retrieval}).encode("utf-8")).hexdigest()[:32]


def _metadata_json(metadata: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metadata, default=str))


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize prose and technical strings without discarding CLI syntax.

    Lower-casing is the only normalization. Runs of letters, digits, and
    common command/path punctuation remain one token, so options such as
    ``--subaccount`` and paths such as ``/path/to/file`` remain searchable.
    """

    return re.findall(r"[\w./:@=+~-]+", text.casefold(), flags=re.UNICODE)


def rrf_fuse(
    dense_ranked_ids: Iterable[str],
    bm25_ranked_ids: Iterable[str],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse explicit ranked chunk IDs with reciprocal-rank fusion.

    IDs are deduplicated within and across input rankings. Missing ranks
    contribute zero. Ties are deterministic by canonical chunk ID.
    """

    if top_k <= 0 or rrf_k <= 0:
        raise ValueError("top_k and rrf_k must be positive")
    scores: dict[str, float] = {}
    for ranked_ids in (dense_ranked_ids, bm25_ranked_ids):
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranked_ids, 1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_k]


def source_document_id(chunk: Chunk) -> str:
    """Return the stable source identity used for deduplicated reporting."""

    metadata = chunk.metadata
    return f"{metadata.get('repository')}@{metadata.get('commit')}:{metadata.get('path')}"


@dataclass
class PersistentIndex:
    """Loaded, compatibility-validated dense/lexical index."""

    path: Path
    manifest: dict[str, Any]
    collection: Any
    bm25: BM25Okapi
    chunks: dict[str, Chunk]
    embedding_function: EmbeddingLike

    def retrieve(self, query: str, *, mode: str = "hybrid", top_k: int = 5, where: dict[str, Any] | None = None) -> list[SearchResult]:
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("mode must be dense, bm25, or hybrid")
        eligible = {chunk_id for chunk_id, chunk in self.chunks.items() if _matches(chunk, where)}
        dense_rank: list[str] = []
        dense_scores: dict[str, float] = {}
        if mode in {"dense", "hybrid"}:
            result = self.collection.query(query_embeddings=[self.embedding_function.embed_query(query)], n_results=max(top_k, len(eligible)), include=["distances"])
            dense_rank = [chunk_id for chunk_id in result["ids"][0] if chunk_id in eligible]
            dense_scores = {chunk_id: 1.0 - float(distance) for chunk_id, distance in zip(result["ids"][0], result["distances"][0])}
        bm25_rank: list[str] = []
        if mode in {"bm25", "hybrid"}:
            scores = self.bm25.get_scores(tokenize_bm25(query))
            ordered = sorted(enumerate(scores), key=lambda item: (-item[1], self.manifest["chunk_ids"][item[0]]))
            bm25_rank = [self.manifest["chunk_ids"][index] for index, _ in ordered if self.manifest["chunk_ids"][index] in eligible]
        if mode == "dense":
            ordered_ids = dense_rank
        elif mode == "bm25":
            ordered_ids = bm25_rank
        else:
            fused: dict[str, float] = {}
            fused = dict(rrf_fuse(dense_rank, bm25_rank, top_k=max(top_k, len(eligible))))
            ordered_ids = list(fused)
        if mode == "dense":
            return [SearchResult(self.chunks[chunk_id], dense_scores[chunk_id]) for chunk_id in ordered_ids[:top_k]]
        if mode == "hybrid":
            return [SearchResult(self.chunks[chunk_id], fused[chunk_id]) for chunk_id in ordered_ids[:top_k]]
        return [SearchResult(self.chunks[chunk_id], float(self.bm25.get_scores(tokenize_bm25(query))[self.manifest["chunk_ids"].index(chunk_id)])) for chunk_id in ordered_ids[:top_k]]


def _matches(chunk: Chunk, where: dict[str, Any] | None) -> bool:
    return not where or all(chunk.metadata.get(key) == value for key, value in where.items())


def _resolve_embedding_function(embedding_function: EmbeddingLike | None) -> EmbeddingLike:
    if embedding_function is not None:
        return embedding_function
    from .embeddings import build_embedding_adapter

    return build_embedding_adapter()


def _embed_documents_with_resume(target: Path, chunks: list[Chunk], embedder: EmbeddingLike, dimension: int) -> list[list[float]]:
    """Embed in durable batches so an interrupted local build can resume."""

    cache_path = target / "embeddings.jsonl"
    expected_ids = {chunk.chunk_id for chunk in chunks}
    cached: dict[str, list[float]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                vector = row.get("embedding")
                chunk_id = row.get("chunk_id")
                if chunk_id in expected_ids and isinstance(vector, list) and len(vector) == dimension:
                    cached[chunk_id] = [float(value) for value in vector]
    batch_size = max(1, int(os.getenv("BTP_EMBEDDING_BATCH_SIZE", "64")))
    missing = [chunk for chunk in chunks if chunk.chunk_id not in cached]
    if missing:
        with cache_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                vectors = embedder.embed_documents([chunk.text for chunk in batch])
                if len(vectors) != len(batch) or any(len(vector) != dimension for vector in vectors):
                    raise ValueError("embedding function returned an invalid batch")
                for chunk, vector in zip(batch, vectors):
                    cached[chunk.chunk_id] = [float(value) for value in vector]
                    handle.write(json.dumps({"chunk_id": chunk.chunk_id, "embedding": cached[chunk.chunk_id]}, separators=(",", ":")) + "\n")
                handle.flush()
    if set(cached) != expected_ids:
        raise ValueError("embedding cache does not cover the canonical chunk set")
    return [cached[chunk.chunk_id] for chunk in chunks]


def build_index(chunks: Iterable[Chunk], index_dir: Path, *, embedding_model: str, retrieval: str = "hybrid", embedding_function: EmbeddingLike | None = None) -> Path:
    """Build or idempotently reuse an identity-specific Chroma/BM25 index."""

    if retrieval not in {"dense", "bm25", "hybrid"}:
        raise ValueError("retrieval must be dense, bm25, or hybrid")
    chunk_list = sorted(list(chunks), key=lambda item: item.chunk_id)
    if not chunk_list or len({chunk.chunk_id for chunk in chunk_list}) != len(chunk_list):
        raise ValueError("index input must contain non-empty unique chunk IDs")
    embedder = _resolve_embedding_function(embedding_function)
    dimension = len(embedder.embed_query("index compatibility probe"))
    if dimension <= 0:
        raise ValueError("embedding function returned an invalid probe vector")
    records = _chunk_records(chunk_list)
    corpus_fingerprint = _sha(records)
    chunk_config = _config_fingerprint(chunk_list)
    model_config = _model_fingerprint(embedding_model, embedder, dimension)
    config_fingerprint = _sha(chunk_config)
    model_fingerprint = _sha(model_config)
    index_id = _index_identity(corpus_fingerprint, config_fingerprint, model_fingerprint, retrieval)
    root = Path(index_dir)
    target = root if (root / "manifest.json").exists() else root / index_id
    target.mkdir(parents=True, exist_ok=True)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "index_id": index_id,
        "retrieval": retrieval,
        "corpus_fingerprint": corpus_fingerprint,
        "config_fingerprint": config_fingerprint,
        "model_fingerprint": model_fingerprint,
        "chunk_config": chunk_config,
        "model_config": model_config,
        "chunk_ids": [chunk.chunk_id for chunk in chunk_list],
        "chunk_count": len(chunk_list),
        "text_hashes": {chunk.chunk_id: record["text_sha256"] for chunk, record in zip(chunk_list, records)},
    }
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _assert_compatible(existing, expected)
        if (target / "build.complete").exists():
            _validate_persisted_store(target, expected, embedder)
            return target
    else:
        manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
    embeddings = _embed_documents_with_resume(target, chunk_list, embedder, dimension)
    (target / "chunks.jsonl").write_text("".join(json.dumps({"chunk_id": c.chunk_id, "text": c.text, "metadata": _metadata_json(c.metadata)}, sort_keys=True) + "\n" for c in chunk_list), encoding="utf-8")
    (target / "bm25.json").write_text(json.dumps({"chunk_ids": [c.chunk_id for c in chunk_list], "tokens": [tokenize_bm25(c.text) for c in chunk_list]}, sort_keys=True), encoding="utf-8")
    _persist_chroma(target, chunk_list, embeddings, expected)
    manifest_path.write_text(json.dumps({**expected, "built_at": datetime.now(timezone.utc).isoformat(), "storage_counts": {"dense": len(chunk_list), "bm25": len(chunk_list), "duplicate_chunk_ids": 0}}, indent=2, sort_keys=True), encoding="utf-8")
    (target / "build.complete").write_text("ok\n", encoding="utf-8")
    return target


def _persist_chroma(target: Path, chunks: list[Chunk], embeddings: list[list[float]], manifest: dict[str, Any]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(target / "dense"))
    collection = client.get_or_create_collection(name="chunks", configuration={"hnsw": {"space": "cosine"}}, metadata={"index_id": manifest["index_id"], "schema_version": SCHEMA_VERSION, "dimension": manifest["model_config"]["dimensions"]})
    if collection.count() and collection.metadata.get("index_id") != manifest["index_id"]:
        raise ValueError("existing Chroma collection has incompatible index identity")
    if collection.metadata.get("dimension") != manifest["model_config"]["dimensions"]:
        raise ValueError("existing Chroma collection has incompatible vector dimensions")
    batch_size = max(1, min(int(os.getenv("BTP_CHROMA_BATCH_SIZE", "5000")), 5461))
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch_chunks = chunks[start:end]
        collection.upsert(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=embeddings[start:end],
            documents=[c.text for c in batch_chunks],
            metadatas=[_metadata_json(c.metadata) for c in batch_chunks],
        )
    if collection.count() != len(chunks):
        raise ValueError("Chroma count does not match the canonical chunk set")


def _assert_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    fields = ("schema_version", "index_id", "retrieval", "corpus_fingerprint", "config_fingerprint", "model_fingerprint", "chunk_count", "chunk_ids", "text_hashes")
    mismatches = [field for field in fields if existing.get(field) != expected.get(field)]
    if mismatches:
        raise ValueError(f"incompatible existing index manifest: {', '.join(mismatches)}")


def _validate_persisted_store(path: Path, manifest: dict[str, Any], embedder: EmbeddingLike) -> None:
    """Validate persistence without embedding any documents."""

    chunks: dict[str, Chunk] = {}
    with (path / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunks[row["chunk_id"]] = Chunk(row["chunk_id"], row["text"], row["metadata"])
    if sorted(chunks) != sorted(manifest["chunk_ids"]):
        raise ValueError("cached chunk IDs do not match manifest")
    if len(chunks) != manifest["chunk_count"] or len(chunks) != len(set(chunks)):
        raise ValueError("cached chunk count or duplicate count is incompatible")
    bm25_payload = json.loads((path / "bm25.json").read_text(encoding="utf-8"))
    if bm25_payload["chunk_ids"] != manifest["chunk_ids"]:
        raise ValueError("cached BM25 IDs do not match manifest")
    import chromadb

    collection = chromadb.PersistentClient(path=str(path / "dense")).get_collection(name="chunks")
    if collection.count() != manifest["chunk_count"] or collection.metadata.get("dimension") != manifest["model_config"]["dimensions"]:
        raise ValueError("cached Chroma count or dimension is incompatible")
    if sorted(collection.get(include=[])["ids"]) != sorted(manifest["chunk_ids"]):
        raise ValueError("cached Chroma IDs do not match manifest")


def load_index(index_dir: Path, embedding_function: EmbeddingLike | None = None) -> PersistentIndex:
    """Load and validate a persistent index in a fresh process."""

    path = Path(index_dir)
    if not (path / "manifest.json").exists() or not (path / "build.complete").exists():
        raise ValueError("index is incomplete or missing manifest")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    chunks: dict[str, Chunk] = {}
    with (path / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunks[row["chunk_id"]] = Chunk(row["chunk_id"], row["text"], row["metadata"])
    if sorted(chunks) != sorted(manifest["chunk_ids"]):
        raise ValueError("chunk store IDs do not match manifest")
    actual_text_hashes = {chunk_id: hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() for chunk_id, chunk in chunks.items()}
    if actual_text_hashes != manifest["text_hashes"]:
        raise ValueError("chunk text hashes do not match manifest")
    bm25_payload = json.loads((path / "bm25.json").read_text(encoding="utf-8"))
    if bm25_payload["chunk_ids"] != manifest["chunk_ids"]:
        raise ValueError("BM25 IDs do not match manifest order")
    import chromadb

    collection = chromadb.PersistentClient(path=str(path / "dense")).get_collection(name="chunks")
    if collection.count() != manifest["chunk_count"] or collection.metadata.get("index_id") != manifest["index_id"] or collection.metadata.get("dimension") != manifest["model_config"]["dimensions"]:
        raise ValueError("Chroma collection is incompatible with manifest")
    dense_ids = collection.get(include=[])['ids']
    if sorted(dense_ids) != sorted(manifest["chunk_ids"]):
        raise ValueError("Chroma IDs do not match manifest")
    embedder = _resolve_embedding_function(embedding_function)
    model_config = manifest["model_config"]
    actual_dimension = len(embedder.embed_query("compatibility check"))
    if actual_dimension != model_config["dimensions"]:
        raise ValueError("embedding function dimension does not match manifest")
    runtime_config = getattr(embedder, "config", None)
    runtime_model_id = getattr(runtime_config, "model_id", None) or getattr(embedder, "model_id", None)
    if runtime_model_id is not None and runtime_model_id != model_config["model_id"]:
        raise ValueError("embedding model ID does not match manifest")
    runtime_revision = getattr(embedder, "revision", None) or getattr(runtime_config, "revision", None)
    if runtime_revision is not None and model_config["revision"] is not None and runtime_revision != model_config["revision"]:
        raise ValueError("embedding revision does not match manifest")
    for attribute, field in (("input_limit", "input_limit"), ("document_prefix", "document_prefix"), ("query_prefix", "query_prefix")):
        value = getattr(embedder, attribute, None) or getattr(runtime_config, attribute, None)
        if value is not None and model_config[field] is not None and value != model_config[field]:
            raise ValueError(f"embedding {field} does not match manifest")
    return PersistentIndex(path, manifest, collection, BM25Okapi(bm25_payload["tokens"]), chunks, embedder)
