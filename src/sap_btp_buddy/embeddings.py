"""Environment-configured local embedding adapter.

The adapter deliberately keeps model loading and prefix policy in one reusable
module so notebooks, the CLI, and the UI cannot drift apart.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

from dotenv import load_dotenv

from .config import PROJECT_ROOT


@dataclass(frozen=True)
class EmbeddingSmokeResult:
    """Non-sensitive embedding verification metadata."""

    model_id: str
    revision: str | None
    dimensions: int
    input_limit: int
    document_prefix: str
    query_prefix: str
    elapsed_ms: float
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local embedding settings loaded from environment variables."""

    model_id: str
    revision: str | None
    device: str
    document_prefix: str
    query_prefix: str
    input_limit_override: int | None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        # Load the project-local token before SentenceTransformer contacts the Hub.
        # The token is consumed by huggingface_hub and is never returned or logged.
        load_dotenv(PROJECT_ROOT / ".env")
        model_id = os.getenv("BTP_EMBEDDING_MODEL", "intfloat/e5-small-v2").strip()
        if not model_id:
            raise ValueError("BTP_EMBEDDING_MODEL must not be empty")
        known_prefixes = {
            "intfloat/e5-small-v2": ("passage: ", "query: "),
            "BAAI/bge-small-en-v1.5": ("", ""),
        }
        default_document, default_query = known_prefixes.get(model_id, ("", ""))
        document_prefix = os.getenv("BTP_EMBEDDING_DOCUMENT_PREFIX") or default_document
        query_prefix = os.getenv("BTP_EMBEDDING_QUERY_PREFIX") or default_query
        raw_limit = os.getenv("BTP_EMBEDDING_INPUT_LIMIT", "").strip()
        input_limit = int(raw_limit) if raw_limit else None
        if input_limit is not None and input_limit <= 0:
            raise ValueError("BTP_EMBEDDING_INPUT_LIMIT must be positive")
        return cls(
            model_id=model_id,
            revision=os.getenv("BTP_EMBEDDING_REVISION") or None,
            device=os.getenv("BTP_EMBEDDING_DEVICE", "cpu"),
            document_prefix=document_prefix,
            query_prefix=query_prefix,
            input_limit_override=input_limit,
        )


class LocalEmbeddingAdapter:
    """Sentence-Transformers adapter with explicit model-card formatting."""

    def __init__(self, config: EmbeddingConfig | None = None, model: Any | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.config.model_id,
                revision=self.config.revision,
                device=self.config.device,
            )
        self.model = model
        self.input_limit = self._resolve_input_limit()
        self.revision = self._resolve_revision()
        self._verify_prefix_policy()

    def _resolve_input_limit(self) -> int:
        limits = [self.model.max_seq_length]
        tokenizer_limit = getattr(getattr(self.model, "tokenizer", None), "model_max_length", None)
        if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
            limits.append(tokenizer_limit)
        limit = min(int(value) for value in limits if isinstance(value, int) and value > 0)
        if self.config.input_limit_override is not None:
            limit = min(limit, self.config.input_limit_override)
        return limit

    def _resolve_revision(self) -> str | None:
        init_kwargs = getattr(getattr(self.model, "tokenizer", None), "init_kwargs", {})
        return self.config.revision or init_kwargs.get("_commit_hash")

    def _verify_prefix_policy(self) -> None:
        if self.config.model_id == "intfloat/e5-small-v2":
            if (self.config.document_prefix, self.config.query_prefix) != ("passage: ", "query: "):
                raise ValueError("intfloat/e5-small-v2 requires passage:/query: prefixes")

    def format_document(self, text: str) -> str:
        return f"{self.config.document_prefix}{text}"

    def format_query(self, text: str) -> str:
        return f"{self.config.query_prefix}{text}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [self.format_document(text) for text in texts]
        return self._encode(formatted)

    def _encode(self, formatted: Sequence[str]) -> list[list[float]]:
        batch_size = int(os.getenv("BTP_EMBEDDING_BATCH_SIZE", "64"))
        encoded = self.model.encode(formatted, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [[float(value) for value in vector] for vector in encoded]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.format_query(text)])[0]

    def smoke_call(self, text: str = "SAP BTP account hierarchy") -> EmbeddingSmokeResult:
        started = time.perf_counter()
        vector = self.embed_documents([text])[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return EmbeddingSmokeResult(
            model_id=self.config.model_id,
            revision=self.revision,
            dimensions=len(vector),
            input_limit=self.input_limit,
            document_prefix=self.config.document_prefix,
            query_prefix=self.config.query_prefix,
            elapsed_ms=round(elapsed_ms, 3),
        )


def build_embedding_adapter() -> LocalEmbeddingAdapter:
    """Build the configured local adapter."""

    return LocalEmbeddingAdapter()
