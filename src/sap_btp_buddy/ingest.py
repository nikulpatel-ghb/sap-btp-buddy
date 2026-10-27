"""Deterministic document loading, metadata, and cleaning entry points."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .cleaning import clean_document
from .corpus import build_github_source_url, normalized_relative_path
from .models import CorpusDocument


def ingest_documents(
    paths: Iterable[Path],
    *,
    repository: str | None = None,
    commit: str | None = None,
    repository_root: Path | None = None,
    allowed_paths: Iterable[str] | None = None,
    topic_by_path: Mapping[str, str] | None = None,
    ingested_at_utc: str | None = None,
) -> list[CorpusDocument]:
    """Load and clean source documents while preserving provenance.

    The concrete Markdown cleaning policy will be implemented behind this public
    function so notebooks and UI code never duplicate ingestion logic.
    """

    root = repository_root.resolve() if repository_root is not None else None
    allowed = set(allowed_paths) if allowed_paths is not None else None
    timestamp = ingested_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ordered_paths = sorted(
        {Path(path) for path in paths},
        key=lambda path: normalized_relative_path(path, root) if root else path.as_posix(),
    )
    documents: list[CorpusDocument] = []
    for path in ordered_paths:
        relative_path = normalized_relative_path(path, root) if root else path.name
        if allowed is not None and relative_path not in allowed:
            continue
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        metadata: dict[str, object] = {
            "path": relative_path,
            "ingestion_time_utc": timestamp,
            "raw_content_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
        if repository is not None and commit is not None:
            metadata["repository"] = repository
            metadata["commit"] = commit
            metadata["source_url"] = build_github_source_url(
                repository,
                commit,
                relative_path,
                repository_root=root,
            )
        if topic_by_path and relative_path in topic_by_path:
            metadata["topic"] = topic_by_path[relative_path]
        documents.append(clean_document(CorpusDocument(path, text, metadata)))
    return documents
