"""Corpus discovery and commit-pinned source URL boundaries."""

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from .models import CorpusDocument


APPROVED_REPOSITORIES = frozenset(
    {
        "SAP-docs/btp-cloud-platform",
        "SAP-docs/btp-best-practices-guide",
    }
)
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def normalized_relative_path(path: Path, root: Path) -> str:
    """Return a validated, POSIX-style path relative to ``root``."""

    relative = path.resolve().relative_to(root.resolve())
    value = relative.as_posix()
    if value.startswith("../") or value == "..":
        raise ValueError(f"Path escapes repository root: {path}")
    return value


def build_github_source_url(
    repository: str,
    commit: str,
    relative_path: str,
    *,
    repository_root: Path | None = None,
) -> str:
    """Build a GitHub URL only after validating repository, commit, and path."""

    if repository not in APPROVED_REPOSITORIES:
        raise ValueError(f"Repository is not approved: {repository}")
    if not _FULL_COMMIT_RE.fullmatch(commit):
        raise ValueError("commit must be a full 40-character lowercase SHA")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Relative path is unsafe: {relative_path}")
    normalized = candidate.as_posix()
    if not normalized.endswith(".md"):
        raise ValueError(f"Only Markdown source paths are allowed: {relative_path}")
    if repository_root is not None and not (repository_root / candidate).is_file():
        raise FileNotFoundError(f"Validated source path does not exist: {relative_path}")
    encoded_path = quote(normalized, safe="/._-")
    return f"https://github.com/{repository}/blob/{commit}/{encoded_path}"


def discover_corpus(
    root: Path,
    patterns: Iterable[str] = ("**/*.md",),
) -> list[Path]:
    """Return deterministic Markdown paths under ``root`` matching ``patterns``."""

    paths = {path for pattern in patterns for path in root.glob(pattern)}
    return sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: normalized_relative_path(path, root),
    )


def refresh_allowlist(root: Path, allowlisted_paths: Iterable[str]) -> dict[str, list[str]]:
    """Reconcile an allowlist against current Markdown files without mutating it."""

    current = {normalized_relative_path(path, root) for path in discover_corpus(root)}
    requested = set(allowlisted_paths)
    return {
        "included": sorted(requested & current),
        "stale_deleted": sorted(requested - current),
        "new_unallowlisted": sorted(current - requested),
    }


def build_corpus_manifest(paths: Iterable[Path]) -> dict[str, object]:
    """Describe selected source paths; hashing is an ingestion concern."""

    ordered = sorted(str(path) for path in paths)
    return {"paths": ordered, "count": len(ordered), "status": "pending"}
