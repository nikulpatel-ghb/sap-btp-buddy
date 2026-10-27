"""Small data contracts shared by the library, notebook, CLI, and UI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class CorpusDocument:
    """A source document with stable provenance."""

    source_path: Path
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable chunk linked to its source document."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its retrieval score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """A generated answer and the evidence used to support it."""

    question: str
    text: str
    citations: tuple[SearchResult, ...] = ()
    status: str = "pending"


@dataclass(frozen=True)
class EvaluationReport:
    """Per-question evaluation output plus aggregate metrics."""

    rows: tuple[dict[str, Any], ...]
    metrics: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)


AnswerFn = Callable[[str], Answer]
QuestionSet = Sequence[str]
