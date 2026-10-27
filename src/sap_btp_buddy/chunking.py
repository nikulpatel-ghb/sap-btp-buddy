"""Deterministic token and Markdown-heading-aware chunking.

Heading-aware here means structural Markdown chunking. It is deliberately not
called semantic chunking: no embedding similarity or learned boundary model is
used to choose boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from .models import Chunk, CorpusDocument


class TokenizerLike(Protocol):
    """Minimal tokenizer surface required by the chunkers."""

    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[Any]: ...


class WhitespaceTokenizer:
    """Deterministic local diagnostic tokenizer until an embedding model is selected."""

    name_or_path = "whitespace-fallback"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return re.findall(r"\S+", text)


@dataclass(frozen=True)
class _Unit:
    text: str
    start_line: int
    end_line: int
    kind: str
    breadcrumb: tuple[str, ...]


@dataclass(frozen=True)
class _Section:
    breadcrumb: tuple[str, ...]
    units: tuple[_Unit, ...]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _encode(tokenizer: TokenizerLike, text: str) -> list[Any]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    return len(_encode(tokenizer, text))


def _tokenizer_name(tokenizer: TokenizerLike) -> str:
    return str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__))


def _title(document: CorpusDocument) -> str:
    title = document.metadata.get("title")
    if title:
        return str(title)
    for line in document.text.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return ""


def _sections(document: CorpusDocument) -> list[_Section]:
    title = _title(document)
    stack: list[str] = []
    sections: list[_Section] = []
    current: list[_Unit] = []
    current_breadcrumb: tuple[str, ...] = (title,) if title else ()
    lines = document.text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING_RE.match(line.rstrip("\n"))
        if heading:
            if current:
                sections.append(_Section(current_breadcrumb, tuple(current)))
                current = []
            level = len(heading.group(1))
            stack = stack[: level - 1]
            stack.append(heading.group(2).strip())
            current_breadcrumb = tuple(value for value in ((title,) + tuple(stack)) if value)
            current.append(_Unit(line, index + 1, index + 1, "heading", current_breadcrumb))
            index += 1
            continue
        fence = _FENCE_RE.match(line)
        if fence:
            start = index
            index += 1
            while index < len(lines):
                if _FENCE_RE.match(lines[index]) and lines[index].lstrip().startswith(fence.group(1)[0]):
                    index += 1
                    break
                index += 1
            current.append(_Unit("".join(lines[start:index]), start + 1, index, "code", current_breadcrumb))
            continue
        if index + 1 < len(lines) and "|" in line and _TABLE_SEPARATOR_RE.match(lines[index + 1]):
            start = index
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                index += 1
            current.append(_Unit("".join(lines[start:index]), start + 1, index, "table", current_breadcrumb))
            continue
        if line.lstrip().lower().startswith("<table"):
            start = index
            index += 1
            while index < len(lines):
                index += 1
                if "</table>" in lines[index - 1].lower():
                    break
            current.append(_Unit("".join(lines[start:index]), start + 1, index, "table", current_breadcrumb))
            continue
        current.append(_Unit(line, index + 1, index + 1, "text", current_breadcrumb))
        index += 1
    if current:
        sections.append(_Section(current_breadcrumb, tuple(current)))
    return sections or [_Section(current_breadcrumb, ())]


def _split_text(text: str, budget: int, tokenizer: TokenizerLike) -> list[str]:
    if _token_count(tokenizer, text) <= budget:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if _token_count(tokenizer, remaining) <= budget:
            parts.append(remaining)
            break
        low, high = 1, len(remaining)
        best = 1
        while low <= high:
            middle = (low + high) // 2
            if _token_count(tokenizer, remaining[:middle]) <= budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        cut = max(remaining.rfind("\n", 0, best + 1), remaining.rfind(" ", 0, best + 1))
        if cut <= 0:
            cut = best
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    return parts


def _split_unit(unit: _Unit, budget: int, tokenizer: TokenizerLike) -> list[_Unit]:
    if _token_count(tokenizer, unit.text) <= budget:
        return [unit]
    lines = unit.text.splitlines(keepends=True)
    result: list[_Unit] = []
    for line_index, line in enumerate(lines):
        pieces = _split_text(line, budget, tokenizer) if _token_count(tokenizer, line) > budget else [line]
        for piece in pieces:
            line_number = unit.start_line + line_index
            result.append(_Unit(piece, line_number, line_number, unit.kind, unit.breadcrumb))
    return result


def _tail_unit(unit: _Unit, token_count: int, tokenizer: TokenizerLike) -> _Unit | None:
    """Return a deterministic token-bounded tail for structural overlap."""

    if token_count <= 0:
        return None
    if _token_count(tokenizer, unit.text) <= token_count:
        return unit
    low, high = 0, len(unit.text)
    best = len(unit.text)
    while low <= high:
        middle = (low + high) // 2
        if _token_count(tokenizer, unit.text[middle:]) <= token_count:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    return _Unit(unit.text[best:], unit.start_line, unit.end_line, unit.kind, unit.breadcrumb)


def _context(title: str, breadcrumb: tuple[str, ...]) -> str:
    values = [value for value in breadcrumb if value]
    if len(values) > 1 and values[0] == values[1]:
        values.pop(1)
    path = " > ".join(values)
    values = [f"Title: {title}"] if title else []
    if path:
        values.append(f"Heading path: {path}")
    return "\n".join(values)


def _with_context(context: str, body: str) -> str:
    return f"{context}\n\n{body}" if context else body


def _chunk_id(document: CorpusDocument, strategy: str, config: dict[str, Any], start: int, end: int, ordinal: int, body: str) -> str:
    identity = {
        "repository": document.metadata.get("repository"),
        "commit": document.metadata.get("commit"),
        "path": document.metadata.get("path", document.source_path.as_posix()),
        "cleaned_content_sha256": document.metadata.get("cleaned_content_sha256"),
        "strategy": strategy,
        "config": config,
        "source_line_start": start,
        "source_line_end": end,
        "ordinal": ordinal,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _make_chunk(document: CorpusDocument, strategy: str, config: dict[str, Any], tokenizer: TokenizerLike, context: str, body: str, start: int, end: int, kind: str, ordinal: int, overflow: bool) -> Chunk:
    text = _with_context(context, body)
    breadcrumb = []
    if "Heading path:" in context:
        breadcrumb = context.split("Heading path:", 1)[1].strip().split(" > ")
    document_metadata = {
        key: value
        for key, value in document.metadata.items()
        if key not in {"line_provenance", "cleaning_exclusions"}
    }
    return Chunk(
        chunk_id=_chunk_id(document, strategy, config, start, end, ordinal, body),
        text=text,
        metadata={
            **document_metadata,
            "chunk_strategy": strategy,
            "chunker_version": "v1",
            "source_line_start": start,
            "source_line_end": end,
            "source_text": body,
            "heading_breadcrumb": breadcrumb,
            "block_kind": kind,
            "contains_code": kind == "code",
            "contains_table": kind == "table",
            "overflow": overflow,
            "token_count": _token_count(tokenizer, text),
            "body_token_count": _token_count(tokenizer, body),
            "embedding_text_has_context": bool(context),
            "max_input_tokens": config["max_input_tokens"],
            "requested_overlap_tokens": config["chunk_overlap"],
            "token_limit_status": config["token_limit_status"],
        },
    )


def _pack_units(document: CorpusDocument, units: list[_Unit], *, strategy: str, tokenizer: TokenizerLike, max_input_tokens: int, title: str, chunk_overlap: int, config: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: list[_Unit] = []
    carry: _Unit | None = None
    ordinal = 0

    def flush() -> None:
        nonlocal pending, ordinal
        if not pending:
            return
        pieces: list[tuple[_Unit, bool]] = []
        for unit in pending:
            context = _context(title, unit.breadcrumb)
            context_tokens = _token_count(tokenizer, context + "\n\n") if context else 0
            if context_tokens >= max_input_tokens:
                raise ValueError("title and heading context exceed the configured token cap")
            body_budget = max(1, max_input_tokens - context_tokens)
            unit_overflow = _token_count(tokenizer, unit.text) > body_budget
            pieces.extend((piece, unit_overflow) for piece in _split_unit(unit, body_budget, tokenizer))
        for piece, piece_overflow in pieces:
            ordinal += 1
            piece_context = _context(title, piece.breadcrumb)
            overflow = piece.kind in {"code", "table"} and piece_overflow
            chunks.append(_make_chunk(document, strategy, config, tokenizer, piece_context, piece.text, piece.start_line, piece.end_line, piece.kind, ordinal, overflow))
        pending = []

    for unit in units:
        candidate_units = ([carry] if carry else []) + pending + [unit]
        candidate = "".join(item.text for item in candidate_units)
        context = _context(title, unit.breadcrumb)
        if _token_count(tokenizer, _with_context(context, candidate)) <= max_input_tokens:
            pending = candidate_units
            carry = None
            continue
        previous = pending[-1] if pending else carry
        flush()
        carry = _tail_unit(previous, chunk_overlap, tokenizer) if previous else None
        pending = ([carry] if carry else []) + [unit]
    flush()
    return chunks


def _fixed_chunks(document: CorpusDocument, tokenizer: TokenizerLike, max_input_tokens: int, overlap: int, config: dict[str, Any]) -> list[Chunk]:
    units = [unit for section in _sections(document) for unit in section.units]
    return _pack_units(document, units, strategy="fixed_token", tokenizer=tokenizer, max_input_tokens=max_input_tokens, title=_title(document), chunk_overlap=overlap, config=config)


def _heading_chunks(document: CorpusDocument, tokenizer: TokenizerLike, max_input_tokens: int, overlap: int, config: dict[str, Any]) -> list[Chunk]:
    return [chunk for section in _sections(document) for chunk in _pack_units(document, list(section.units), strategy="markdown_heading_aware", tokenizer=tokenizer, max_input_tokens=max_input_tokens, title=_title(document), chunk_overlap=overlap, config=config)]


def chunk_documents(documents: Iterable[CorpusDocument], *, strategy: str = "markdown_heading_aware", chunk_size: int = 400, chunk_overlap: int = 60, tokenizer: TokenizerLike | None = None, input_limit: int | None = None) -> list[Chunk]:
    """Split documents with a measured tokenizer cap and stable provenance."""

    if strategy not in {"fixed_token", "markdown_heading_aware"}:
        raise ValueError("strategy must be 'fixed_token' or 'markdown_heading_aware'")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller")
    active_tokenizer = tokenizer or WhitespaceTokenizer()
    configured_limit = input_limit or chunk_size
    if configured_limit <= 0:
        raise ValueError("input_limit must be positive")
    max_input_tokens = min(chunk_size, configured_limit)
    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "input_limit": input_limit,
        "max_input_tokens": max_input_tokens,
        "tokenizer": _tokenizer_name(active_tokenizer),
        "token_limit_status": "MEASURED" if tokenizer is not None and input_limit is not None and not isinstance(active_tokenizer, WhitespaceTokenizer) else "PENDING_SELECTED_EMBEDDING_MODEL",
    }
    result: list[Chunk] = []
    for document in sorted(documents, key=lambda item: str(item.metadata.get("path", item.source_path))):
        if strategy == "fixed_token":
            result.extend(_fixed_chunks(document, active_tokenizer, max_input_tokens, chunk_overlap, config))
        else:
            result.extend(_heading_chunks(document, active_tokenizer, max_input_tokens, chunk_overlap, config))
    return result


def chunk_statistics(chunks: Iterable[Chunk], tokenizer: TokenizerLike | None = None) -> dict[str, Any]:
    """Return count, token quantiles, duplicate, and block statistics."""

    active_tokenizer = tokenizer or WhitespaceTokenizer()
    rows = list(chunks)
    counts = sorted(_token_count(active_tokenizer, chunk.text) for chunk in rows)
    if not counts:
        return {"count": 0, "token_min": None, "token_median": None, "token_p95": None, "token_max": None, "duplicate_rate": 0.0, "code_chunks": 0, "table_chunks": 0, "overflow_chunks": 0}

    def percentile(values: list[int], fraction: float) -> int:
        index = (len(values) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        if lower == upper:
            return values[lower]
        return round(values[lower] + (values[upper] - values[lower]) * (index - lower))

    fingerprints = [hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() for chunk in rows]
    return {
        "count": len(rows),
        "token_min": counts[0],
        "token_median": percentile(counts, 0.5),
        "token_p95": percentile(counts, 0.95),
        "token_max": counts[-1],
        "duplicate_rate": 1.0 - (len(set(fingerprints)) / len(fingerprints)),
        "code_chunks": sum(chunk.metadata.get("contains_code", False) for chunk in rows),
        "table_chunks": sum(chunk.metadata.get("contains_table", False) for chunk in rows),
        "overflow_chunks": sum(chunk.metadata.get("overflow", False) for chunk in rows),
    }


def serialize_chunk(chunk: Chunk) -> dict[str, Any]:
    """Serialize a chunk without dropping provenance metadata."""

    return {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
