"""Conservative Markdown cleaning with raw-to-clean line provenance."""

import hashlib
import re
from typing import Any

from .models import CorpusDocument


_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*#*\s*$")
_LOIO_COMMENT_RE = re.compile(r"^\s*<!--\s*loio[0-9a-f]+\s*-->\s*$", re.IGNORECASE)
_SAP_ICONS_LINK_RE = re.compile(
    r'^\s*<link\s+rel="stylesheet"[^>]*sap-icons\.css[^>]*>\s*$',
    re.IGNORECASE,
)


def _title(lines: list[str]) -> str | None:
    for line in lines:
        match = _TITLE_RE.match(line.rstrip("\r\n"))
        if match:
            return match.group(1).strip()
    return None


def _numbered_normalized_lines(text: str) -> list[tuple[int, str]]:
    raw_lines = text.splitlines()
    return [(number, line + "\n") for number, line in enumerate(raw_lines, start=1)]


def clean_document(document: CorpusDocument) -> CorpusDocument:
    """Normalize line endings and remove only terminal blank lines.

    Markdown content, tables, warnings, links, lists, and fenced code are not
    rewritten. Every retained output line maps to its original physical line.
    """

    numbered_lines = _numbered_normalized_lines(document.text)
    removed: list[dict[str, Any]] = []
    retained_lines: list[tuple[int, str]] = []
    for line_number, line in numbered_lines:
        content = line.rstrip("\n")
        if _LOIO_COMMENT_RE.fullmatch(content):
            removed.append({"reason": "sap_loio_html_comment", "original_line": line_number})
        elif _SAP_ICONS_LINK_RE.fullmatch(content):
            removed.append({"reason": "sap_icons_stylesheet_link", "original_line": line_number})
        else:
            retained_lines.append((line_number, line))
    numbered_lines = retained_lines
    while numbered_lines and not numbered_lines[-1][1].rstrip("\n").strip():
        line_number, _ = numbered_lines.pop()
        removed.append({"reason": "terminal_blank_line", "original_line": line_number})

    cleaned_text = "".join(line for _, line in numbered_lines)
    line_map = [
        {
            "cleaned_line": index,
            "original_line_start": original_line,
            "original_line_end": original_line,
            "transformation": "line_ending_normalized",
        }
        for index, (original_line, _) in enumerate(numbered_lines, start=1)
    ]
    metadata = dict(document.metadata)
    metadata.setdefault("title", _title([line for _, line in numbered_lines]))
    metadata.setdefault("original_line_start", 1 if numbered_lines else None)
    metadata.setdefault("original_line_end", len(document.text.splitlines()))
    metadata["cleaned_line_count"] = len(numbered_lines)
    metadata["removed_line_count"] = len(removed)
    metadata["cleaning_exclusions"] = list(reversed(removed))
    metadata["line_provenance"] = line_map
    metadata["cleaned_content_sha256"] = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    metadata.setdefault("cleaning_policy_version", "v1-terminal-blank-lines-only")
    return CorpusDocument(document.source_path, cleaned_text, metadata)
