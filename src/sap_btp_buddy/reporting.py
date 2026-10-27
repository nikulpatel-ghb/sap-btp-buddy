"""Report validation and serialization used by the notebook and CLI."""

import json
import re
from pathlib import Path
from typing import Any

from .models import EvaluationReport


def export_report(report: EvaluationReport, output_path: Path) -> Path:
    """Export a report artifact for notebook display and Google Doc transfer."""

    raise NotImplementedError(
        "Report export is the next build milestone; keep this public API stable."
    )


_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def validate_report(report_path: Path) -> dict[str, Any]:
    """Validate a Markdown or Jupyter report without executing it.

    The validator is deliberately strict for submission use: broken local links,
    missing report files, and unresolved ``PENDING`` markers are reported as
    issues. External URLs are syntax-checked but not fetched.
    """

    report_path = report_path.resolve()
    result: dict[str, Any] = {
        "report": str(report_path),
        "format": report_path.suffix.lower().lstrip("."),
        "missing_files": [],
        "broken_links": [],
        "pending_fields": [],
        "external_links": [],
        "status": "FAIL",
    }
    if not report_path.is_file():
        result["missing_files"].append(str(report_path))
        return result

    try:
        if report_path.suffix.lower() == ".ipynb":
            notebook = json.loads(report_path.read_text(encoding="utf-8"))
            sources = [
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") in {"markdown", "code"}
            ]
        else:
            sources = [report_path.read_text(encoding="utf-8")]
    except (OSError, json.JSONDecodeError) as exc:
        result["missing_files"].append(f"{report_path}: unreadable report ({exc})")
        return result

    for source_index, source in enumerate(sources, start=1):
        for line_number, line in enumerate(source.splitlines(), start=1):
            if "PENDING" in line:
                result["pending_fields"].append(
                    {"source": source_index, "line": line_number, "text": line.strip()}
                )
        for target in _MARKDOWN_LINK.findall(source):
            target = target.strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith("#"):
                continue
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
                result["external_links"].append(target)
                continue
            local_target = (report_path.parent / target.split("#", 1)[0]).resolve()
            if not local_target.is_file() and not local_target.is_dir():
                result["broken_links"].append(
                    {"source": source_index, "target": target, "resolved": str(local_target)}
                )

    result["status"] = "PASS" if not (
        result["missing_files"] or result["broken_links"] or result["pending_fields"]
    ) else "FAIL"
    result["summary"] = {
        "missing_files": len(result["missing_files"]),
        "broken_links": len(result["broken_links"]),
        "pending_fields": len(result["pending_fields"]),
        "external_links": len(result["external_links"]),
    }
    return result
