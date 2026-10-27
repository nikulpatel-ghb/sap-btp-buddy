"""Retrieval-only development baseline evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .indexing import PersistentIndex


def _document_id(repository: str | None, commit: str | None, path: str | None) -> str:
    return f"{repository}@{commit}:{path}"


def _gold_document_id(section_id: str, repository_by_label: dict[str, str], commit_by_repository: dict[str, str]) -> str:
    label, remainder = section_id.split(":", 1)
    path = remainder.split("#", 1)[0]
    repository = repository_by_label[label]
    return _document_id(repository, commit_by_repository[repository], path)


def _group_hit(chunk: dict[str, Any], section_ids: list[str], repository_by_label: dict[str, str], commit_by_repository: dict[str, str]) -> bool:
    chunk_document = _document_id(chunk["repository"], chunk["commit"], chunk["path"])
    breadcrumb = " > ".join(chunk.get("heading_breadcrumb", []))
    for section_id in section_ids:
        label, remainder = section_id.split(":", 1)
        path, _, heading = remainder.partition("#")
        expected = _document_id(repository_by_label[label], commit_by_repository[repository_by_label[label]], path)
        if chunk_document == expected and (not heading or heading.casefold() in breadcrumb.casefold()):
            return True
    return False


def run_dense_dev_baseline(index: PersistentIndex, questions_path: Path, output_path: Path, repository_by_label: dict[str, str], commit_by_repository: dict[str, str]) -> dict[str, Any]:
    """Run dense top-5 retrieval on development questions only."""

    rows: list[dict[str, Any]] = []
    for line in questions_path.read_text(encoding="utf-8-sig").splitlines():
        question = json.loads(line)
        status = question.get("support_status")
        required_groups = question.get("evidence_groups", [])
        answerable = status == "IN_CORPUS" and bool(required_groups)
        started = time.perf_counter()
        retrieved = index.retrieve(question["question"], mode="dense", top_k=5)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        ranked_chunks: list[dict[str, Any]] = []
        ranked_document_ids: list[str] = []
        for rank, result in enumerate(retrieved, 1):
            metadata = result.chunk.metadata
            document_id = _document_id(metadata.get("repository"), metadata.get("commit"), metadata.get("path"))
            ranked_chunks.append({"rank": rank, "chunk_id": result.chunk.chunk_id, "document_id": document_id, "score": result.score})
            if document_id not in ranked_document_ids:
                ranked_document_ids.append(document_id)
        gold_documents = sorted({_gold_document_id(section_id, repository_by_label, commit_by_repository) for group in required_groups for section_id in group.get("source_section_ids", [])})
        hit_documents = sorted(set(gold_documents).intersection(ranked_document_ids))
        hit_groups = [group["group_id"] for group in required_groups if any(_group_hit(result.chunk.metadata, group.get("source_section_ids", []), repository_by_label, commit_by_repository) for result in retrieved)]
        row = {
            "id": question["id"],
            "split": "dev",
            "question": question["question"],
            "support_status": status,
            "answerable_for_recall": answerable,
            "ranked_chunks": ranked_chunks,
            "ranked_document_ids": ranked_document_ids,
            "latency_ms": latency_ms,
            "gold_document_ids": gold_documents,
            "hit_document_ids": hit_documents,
            "required_group_ids": [group["group_id"] for group in required_groups],
            "hit_group_ids": hit_groups,
            "excluded_reason": None if answerable else "unsupported_or_unanswerable",
        }
        if answerable:
            row["document_recall_at_5"] = len(hit_documents) / len(gold_documents) if gold_documents else None
            row["query_hit_at_5"] = bool(hit_documents)
            row["group_coverage"] = len(hit_groups) / len(required_groups)
            row["full_evidence_coverage"] = len(hit_groups) == len(required_groups)
        else:
            row["document_recall_at_5"] = None
            row["query_hit_at_5"] = None
            row["group_coverage"] = None
            row["full_evidence_coverage"] = None
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    answerable_rows = [row for row in rows if row["answerable_for_recall"]]
    return {
        "split": "dev",
        "retrieval": "dense",
        "top_k": 5,
        "rows": len(rows),
        "answerable_rows": len(answerable_rows),
        "excluded_rows": len(rows) - len(answerable_rows),
        "document_recall_at_5": sum(row["document_recall_at_5"] for row in answerable_rows) / len(answerable_rows) if answerable_rows else None,
        "query_hit_rate_at_5": sum(row["query_hit_at_5"] for row in answerable_rows) / len(answerable_rows) if answerable_rows else None,
        "full_evidence_coverage": sum(row["full_evidence_coverage"] for row in answerable_rows) / len(answerable_rows) if answerable_rows else None,
        "evidence_path": str(output_path),
    }
