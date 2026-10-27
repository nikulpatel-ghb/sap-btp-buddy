"""Controlled development-set retrieval experiments."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from .baseline import _document_id, _group_hit, _gold_document_id
from .indexing import PersistentIndex, source_document_id, tokenize_bm25


ARMS = {
    "A": {"chunker": "fixed_token", "mode": "dense", "rerank": False},
    "B": {"chunker": "markdown_heading_aware", "mode": "dense", "rerank": False},
    "C": {"chunker": "markdown_heading_aware", "mode": "hybrid", "rerank": False},
}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def _exact_term_miss(question: dict[str, Any], results: list[Any]) -> bool | None:
    """Measure missing technical query terms for exact-CLI dev questions.

    This is a retrieval diagnostic, not an answer-quality judgment. It is
    reported only for reviewed, in-corpus ``exact_cli`` questions.
    """

    if "exact_cli" not in question.get("labels", []) or question.get("support_status") != "IN_CORPUS":
        return None
    technical = [token for token in tokenize_bm25(question["question"]) if any(mark in token for mark in ("-", "/", ":", "=", ".")) or token in {"btp", "cf"}]
    if not technical:
        return None
    retrieved_text = "\n".join(result.chunk.text.casefold() for result in results)
    return not all(token in retrieved_text for token in technical)


def _evaluate_row(question: dict[str, Any], results: list[Any], latency_ms: float, repository_by_label: Mapping[str, str], commit_by_repository: Mapping[str, str]) -> dict[str, Any]:
    required_groups = question.get("evidence_groups", [])
    answerable = question.get("support_status") == "IN_CORPUS" and bool(required_groups)
    ranked_chunks = []
    ranked_document_ids = []
    for rank, result in enumerate(results, 1):
        document_id = source_document_id(result.chunk)
        ranked_chunks.append({"rank": rank, "chunk_id": result.chunk.chunk_id, "document_id": document_id, "score": result.score})
        if document_id not in ranked_document_ids:
            ranked_document_ids.append(document_id)
    gold_documents = sorted({_gold_document_id(section_id, dict(repository_by_label), dict(commit_by_repository)) for group in required_groups for section_id in group.get("source_section_ids", [])})
    hit_documents = sorted(set(gold_documents).intersection(ranked_document_ids))
    hit_groups = [group["group_id"] for group in required_groups if any(_group_hit(result.chunk.metadata, group.get("source_section_ids", []), dict(repository_by_label), dict(commit_by_repository)) for result in results)]
    row: dict[str, Any] = {
        "id": question["id"],
        "question": question["question"],
        "labels": question.get("labels", []),
        "support_status": question.get("support_status"),
        "ranked_chunks": ranked_chunks,
        "ranked_document_ids": ranked_document_ids,
        "latency_ms": round(latency_ms, 3),
        "gold_document_ids": gold_documents,
        "hit_document_ids": hit_documents,
        "required_group_ids": [group["group_id"] for group in required_groups],
        "hit_group_ids": hit_groups,
        "answerable_for_recall": answerable,
        "exact_term_miss": _exact_term_miss(question, results),
    }
    if answerable:
        row["document_recall_at_5"] = len(hit_documents) / len(gold_documents) if gold_documents else None
        row["query_hit_at_5"] = bool(hit_documents)
        row["evidence_coverage"] = len(hit_groups) / len(required_groups)
        row["full_evidence_coverage"] = len(hit_groups) == len(required_groups)
    else:
        row.update({"document_recall_at_5": None, "query_hit_at_5": None, "evidence_coverage": None, "full_evidence_coverage": None})
    return row


def run_arm(index: PersistentIndex, questions_path: Path, output_path: Path, repository_by_label: Mapping[str, str], commit_by_repository: Mapping[str, str], *, arm: str, top_k: int = 5) -> dict[str, Any]:
    """Run one preregistered arm and write raw per-query evidence."""

    if arm not in ARMS:
        raise ValueError(f"unknown experiment arm: {arm}")
    if ARMS[arm]["rerank"]:
        raise NotImplementedError("Reranking is NOT IMPLEMENTED; no reranker is configured")
    rows = []
    for line in questions_path.read_text(encoding="utf-8-sig").splitlines():
        question = json.loads(line)
        started = time.perf_counter()
        results = index.retrieve(question["question"], mode=ARMS[arm]["mode"], top_k=top_k)
        rows.append(_evaluate_row(question, results, (time.perf_counter() - started) * 1000, repository_by_label, commit_by_repository))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    answerable = [row for row in rows if row["answerable_for_recall"]]
    exact = [row["exact_term_miss"] for row in answerable if row["exact_term_miss"] is not None]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "arm": arm,
        "settings": ARMS[arm],
        "top_k": top_k,
        "rows": len(rows),
        "answerable_rows": len(answerable),
        "document_recall_at_5": round(statistics.mean(row["document_recall_at_5"] for row in answerable), 6) if answerable else None,
        "query_hit_rate_at_5": round(statistics.mean(row["query_hit_at_5"] for row in answerable), 6) if answerable else None,
        "evidence_coverage": round(statistics.mean(row["evidence_coverage"] for row in answerable), 6) if answerable else None,
        "full_evidence_coverage": round(statistics.mean(row["full_evidence_coverage"] for row in answerable), 6) if answerable else None,
        "exact_term_miss_rate": round(statistics.mean(exact), 6) if exact else None,
        "latency_ms": {"median": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95), "max": max(latencies) if latencies else None},
        "raw_results": str(output_path),
    }


def compare_experiment(summary_rows: list[dict[str, Any]], *, latency_p95_budget_ms: float = 2000.0) -> dict[str, Any]:
    """Choose only from measured comparable rows; otherwise remain unmeasured."""

    measured = [row for row in summary_rows if row.get("document_recall_at_5") is not None and row.get("full_evidence_coverage") is not None]
    if not measured:
        return {"status": "NOT MEASURED", "selected_arm": None, "reason": "No comparable measured arm results."}
    eligible = [row for row in measured if (row.get("latency_ms") or {}).get("p95") is not None and row["latency_ms"]["p95"] <= latency_p95_budget_ms]
    pool = eligible or measured
    selected = sorted(pool, key=lambda row: (-row["full_evidence_coverage"], -row["document_recall_at_5"], (row["latency_ms"]["p95"] or float("inf")), row["arm"]))[0]
    return {"status": "MEASURED", "selected_arm": selected["arm"], "latency_budget_ms": latency_p95_budget_ms, "budget_met": bool(eligible), "reason": "Higher full evidence coverage, then document recall, then latency; ties resolve to the simpler/earlier arm."}
