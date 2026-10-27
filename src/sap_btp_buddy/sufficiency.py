"""Evidence-sufficiency routing for development-calibrated question records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .generation import GroundedResult, _runtime
from .models import SearchResult


@dataclass(frozen=True)
class SufficiencyDecision:
    status: str
    message: str
    signals: dict[str, Any]
    missing_group_ids: tuple[str, ...] = ()


def assess_sufficiency(
    question: Mapping[str, Any],
    results: Sequence[SearchResult],
    *,
    repository_by_label: Mapping[str, str],
    commit_by_repository: Mapping[str, str],
    backend_error: str | None = None,
) -> SufficiencyDecision:
    """Route a reviewed development question without emitting confidence percentages."""

    if backend_error:
        return SufficiencyDecision("error", "The retrieval backend failed; no answer was produced.", {"backend_error": backend_error, "numeric_confidence": None})
    status = question.get("support_status")
    labels = set(question.get("labels", []))
    groups = list(question.get("evidence_groups", []))
    if status == "OUT_OF_CORPUS" or status == "OUT_OF_CORPUS_POLICY_FIXTURE":
        return SufficiencyDecision("insufficient_evidence", "The approved corpus does not establish that fact; I cannot verify it from the selected sources.", {"support_status": status, "retrieved_count": len(results), "numeric_confidence": None})
    if "ambiguous" in labels or "missing_information" in labels:
        return SufficiencyDecision("clarify", "Which account level, target, region, and identity-provider scope should this answer apply to?", {"support_status": status, "labels": sorted(labels), "retrieved_count": len(results), "numeric_confidence": None})
    if not results:
        return SufficiencyDecision("insufficient_evidence", "The approved retrieved evidence does not establish an answer.", {"support_status": status, "retrieved_count": 0, "numeric_confidence": None})
    if not groups:
        if status == "RUNTIME":
            return SufficiencyDecision("answered", "Retrieved evidence is available; answer only supported claims.", {"support_status": status, "retrieved_count": len(results), "top_k_policy": 5, "required_group_coverage": None, "numeric_confidence": None})
        return SufficiencyDecision("insufficient_evidence", "No reviewed evidence group is available for this request.", {"support_status": status, "retrieved_count": len(results), "numeric_confidence": None})

    from .baseline import _group_hit

    hit_group_ids = tuple(group["group_id"] for group in groups if any(_group_hit(result.chunk.metadata, group.get("source_section_ids", []), dict(repository_by_label), dict(commit_by_repository)) for result in results))
    missing = tuple(group["group_id"] for group in groups if group["group_id"] not in hit_group_ids)
    signals = {
        "support_status": status,
        "retrieved_count": len(results),
        "top_k_policy": 5,
        "required_group_count": len(groups),
        "hit_group_count": len(hit_group_ids),
        "required_group_coverage": len(hit_group_ids) / len(groups),
        "full_group_coverage_required": True,
        "numeric_confidence": None,
    }
    if missing:
        return SufficiencyDecision("insufficient_evidence", "The retrieved evidence supports only part of the requested scope; the missing evidence group cannot be stated safely.", signals, missing)
    return SufficiencyDecision("answered", "All reviewed evidence groups were retrieved; answer only supported claims.", signals)


def apply_sufficiency_policy(
    question: Mapping[str, Any],
    results: Sequence[SearchResult],
    *,
    generator: Any,
    repository_by_label: Mapping[str, str],
    commit_by_repository: Mapping[str, str],
    backend_error: str | None = None,
) -> GroundedResult:
    """Apply policy first, then invoke grounded generation only when answerable."""

    decision = assess_sufficiency(question, results, repository_by_label=repository_by_label, commit_by_repository=commit_by_repository, backend_error=backend_error)
    if decision.status != "answered":
        return GroundedResult(decision.status, decision.message, (), (), (), decision.missing_group_ids, (), {"valid": True, "errors": [], "policy": decision.signals}, _runtime(generator))
    from .generation import generate_grounded_answer

    result = generate_grounded_answer(str(question.get("question", "")), list(results), generator)
    return GroundedResult(result.status, result.answer, result.claims, result.cli_commands, result.limitations, result.missing_information, result.evidence, {**result.validation, "policy": decision.signals}, result.runtime)
