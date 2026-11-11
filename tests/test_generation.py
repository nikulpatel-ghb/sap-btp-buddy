import json

from sap_btp_buddy.generation import build_evidence_passages, generate_grounded_answer, validate_payload
from sap_btp_buddy.models import Chunk, SearchResult
from sap_btp_buddy.sufficiency import apply_sufficiency_policy, assess_sufficiency


COMMIT = "a" * 40
REPOSITORY = "SAP-docs/btp-cloud-platform"


def _result(text="The btp CLI lists subaccounts."):
    path = "docs/example.md"
    return SearchResult(Chunk("E1", text, {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "path": path,
        "title": "Example",
        "source_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/{path}",
        "source_line_start": 1,
        "source_line_end": 3,
        "content_sha256": "b" * 64,
    }), 0.9)


class FakeGenerator:
    model_id = "fixture/model"

    class config:
        model_id = "fixture/model"
        base_url = "https://api.example.invalid/v1/"
        max_retries = 0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, question, evidence):
        self.calls.append((question, evidence))
        return self.outputs.pop(0)


class TimeoutGenerator(FakeGenerator):
    def generate(self, question, evidence):
        raise TimeoutError("fixture timeout")


def _payload(status="answered", evidence_ids=None):
    ids = evidence_ids or ["E1"]
    return json.dumps({
        "status": status,
        "answer": "The evidence says this.",
        "claims": [{"claim_id": "C1", "text": "The evidence says this.", "evidence_ids": ids}] if status == "answered" else [],
        "cli_commands": [],
        "limitations": [] if status == "answered" else ["The evidence is incomplete."],
        "missing_information": [],
        "evidence_ids": ids if status == "answered" else [],
    })


def test_valid_citation_renders_commit_pinned_source():
    generator = FakeGenerator([_payload()])
    result = generate_grounded_answer("What does it say?", [_result()], generator)
    assert result.status == "answered"
    assert result.validation["valid"] is True
    assert result.runtime["prompt_version"]
    assert "github.com/SAP-docs/btp-cloud-platform/blob/" in result.evidence[0]["source_url"]


def test_markdown_fenced_json_response_is_accepted():
    generator = FakeGenerator([f"```json\n{_payload()}\n```"])
    result = generate_grounded_answer("What does it say?", [_result()], generator)
    assert result.status == "answered"
    assert result.validation["valid"] is True
    assert len(generator.calls) == 1


def test_known_extra_top_level_evidence_ids_are_canonicalized():
    second = _result("A second passage.")
    second = SearchResult(Chunk("E2", second.chunk.text, second.chunk.metadata), 0.8)
    payload = json.loads(_payload())
    payload["evidence_ids"] = ["E1", "E2"]
    generator = FakeGenerator([json.dumps(payload)])
    result = generate_grounded_answer("What does it say?", [_result(), second], generator)
    assert result.status == "answered"
    assert result.validation["valid"] is True


def test_empty_retrieval_is_insufficient_evidence_without_provider_call():
    generator = FakeGenerator([])
    result = generate_grounded_answer("Unknown?", [], generator)
    assert result.status == "insufficient_evidence"
    assert generator.calls == []


def test_invalid_citation_repair_failure_is_explicit_error():
    generator = FakeGenerator([_payload(evidence_ids=["UNKNOWN"]), _payload(evidence_ids=["UNKNOWN"])])
    result = generate_grounded_answer("What does it say?", [_result()], generator)
    assert result.status == "error"
    assert result.runtime["repair_attempts"] == 1
    assert len(generator.calls) == 2


def test_insufficient_evidence_explanation_is_not_treated_as_answered_claims():
    payload = {
        "status": "insufficient_evidence",
        "answer": "The supplied passages do not establish the requested definition.",
        "claims": [{"claim_id": "C1", "text": "The supplied passages do not establish the requested definition.", "evidence_ids": ["E1"]}],
        "cli_commands": [],
        "limitations": ["The evidence is incomplete."],
        "missing_information": ["A definition of the requested term."],
        "evidence_ids": ["E1"],
    }
    generator = FakeGenerator([json.dumps(payload)])

    result = generate_grounded_answer("What is latency in BTP?", [_result()], generator)

    assert result.status == "insufficient_evidence"
    assert result.validation["valid"] is True
    assert result.claims == ()
    assert "requested definition" in result.limitations[-1]


def test_malformed_response_repair_failure_is_explicit_error():
    generator = FakeGenerator(["not-json", "still-not-json"])
    result = generate_grounded_answer("What does it say?", [_result()], generator)
    assert result.status == "error"
    assert "citation validation" in result.answer


def test_validate_payload_separates_citation_gate_from_claim_support():
    passages = build_evidence_passages([_result()])
    validation = validate_payload(json.loads(_payload()), passages)
    assert validation["valid"] is True
    assert validation["claim_support"] == "SEPARATE_EVALUATION_REQUIRED"


def _question(status="IN_CORPUS", labels=None, groups=None):
    return {
        "question": "What does the source say?",
        "support_status": status,
        "labels": labels or [],
        "evidence_groups": groups if groups is not None else [{"group_id": "G1", "source_section_ids": ["primary:docs/example.md"]}],
    }


def test_policy_answers_only_when_all_reviewed_groups_are_retrieved():
    decision = assess_sufficiency(_question(), [_result()], repository_by_label={"primary": REPOSITORY}, commit_by_repository={REPOSITORY: COMMIT})
    assert decision.status == "answered"
    assert decision.signals["numeric_confidence"] is None


def test_policy_distinguishes_missing_evidence_ambiguity_and_out_of_corpus():
    maps = {"primary": REPOSITORY}, {REPOSITORY: COMMIT}
    missing = assess_sufficiency(_question(groups=[{"group_id": "G1", "source_section_ids": ["primary:missing.md"]}]), [_result()], repository_by_label=maps[0], commit_by_repository=maps[1])
    ambiguous = assess_sufficiency(_question(labels=["ambiguous"]), [_result()], repository_by_label=maps[0], commit_by_repository=maps[1])
    unsupported = assess_sufficiency(_question(status="OUT_OF_CORPUS"), [_result()], repository_by_label=maps[0], commit_by_repository=maps[1])
    assert missing.status == "insufficient_evidence"
    assert ambiguous.status == "clarify"
    assert unsupported.status == "insufficient_evidence"


def test_policy_keeps_provider_timeout_as_technical_error():
    result = apply_sufficiency_policy(_question(), [_result()], generator=TimeoutGenerator([]), repository_by_label={"primary": REPOSITORY}, commit_by_repository={REPOSITORY: COMMIT})
    assert result.status == "error"
    assert "backend failed" in result.answer
