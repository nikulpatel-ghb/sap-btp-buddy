import json

from sap_btp_buddy.generation import generate_grounded_answer
from sap_btp_buddy.graph import build_answer_graph, graph_mermaid
from sap_btp_buddy.models import Chunk, SearchResult
from sap_btp_buddy.sufficiency import apply_sufficiency_policy


COMMIT = "a" * 40
REPOSITORY = "SAP-docs/btp-cloud-platform"
PATH = "docs/example.md"


def _result():
    return SearchResult(Chunk("E1", "The documented fact is here.", {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "path": PATH,
        "title": "Example",
        "source_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/{PATH}",
        "source_line_start": 1,
        "source_line_end": 2,
    }), 0.9)


def _question(status="IN_CORPUS", labels=None, groups=None):
    return {"question": "What is the documented fact?", "support_status": status, "labels": labels or [], "evidence_groups": groups if groups is not None else [{"group_id": "G1", "source_section_ids": ["primary:docs/example.md"]}]}


def _payload():
    return json.dumps({"status": "answered", "answer": "The documented fact is here.", "claims": [{"claim_id": "C1", "text": "The documented fact is here.", "evidence_ids": ["E1"]}], "cli_commands": [], "limitations": [], "missing_information": [], "evidence_ids": ["E1"]})


class FakeIndex:
    def __init__(self, results=None, failures=0):
        self.results, self.failures, self.calls = results or [_result()], failures, 0

    def retrieve(self, question, *, mode, top_k):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("fixture retrieval timeout")
        return self.results


class FakeGenerator:
    class config:
        model_id = "fixture/model"
        base_url = "https://api.example.invalid/v1/"
        max_retries = 0

    def __init__(self):
        self.calls = 0

    def generate(self, question, evidence):
        self.calls += 1
        return _payload()


def _graph(question, index=None, generator=None, retries=1):
    return build_answer_graph(index or FakeIndex(), generator or FakeGenerator(), repository_by_label={"primary": REPOSITORY}, commit_by_repository={REPOSITORY: COMMIT}, retrieval_mode="dense", max_retrieval_retries=retries)


def test_all_terminal_policy_branches_and_graph_render():
    answered = _graph(_question()).invoke({"question": _question()})
    clarify_q = _question(labels=["ambiguous"])
    clarify = _graph(clarify_q).invoke({"question": clarify_q})
    unsupported_q = _question(status="OUT_OF_CORPUS")
    unsupported = _graph(unsupported_q).invoke({"question": unsupported_q})
    error = _graph(_question(), FakeIndex(failures=2), retries=1).invoke({"question": _question()})
    assert answered["result"].status == "answered"
    assert clarify["result"].status == "clarify"
    assert unsupported["result"].status == "insufficient_evidence"
    assert error["result"].status == "error"
    assert "retrieve" in graph_mermaid(_graph(_question()))
    assert all(event["node"] in {"retrieve", "decide", "generate", "terminal"} for event in error["trace"])


def test_retrieval_retry_limit_is_bounded():
    index = FakeIndex(failures=10)
    result = _graph(_question(), index, retries=1).invoke({"question": _question()})
    assert index.calls == 2
    assert result["retrieval_attempts"] == 2
    assert result["result"].status == "error"


def test_graph_matches_standalone_backend_for_same_question_and_configuration():
    question = _question()
    graph_generator = FakeGenerator()
    standalone_generator = FakeGenerator()
    graph_result = _graph(question, FakeIndex(), graph_generator).invoke({"question": question})["result"]
    standalone_result = apply_sufficiency_policy(question, [_result()], generator=standalone_generator, repository_by_label={"primary": REPOSITORY}, commit_by_repository={REPOSITORY: COMMIT})
    assert graph_result.status == standalone_result.status
    assert graph_result.answer == standalone_result.answer
    assert graph_result.claims == standalone_result.claims
