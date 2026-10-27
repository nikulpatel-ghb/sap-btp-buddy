"""Bounded LangGraph orchestration for SAP BTP Buddy answers."""

from __future__ import annotations

import time
from typing import Any, Mapping, TypedDict

from langgraph.graph import END, START, StateGraph

from .generation import GroundedResult, _runtime, generate_grounded_answer
from .sufficiency import SufficiencyDecision, assess_sufficiency


class AnswerGraphState(TypedDict, total=False):
    question: dict[str, Any]
    results: list[Any]
    decision: SufficiencyDecision
    result: GroundedResult
    trace: list[dict[str, Any]]
    backend_error: str | None
    retrieval_attempts: int


def _event(state: AnswerGraphState, node: str, started: float, **details: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), **details}]


def build_answer_graph(
    index: Any,
    generator: Any,
    *,
    repository_by_label: Mapping[str, str],
    commit_by_repository: Mapping[str, str],
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    max_retrieval_retries: int = 1,
) -> Any:
    """Build a compiled graph with bounded retrieval retries and terminal branches."""

    if top_k != 5:
        raise ValueError("the answer graph requires the fixed top_k=5 policy")
    if max_retrieval_retries < 0 or max_retrieval_retries > 2:
        raise ValueError("max_retrieval_retries must be between 0 and 2")

    def retrieve_node(state: AnswerGraphState) -> AnswerGraphState:
        started = time.perf_counter()
        attempts = 0
        last_error: Exception | None = None
        while attempts <= max_retrieval_retries:
            attempts += 1
            try:
                results = list(index.retrieve(state["question"]["question"], mode=retrieval_mode, top_k=top_k))
                return {"results": results, "retrieval_attempts": attempts, "backend_error": None, "trace": _event(state, "retrieve", started, attempts=attempts, result_count=len(results), retry_limit=max_retrieval_retries)}
            except Exception as exc:
                last_error = exc
        return {"results": [], "retrieval_attempts": attempts, "backend_error": type(last_error).__name__ if last_error else "UnknownError", "trace": _event(state, "retrieve", started, attempts=attempts, result_count=0, retry_limit=max_retrieval_retries, outcome="error")}

    def decide_node(state: AnswerGraphState) -> AnswerGraphState:
        started = time.perf_counter()
        decision = assess_sufficiency(state["question"], state.get("results", []), repository_by_label=repository_by_label, commit_by_repository=commit_by_repository, backend_error=state.get("backend_error"))
        return {"decision": decision, "trace": _event(state, "decide", started, status=decision.status, missing_group_ids=list(decision.missing_group_ids), signals=decision.signals)}

    def generate_node(state: AnswerGraphState) -> AnswerGraphState:
        started = time.perf_counter()
        result = generate_grounded_answer(state["question"]["question"], state.get("results", []), generator)
        return {"result": result, "trace": _event(state, "generate", started, status=result.status, validation_valid=result.validation.get("valid"))}

    def terminal_node(state: AnswerGraphState) -> AnswerGraphState:
        started = time.perf_counter()
        decision = state["decision"]
        result = GroundedResult(decision.status, decision.message, (), (), (), decision.missing_group_ids, (), {"valid": True, "errors": [], "policy": decision.signals}, _runtime(generator))
        return {"result": result, "trace": _event(state, "terminal", started, status=decision.status, branch=decision.status)}

    def generate_or_terminal(state: AnswerGraphState) -> str:
        return "generate" if state["decision"].status == "answered" else "terminal"

    builder = StateGraph(AnswerGraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("decide", decide_node)
    builder.add_node("generate", generate_node)
    builder.add_node("terminal", terminal_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "decide")
    builder.add_conditional_edges("decide", generate_or_terminal, {"generate": "generate", "terminal": "terminal"})
    builder.add_edge("generate", END)
    builder.add_edge("terminal", END)
    return builder.compile()


def graph_mermaid(graph: Any) -> str:
    """Return the rendered graph definition for notebook/report embedding."""

    return graph.get_graph().draw_mermaid()
