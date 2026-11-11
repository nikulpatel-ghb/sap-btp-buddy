"""Conversational Streamlit adapter over the shared answer graph."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from sap_btp_buddy.config import PROJECT_ROOT, live_generation_allowed
from sap_btp_buddy.generation_adapter import build_generation_adapter
from sap_btp_buddy.graph import build_answer_graph
from sap_btp_buddy.indexing import load_index


load_dotenv(PROJECT_ROOT / ".env")
EXAMPLES = (
    "What are the main levels in the SAP BTP account model?",
    "What exact btp CLI command is documented for listing subaccounts?",
    "Does the approved corpus establish whether a particular tenant API is available?",
)


@dataclass(frozen=True)
class Runtime:
    graph: Any | None
    startup_error: str | None
    snapshot_date: str
    index_path: str
    retrieval_mode: str


def _resolve_index_path(index_path: str, retrieval_mode: str = "hybrid") -> Path:
    """Resolve a concrete index or the active identity for the requested mode."""

    path = Path(index_path)
    if (path / "manifest.json").is_file() and (path / "build.complete").is_file():
        return path

    candidates = sorted(
        child
        for child in path.iterdir()
        if child.is_dir()
        and (child / "manifest.json").is_file()
        and (child / "build.complete").is_file()
    ) if path.is_dir() else []
    compatible = []
    for candidate in candidates:
        try:
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("retrieval") == retrieval_mode:
            compatible.append(candidate)
    if len(compatible) == 1:
        return compatible[0]
    if not candidates:
        raise ValueError(f"no complete index found under {path}")
    if not compatible:
        available = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(f"no complete {retrieval_mode} index found under {path}; available identities: {available}")
    raise ValueError(f"multiple complete {retrieval_mode} indexes found under {path}; set BTP_INDEX_PATH to one identity directory")


def _snapshot_date() -> str:
    manifest_path = PROJECT_ROOT / "artifacts" / "corpus_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        return str(manifest.get("snapshot_timestamp_utc", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def _approved_maps() -> tuple[dict[str, str], dict[str, str]]:
    manifest = json.loads((PROJECT_ROOT / "artifacts" / "corpus_manifest.json").read_text(encoding="utf-8-sig"))
    repositories = manifest["repositories"]
    by_label = {"primary": repositories[0]["repository"], "secondary": repositories[1]["repository"]}
    commits = {item["repository"]: item["commit"] for item in repositories}
    return by_label, commits


@st.cache_resource(show_spinner=False)
def load_runtime(index_path: str, retrieval_mode: str) -> Runtime:
    """Load only immutable validated resources; never cache secrets or answers."""

    try:
        if not live_generation_allowed(PROJECT_ROOT / "config.yaml"):
            raise RuntimeError("live generation is disabled by config.yaml")
        resolved_index_path = _resolve_index_path(index_path, retrieval_mode)
        index = load_index(resolved_index_path)
        generator = build_generation_adapter()
        by_label, commits = _approved_maps()
        graph = build_answer_graph(index, generator, repository_by_label=by_label, commit_by_repository=commits, retrieval_mode=retrieval_mode, max_retrieval_retries=1)
        return Runtime(graph, None, _snapshot_date(), str(resolved_index_path), retrieval_mode)
    except Exception as exc:  # startup must be visible and sanitized
        return Runtime(None, f"Startup is unavailable: {type(exc).__name__}. Configure a compatible index and generation provider.", _snapshot_date(), index_path, retrieval_mode)


def _render_evidence(result: Any) -> None:
    evidence = getattr(result, "evidence", ())
    if not evidence:
        return
    with st.expander(f"Evidence ({len(evidence)} retrieved passage(s))", expanded=True):
        for item in evidence:
            title = item.get("title") or item.get("path") or "Untitled source"
            section = item.get("heading_breadcrumb") or item.get("section") or "Section not recorded"
            st.markdown(f"**{title}**  \nSection: `{section}`")
            st.caption(f"Lines {item.get('source_line_start', '?')}–{item.get('source_line_end', '?')} · {item.get('commit', 'commit unavailable')} · {item.get('source_url', 'source URL unavailable')}")
            excerpt = str(item.get("text", ""))
            st.code(excerpt[:2000], language="markdown")
            if item.get("source_url"):
                st.link_button("Open pinned source", item["source_url"])


def _render_result(result: Any, elapsed_ms: float) -> None:
    status = getattr(result, "status", "error")
    labels = {"answered": "Answer", "clarify": "Clarification needed", "insufficient_evidence": "Insufficient evidence", "error": "Operational error"}
    st.markdown(f"**{labels.get(status, 'Result')}**")
    if status == "error":
        st.error(result.answer)
    elif status == "insufficient_evidence":
        st.warning(result.answer)
    elif status == "clarify":
        st.info(result.answer)
    else:
        st.markdown(result.answer)
    st.caption(f"Status: `{status}` · latency: `{elapsed_ms:.1f} ms`")
    _render_evidence(result)


def _render_message(message: dict[str, Any]) -> None:
    """Render one persisted conversation turn using native Streamlit chat bubbles."""

    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
        return

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        result = message.get("result")
        if result is None:
            st.error(message.get("content", "The assistant could not produce a result."))
            return
        _render_result(result, float(message.get("elapsed_ms", 0.0)))


def main() -> None:
    st.set_page_config(page_title="SAP BTP Buddy", page_icon="📘", layout="wide")
    st.title("SAP BTP Buddy")
    st.caption("Documentation-grounded conversation over the approved SAP-docs snapshot. Each question is answered independently.")
    st.session_state.setdefault("messages", [])
    index_path = os.getenv("BTP_INDEX_PATH", str(PROJECT_ROOT / "indexes" / "active"))
    retrieval_mode = os.getenv("BTP_RETRIEVAL_MODE", "hybrid")
    runtime = load_runtime(index_path, retrieval_mode)

    with st.sidebar:
        st.subheader("Runtime")
        st.caption(f"Snapshot: `{runtime.snapshot_date}`")
        st.caption(f"Index: `{Path(runtime.index_path).name}`")
        st.caption(f"Retrieval: `{runtime.retrieval_mode}` · top-k `5`")
        if st.button("Reset conversation", width="stretch"):
            for key in ("pending_question", "messages"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.markdown("**Example questions**")
        for index, example in enumerate(EXAMPLES):
            if st.button(example, key=f"example_{index}", width="stretch"):
                st.session_state["pending_question"] = example
                st.rerun()

    if runtime.startup_error:
        st.error(runtime.startup_error)
        st.info("The UI is ready, but no answer will be produced until the validated index and provider configuration are available.")

    for message in st.session_state.messages:
        _render_message(message)

    question = st.chat_input("Ask one SAP BTP documentation question", submit_mode="disable")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        _render_message(user_message)
        if runtime.graph is None:
            fallback = {"role": "assistant", "content": "Startup fallback: the shared graph is unavailable, so no answer was attempted."}
            st.session_state.messages.append(fallback)
            _render_message(fallback)
            return
        record = {"question": question, "support_status": "RUNTIME", "labels": [], "evidence_groups": []}
        started = time.perf_counter()
        with st.spinner("Retrieving approved evidence and validating the answer…"):
            try:
                state = runtime.graph.invoke({"question": record})
                result = state["result"]
            except Exception:
                fallback = {"role": "assistant", "content": "The shared answer graph failed before producing a validated result. Please retry later."}
                st.session_state.messages.append(fallback)
                _render_message(fallback)
                return
        assistant_message = {
            "role": "assistant",
            "result": result,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
        st.session_state.messages.append(assistant_message)
        _render_message(assistant_message)


if __name__ == "__main__":
    main()
