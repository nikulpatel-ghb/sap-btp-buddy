"""Conversational Streamlit adapter over the shared answer graph."""

from __future__ import annotations

import html
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
SAP_LOGO_URL = "https://www.sap.com/dam/application/shared/logos/sap-logo-svg.svg/sap-logo-svg.svg"

BRAND_CSS = """
<style>
:root { --sap-navy:#071d36; --sap-blue:#0a6fff; --sap-cyan:#00aeea; --sap-ink:#14213d; --sap-muted:#53627a; --sap-line:#d9e2ee; --sap-pale:#eaf3ff; --sap-mint:#d9f7ef; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] { background:#fff; }
[data-testid="stMainBlockContainer"] { max-width:1480px; padding-top:1.6rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,var(--sap-navy) 0%,#082849 58%,#061a31 100%); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#edf6ff; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
[data-testid="stSidebar"] button { border-color:rgba(255,255,255,.16); background:rgba(255,255,255,.06); color:#fff; text-align:left; }
[data-testid="stSidebar"] button:hover { border-color:#59b9ff; background:rgba(10,111,255,.38); }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background:linear-gradient(90deg,#087cf2,#075fe4); border-color:#2d9cff; }
.brand-lockup { display:flex; align-items:center; gap:12px; margin:0 0 24px; }
.sap-logo-wrap { display:flex; align-items:center; justify-content:center; width:54px; height:36px; flex:0 0 54px; }
.sap-logo { display:block; width:54px; height:auto; max-height:36px; object-fit:contain; }
.sap-fallback { display:none; background:#0a8fe8; color:#fff; font-weight:800; font-size:18px; padding:7px 8px 5px; letter-spacing:-1px; transform:skew(-12deg); border:1px solid #6dd3ff; }
.brand-name { color:#fff; font-size:22px; font-weight:750; line-height:1.05; }
.brand-subtitle { color:#b7d3ee; font-size:12px; margin-top:5px; }
.hero-title { color:var(--sap-ink); font-size:clamp(2rem,3.4vw,3.05rem); font-weight:800; letter-spacing:-.055em; line-height:1.05; }
.hero-slash { color:var(--sap-cyan); padding-left:8px; }
.hero-subtitle { color:#243c62; font-size:1.08rem; margin:8px 0 20px; }
.scope-banner { display:flex; align-items:center; gap:12px; background:var(--sap-pale); border:1px solid #d8e8fb; border-radius:12px; padding:13px 16px; color:var(--sap-ink); margin-bottom:22px; }
.scope-icon { color:#0c5fbf; font-size:24px; }
.scope-label { font-weight:750; margin-right:5px; }
.answer-kicker { color:#0b5db3; font-size:13px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.status-pill { display:inline-block; border-radius:999px; background:var(--sap-mint); color:#087862; font-size:12px; font-weight:750; padding:5px 10px; margin-bottom:9px; }
.callout { background:linear-gradient(90deg,#e9fbf7,#dff8f1); border-radius:12px; padding:13px 15px; color:#173e49; margin:14px 0; }
.sources-title { color:var(--sap-ink); font-size:1.3rem; font-weight:800; margin:0 0 6px; }
.sources-intro { color:var(--sap-muted); margin-bottom:15px; }
.source-meta { color:var(--sap-muted); font-size:12px; margin:2px 0 9px; }
.source-excerpt { color:#293b56; font-size:13px; line-height:1.42; margin-bottom:8px; }
.source-badge { display:inline-block; float:right; color:#087862; background:var(--sap-mint); border-radius:999px; padding:4px 8px; font-size:11px; font-weight:750; }
.source-card-title { color:var(--sap-ink); font-size:14px; font-weight:750; line-height:1.25; padding-right:72px; }
.sidebar-section-title { color:#fff; font-size:14px; font-weight:750; margin:18px 0 10px; }
.sidebar-note { color:#b7d3ee; font-size:12px; line-height:1.45; }
div[data-testid="stChatMessage"] { border-radius:14px; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border:1px solid var(--sap-line); padding:1rem; box-shadow:0 7px 22px rgba(20,48,87,.05); }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#eef6ff; }
[data-testid="stChatInput"] { border-color:var(--sap-blue); box-shadow:0 4px 16px rgba(10,111,255,.08); }
</style>
"""


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


def _render_sources_panel(result: Any | None) -> None:
    """Render approved source cards for the most recent answer."""

    st.markdown('<div class="sources-title">Sources for this answer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sources-intro">The answer is based on approved SAP BTP documentation.</div>', unsafe_allow_html=True)
    if result is None or not getattr(result, "evidence", ()):
        st.info("Sources will appear here after a validated answer is produced.")
        return
    seen_sources: set[str] = set()
    for item in getattr(result, "evidence", ()):
        source_key = str(item.get("source_url") or item.get("path") or item.get("title") or "unknown")
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        title = str(item.get("title") or item.get("path") or "Untitled source")
        excerpt = " ".join(str(item.get("text") or "").split())
        if len(excerpt) > 280:
            excerpt = excerpt[:277].rstrip() + "..."
        lines = f"Lines {item.get('source_line_start', '?')}-{item.get('source_line_end', '?')}"
        with st.container(border=True):
            st.markdown(f'<span class="source-badge">Retrieved</span><div class="source-card-title">{html.escape(title)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="source-meta">SAP Help Portal - SAP BTP</div>', unsafe_allow_html=True)
            st.caption(lines)
            st.markdown(f'<div class="source-excerpt">&ldquo;{html.escape(excerpt)}&rdquo;</div>', unsafe_allow_html=True)
            if item.get("source_url"):
                st.link_button("Open source", item["source_url"], icon=":material/open_in_new:", width="content")


def _render_result(result: Any, elapsed_ms: float) -> None:
    status = getattr(result, "status", "error")
    labels = {"answered": "Answer", "clarify": "Clarification needed", "insufficient_evidence": "Insufficient evidence", "error": "Operational error"}
    st.markdown(f'<div class="answer-kicker">{html.escape(labels.get(status, "Result"))}</div>', unsafe_allow_html=True)
    if status == "answered":
        st.markdown('<span class="status-pill">Validated against approved sources</span>', unsafe_allow_html=True)
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

    with st.chat_message("assistant", avatar=SAP_LOGO_URL):
        result = message.get("result")
        if result is None:
            st.error(message.get("content", "The assistant could not produce a result."))
            return
        _render_result(result, float(message.get("elapsed_ms", 0.0)))


def _render_sidebar(runtime: Runtime) -> None:
    with st.sidebar:
        st.markdown(f'<div class="brand-lockup"><div class="sap-logo-wrap"><img class="sap-logo" src="{SAP_LOGO_URL}" alt="SAP logo"><span class="sap-fallback">SAP</span></div><div><div class="brand-name">SAP BTP Buddy</div><div class="brand-subtitle">Documentation-grounded AI assistant</div></div></div>', unsafe_allow_html=True)
        if st.button("New conversation", icon=":material/add:", type="primary", width="stretch"):
            for key in ("pending_question", "messages"):
                st.session_state.pop(key, None)
            st.rerun()
        st.markdown('<div class="sidebar-section-title">Example questions</div>', unsafe_allow_html=True)
        for index, example in enumerate(EXAMPLES):
            if st.button(example, key=f"example_{index}", icon=":material/chat_bubble_outline:", width="stretch"):
                st.session_state["pending_question"] = example
                st.rerun()
        st.divider()
        st.markdown('<div class="sidebar-section-title">Approved SAP documentation</div>', unsafe_allow_html=True)
        st.success("In scope", icon=":material/verified_user:")
        st.markdown(f'<div class="sidebar-note">Snapshot: {html.escape(runtime.snapshot_date)}</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="sidebar-section-title">Technical details</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-note">Data source: SAP Help Portal<br><br>Retrieval: {html.escape(runtime.retrieval_mode)} - top-k 5<br><br>Index: {html.escape(Path(runtime.index_path).name)}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="SAP BTP Buddy", page_icon=":material/menu_book:", layout="wide", initial_sidebar_state="expanded")
    st.html(BRAND_CSS)
    st.session_state.setdefault("messages", [])
    index_path = os.getenv("BTP_INDEX_PATH", str(PROJECT_ROOT / "indexes" / "active"))
    retrieval_mode = os.getenv("BTP_RETRIEVAL_MODE", "hybrid")
    runtime = load_runtime(index_path, retrieval_mode)

    _render_sidebar(runtime)

    st.markdown('<div class="hero-title">SAP BTP Buddy <span class="hero-slash">/</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Your documentation-grounded assistant for SAP Business Technology Platform (SAP BTP)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="scope-banner"><span class="scope-icon">[book]</span><span><span class="scope-label">Documentation scope:</span> Approved SAP BTP documentation (SAP Help Portal) &nbsp;|&nbsp; Snapshot: {html.escape(runtime.snapshot_date)}</span></div>', unsafe_allow_html=True)

    if runtime.startup_error:
        st.error(runtime.startup_error)
        st.info("The UI is ready, but no answer will be produced until the validated index and provider configuration are available.")

    # Resolve and persist the submitted turn before rendering the columns so the
    # right-hand source panel is populated during the same Streamlit rerun.
    question = st.chat_input("Ask one SAP BTP documentation question", submit_mode="disable")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        if runtime.graph is None:
            st.session_state.messages.append({"role": "assistant", "content": "Startup fallback: the shared graph is unavailable, so no answer was attempted."})
        else:
            record = {"question": question, "support_status": "RUNTIME", "labels": [], "evidence_groups": []}
            started = time.perf_counter()
            with st.spinner("Retrieving approved evidence and validating the answer..."):
                try:
                    state = runtime.graph.invoke({"question": record})
                    result = state["result"]
                    st.session_state.messages.append({"role": "assistant", "result": result, "elapsed_ms": (time.perf_counter() - started) * 1000})
                except Exception:
                    st.session_state.messages.append({"role": "assistant", "content": "The shared answer graph failed before producing a validated result. Please retry later."})

    latest_result = next((message.get("result") for message in reversed(st.session_state.messages) if message.get("role") == "assistant" and message.get("result") is not None), None)
    conversation_col, sources_col = st.columns([2.15, 1.0], gap="large")
    with conversation_col:
        if not st.session_state.messages:
            with st.container(border=True):
                st.markdown("### Ask about SAP BTP documentation")
                st.markdown("Get concise, evidence-linked answers from the approved SAP Help Portal snapshot.")
                st.markdown('<div class="callout"><strong>Each question is answered independently.</strong><br>Try one of the example questions in the sidebar to get started.</div>', unsafe_allow_html=True)
        for message in st.session_state.messages:
            _render_message(message)
    with sources_col:
        with st.container(border=True):
            _render_sources_panel(latest_result)



if __name__ == "__main__":
    main()
