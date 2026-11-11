from __future__ import annotations

import json

from app.streamlit_app import _resolve_index_path
from streamlit.testing.v1 import AppTest


def _write_index_marker(root, name: str, retrieval: str):
    path = root / name
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"retrieval": retrieval}), encoding="utf-8")
    (path / "build.complete").write_text("ok\n", encoding="utf-8")
    return path


def test_active_container_selects_index_for_requested_retrieval_mode(tmp_path):
    hybrid = _write_index_marker(tmp_path, "hybrid-id", "hybrid")
    _write_index_marker(tmp_path, "dense-id", "dense")

    assert _resolve_index_path(str(tmp_path), "hybrid") == hybrid
    assert _resolve_index_path(str(tmp_path), "dense").name == "dense-id"


def test_branded_empty_state_renders_without_runtime_dependencies():
    script = """
import streamlit as st
import app.streamlit_app as ui

ui.load_runtime = lambda index_path, retrieval_mode: ui.Runtime(
    None, None, "2025-01-15", "hybrid-demo", retrieval_mode
)
ui.main()
"""
    app = AppTest.from_string(script).run(timeout=15)

    assert not app.exception
    assert any("New conversation" in button.label for button in app.button)
    assert any("Example questions" in item.value for item in app.markdown)
    assert any("Sources for this answer" in item.value for item in app.markdown)
    assert app.chat_input[0].placeholder == "Ask one SAP BTP documentation question"


def test_sources_panel_populates_from_latest_answer_in_same_run():
    script = """
from types import SimpleNamespace
import streamlit as st
import app.streamlit_app as ui

result = SimpleNamespace(
    status="answered",
    answer="A validated answer.",
    evidence=[{
        "title": "SAP BTP: Account Model",
        "path": "docs/account-model.md",
        "text": "The global account is the top-level account.",
        "source_line_start": 10,
        "source_line_end": 14,
        "source_url": "https://example.com/pinned-source",
    }],
)
class FakeGraph:
    def invoke(self, state):
        return {"result": result}

ui.load_runtime = lambda index_path, retrieval_mode: ui.Runtime(
    FakeGraph(), None, "2025-01-15", "hybrid-demo", retrieval_mode
)
st.session_state["pending_question"] = "How is an SAP BTP account structured?"
ui.main()
"""
    app = AppTest.from_string(script).run(timeout=15)

    assert not app.exception
    assert any("SAP BTP: Account Model" in item.value for item in app.markdown)
    assert any(link.url == "https://example.com/pinned-source" for link in app.link_button)
