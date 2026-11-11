from __future__ import annotations

import json

from app.streamlit_app import _resolve_index_path


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
