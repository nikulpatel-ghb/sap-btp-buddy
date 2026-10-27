from sap_btp_buddy.corpus import refresh_allowlist


def test_refresh_delete_removes_stale_selected_record(tmp_path):
    root = tmp_path / "fixture-repo"
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("# A\n", encoding="utf-8")
    (docs / "b.md").write_text("# B\n", encoding="utf-8")
    allowlist = ["docs/a.md", "docs/b.md"]
    (docs / "b.md").unlink()
    refreshed = refresh_allowlist(root, allowlist)
    assert refreshed == {"included": ["docs/a.md"], "stale_deleted": ["docs/b.md"], "new_unallowlisted": []}
