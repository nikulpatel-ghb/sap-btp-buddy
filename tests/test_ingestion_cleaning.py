from pathlib import Path

from sap_btp_buddy import build_github_source_url, clean_document, discover_corpus, ingest_documents
from sap_btp_buddy.models import CorpusDocument


def test_cli_and_fenced_code_are_preserved_with_line_provenance() -> None:
    source = "# CLI\n\n```text\nbtp list accounts/subaccount\n```\n"
    cleaned = clean_document(CorpusDocument(Path("cli.md"), source))

    assert "btp list accounts/subaccount" in cleaned.text
    assert "```text\n" in cleaned.text
    assert cleaned.metadata["line_provenance"][3]["original_line_start"] == 4
    assert cleaned.metadata["cleaned_content_sha256"]


def test_markdown_table_is_retained() -> None:
    source = "# Table\n\n| Name | Value |\n| --- | --- |\n| plan | free |\n"
    cleaned = clean_document(CorpusDocument(Path("table.md"), source))

    assert "| Name | Value |" in cleaned.text
    assert "| plan | free |" in cleaned.text
    assert cleaned.metadata["removed_line_count"] == 0


def test_only_proven_sap_boilerplate_lines_are_excluded() -> None:
    source = (
        "<!-- loio1234abcd -->\n"
        '<link rel="stylesheet" type="text/css" href="../css/sap-icons.css"/>\n'
        "# CLI\n"
        "`btp login`\n"
        "\n"
    )
    cleaned = clean_document(CorpusDocument(Path("boilerplate.md"), source))

    assert "loio1234abcd" not in cleaned.text
    assert "sap-icons.css" not in cleaned.text
    assert "# CLI" in cleaned.text
    assert "`btp login`" in cleaned.text
    assert [event["reason"] for event in cleaned.metadata["cleaning_exclusions"]] == [
        "terminal_blank_line",
        "sap_icons_stylesheet_link",
        "sap_loio_html_comment",
    ]


def test_relative_links_are_retained_and_url_uses_validated_commit_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source_path = root / "docs" / "guide.md"
    source_path.parent.mkdir()
    source_path.write_text("# Guide\nSee [other](other.md).\n", encoding="utf-8")

    url = build_github_source_url(
        "SAP-docs/btp-cloud-platform",
        "920e76fb4c59fd82c0e6757040c38f89a02f0374",
        "docs/guide.md",
        repository_root=root,
    )
    document = ingest_documents(
        [source_path],
        repository="SAP-docs/btp-cloud-platform",
        commit="920e76fb4c59fd82c0e6757040c38f89a02f0374",
        repository_root=root,
        allowed_paths=["docs/guide.md"],
        ingested_at_utc="2026-10-19T00:00:00Z",
    )[0]

    assert document.metadata["source_url"] == url
    assert "[other](other.md)" in document.text
    assert document.metadata["path"] == "docs/guide.md"
    assert document.metadata["ingestion_time_utc"] == "2026-10-19T00:00:00Z"


def test_allowlist_excludes_unselected_paths_and_order_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "b.md").parent.mkdir(parents=True)
    (root / "b.md").write_text("# B\n", encoding="utf-8")
    (root / "a.md").write_text("# A\n", encoding="utf-8")
    (root / "not-markdown.txt").write_text("not evidence\n", encoding="utf-8")

    discovered = discover_corpus(root)
    documents = ingest_documents(
        discovered,
        repository="SAP-docs/btp-cloud-platform",
        commit="920e76fb4c59fd82c0e6757040c38f89a02f0374",
        repository_root=root,
        allowed_paths=["b.md"],
    )

    assert [path.name for path in discovered] == ["a.md", "b.md"]
    assert [doc.metadata["path"] for doc in documents] == ["b.md"]
    assert "not evidence" not in documents[0].text
