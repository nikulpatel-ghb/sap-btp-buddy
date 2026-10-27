from pathlib import Path

from sap_btp_buddy.chunking import WhitespaceTokenizer, chunk_documents, chunk_statistics
from sap_btp_buddy.models import CorpusDocument


def _document() -> CorpusDocument:
    return CorpusDocument(
        Path("guide.md"),
        "\n".join(
            [
                "# Account Guide",
                "## CLI",
                "Use the command carefully.",
                "```text",
                "btp list accounts/subaccount one two three four five six seven eight nine",
                "btp target --subaccount abc",
                "```",
                "## Table",
                "| Name | Value |",
                "| --- | --- |",
                "| plan | free |",
                "| quota | 10 |",
            ]
        )
        + "\n",
        metadata={
            "repository": "SAP-docs/btp-cloud-platform",
            "commit": "920e76fb4c59fd82c0e6757040c38f89a02f0374",
            "path": "docs/guide.md",
            "title": "Account Guide",
            "cleaned_content_sha256": "cleaned-hash",
        },
    )


def test_both_chunkers_respect_cap_and_preserve_context_and_block_metadata() -> None:
    tokenizer = WhitespaceTokenizer()
    for strategy in ("fixed_token", "markdown_heading_aware"):
        chunks = chunk_documents(
            [_document()],
            strategy=strategy,
            chunk_size=16,
            chunk_overlap=3,
            tokenizer=tokenizer,
            input_limit=16,
        )
        assert chunks
        assert all(chunk.metadata["token_count"] <= 16 for chunk in chunks)
        assert all("Account Guide" in chunk.text for chunk in chunks)
        assert all(chunk.metadata["source_line_start"] <= chunk.metadata["source_line_end"] for chunk in chunks)
        assert any(chunk.metadata["contains_code"] for chunk in chunks)
        assert any(chunk.metadata["contains_table"] for chunk in chunks)
        assert any(chunk.metadata["overflow"] for chunk in chunks)
        assert all(chunk.metadata["token_limit_status"] == "PENDING_SELECTED_EMBEDDING_MODEL" for chunk in chunks)


def test_chunk_ids_are_deterministic_and_statistics_are_reported() -> None:
    kwargs = {"strategy": "fixed_token", "chunk_size": 18, "chunk_overlap": 2, "input_limit": 18}
    first = chunk_documents([_document()], **kwargs)
    second = chunk_documents([_document()], **kwargs)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    stats = chunk_statistics(first)
    assert stats["count"] == len(first)
    assert stats["token_min"] <= stats["token_median"] <= stats["token_max"]
    assert 0.0 <= stats["duplicate_rate"] <= 1.0


def test_chunk_metadata_keeps_document_identity_without_repeating_line_map() -> None:
    document = _document()
    document = CorpusDocument(
        document.source_path,
        document.text,
        {**document.metadata, "line_provenance": [{"original_line": index} for index in range(1000)]},
    )
    chunk = chunk_documents([document], strategy="fixed_token", chunk_size=20, chunk_overlap=2, input_limit=20)[0]

    assert chunk.metadata["repository"] == "SAP-docs/btp-cloud-platform"
    assert "line_provenance" not in chunk.metadata
    assert chunk.metadata["source_line_start"] >= 1


def test_html_table_is_classified_as_a_table_block() -> None:
    document = CorpusDocument(Path("html-table.md"), "# T\n<table>\n<tr><td>A</td></tr>\n</table>\n", {"title": "T"})
    chunks = chunk_documents([document], strategy="markdown_heading_aware", chunk_size=20, chunk_overlap=2, input_limit=20)

    assert any(chunk.metadata["contains_table"] for chunk in chunks)
    assert any("<table>" in chunk.metadata["source_text"] for chunk in chunks)


def test_invalid_strategy_and_overlap_fail_explicitly() -> None:
    try:
        chunk_documents([_document()], strategy="semantic")
    except ValueError as error:
        assert "strategy" in str(error)
    else:
        raise AssertionError("invalid strategy was accepted")

    try:
        chunk_documents([_document()], chunk_size=4, chunk_overlap=4)
    except ValueError as error:
        assert "chunk_overlap" in str(error)
    else:
        raise AssertionError("invalid overlap was accepted")
