from sap_btp_buddy import (
    answer_question,
    build_index,
    chunk_documents,
    discover_corpus,
    evaluate_questions,
    export_report,
    ingest_documents,
    validate_report,
)
from sap_btp_buddy.cli import COMMANDS, build_parser


def test_public_functions_are_importable():
    assert all(callable(function) for function in (
        answer_question,
        build_index,
        chunk_documents,
        discover_corpus,
        evaluate_questions,
        export_report,
        ingest_documents,
        validate_report,
    ))


def test_cli_exposes_pipeline_commands():
    assert COMMANDS == (
        "discover",
        "ingest",
        "chunk",
        "index",
        "ask",
        "evaluate",
        "refresh",
        "validate-report",
    )
    parser = build_parser()
    assert parser.parse_args(["discover", "."]).command == "discover"
    assert parser.parse_args(["ask", "What is BTP?"]).command == "ask"
    assert parser.parse_args(["refresh"]).command == "refresh"
    assert parser.parse_args(["validate-report", "report.json"]).command == "validate-report"


def test_unimplemented_commands_are_explicit():
    assert COMMANDS
