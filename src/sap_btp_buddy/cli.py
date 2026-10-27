"""Single CLI adapter over the public package functions."""

import argparse
from pathlib import Path

from .reporting import validate_report


COMMANDS = (
    "discover",
    "ingest",
    "chunk",
    "index",
    "ask",
    "evaluate",
    "refresh",
    "validate-report",
)

UNIMPLEMENTED_COMMANDS = frozenset(COMMANDS)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without executing any pipeline work."""

    parser = argparse.ArgumentParser(prog="sap-btp-buddy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="discover corpus files")
    discover.add_argument("root", type=Path)

    ingest = subparsers.add_parser("ingest", help="load and clean discovered documents")
    ingest.add_argument("manifest", type=Path)

    chunk = subparsers.add_parser("chunk", help="chunk ingested documents")
    chunk.add_argument("documents", type=Path)

    index = subparsers.add_parser("index", help="build dense, lexical, or hybrid indexes")
    index.add_argument("chunks", type=Path)
    index.add_argument("--output", type=Path, default=Path("artifacts/index"))

    answer = subparsers.add_parser("ask", help="answer one question with citations")
    answer.add_argument("question")
    answer.add_argument("--index", type=Path, default=Path("artifacts/index"))

    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen question set")
    evaluate.add_argument("questions", type=Path)
    evaluate.add_argument("--index", type=Path, default=Path("artifacts/index"))

    refresh = subparsers.add_parser("refresh", help="refresh a pinned corpus and rebuild artifacts")
    refresh.add_argument("--config", type=Path, default=Path("config.yaml"))

    report = subparsers.add_parser("validate-report", help="validate report completeness and evidence")
    report.add_argument("report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse a command and dispatch to its library implementation."""

    args = build_parser().parse_args(argv)
    if args.command == "validate-report":
        result = validate_report(args.report)
        print(__import__("json").dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1

    print(
        f"PENDING: CLI command '{args.command}' is defined but not implemented; no result was produced.",
        file=__import__("sys").stderr,
    )
    return 2
