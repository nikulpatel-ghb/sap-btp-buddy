import json

from sap_btp_buddy.reporting import validate_report


def test_validate_report_flags_pending_and_broken_links(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Report\nStatus: PENDING\n[missing](missing.json)\n", encoding="utf-8")

    result = validate_report(report)

    assert result["status"] == "FAIL"
    assert result["summary"]["pending_fields"] == 1
    assert result["summary"]["broken_links"] == 1


def test_validate_report_passes_existing_and_external_links(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text(
        "# Report\n[Evidence](evidence.json)\n[Docs](https://example.com/docs)\n",
        encoding="utf-8",
    )

    result = validate_report(report)

    assert result["status"] == "PASS"
    assert result["summary"] == {
        "missing_files": 0,
        "broken_links": 0,
        "pending_fields": 0,
        "external_links": 1,
    }
