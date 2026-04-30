"""Checks for the active known-issues document."""

from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"


def test_payload_field_mismatch_is_not_an_active_known_issue() -> None:
    text = (DOCS_ROOT / "known-issues.md").read_text()

    assert "LLM payload field-name mismatch" not in text
    assert "rocm-systems#4979" not in text


def test_payload_field_mismatch_note_is_archived() -> None:
    text = (DOCS_ROOT / "archive" / "migration-to-agentic.md").read_text()

    assert "rocm-systems#4979: LLM payload field-name mismatch" in text
    assert "current agent and deterministic payload paths" in text
