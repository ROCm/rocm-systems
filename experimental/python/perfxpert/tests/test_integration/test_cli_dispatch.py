"""Tests for analyze.py CLI feature-flag dispatch (Phase 6: PERFXPERT_LEGACY)."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert import analyze as analyze_mod


@pytest.fixture
def fake_db(tmp_path):
    import sqlite3
    db = tmp_path / "fake.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY, name TEXT, duration_ns INTEGER
        );
        INSERT INTO rocpd_kernel_dispatch VALUES (1, 'matmul', 1000);
    """)
    conn.commit()
    conn.close()
    return db


def test_cli_default_is_agentic(fake_db, monkeypatch):
    """Without PERFXPERT_LEGACY, CLI uses agentic path (Phase 6 default)."""
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        with mock.patch.object(analyze_mod, "_execute_legacy") as legacy:
            agentic.return_value = 0
            analyze_mod.execute(input=mock.MagicMock(), format="text")
            agentic.assert_called_once()
            legacy.assert_not_called()


def test_cli_legacy_flag_on_routes_to_legacy(fake_db, monkeypatch):
    """With PERFXPERT_LEGACY=1, CLI uses legacy path."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    with mock.patch.object(analyze_mod, "_execute_legacy") as legacy:
        with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
            legacy.return_value = 0
            analyze_mod.execute(input=mock.MagicMock(), format="text")
            legacy.assert_called_once()
            agentic.assert_not_called()


@pytest.mark.parametrize(
    "fmt, method_name, rendered",
    [
        ("text", "to_text", "plain report"),
        ("json", "to_json", "{\"ok\": true}"),
        ("markdown", "to_markdown", "# report"),
        ("webview", "to_webview", "<html></html>"),
    ],
)
def test_execute_agentic_uses_existing_result_serializers(fmt, method_name, rendered, capsys):
    """CLI agentic mode should delegate formatting to AnalysisResult methods."""
    result = mock.Mock()
    getattr(result, method_name).return_value = rendered
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.ai_analysis.api.analyze_database", return_value=result) as analyze_db:
        analyze_mod._execute_agentic(
            input=fake_input,
            format=fmt,
            prompt="why is matmul slow?",
            llm="openai",
            top_kernels=3,
        )

    captured = capsys.readouterr()
    assert captured.out.strip() == rendered
    getattr(result, method_name).assert_called_once_with()
    analyze_db.assert_called_once_with(
        database_path="/tmp/fake.db",
        custom_prompt="why is matmul slow?",
        enable_llm=True,
        llm_provider="openai",
        llm_api_key=None,
        llm_thinking_tokens=None,
        verbose=False,
        top_kernels=3,
        att_dir=None,
    )
