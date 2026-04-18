"""Tests for analyze.py CLI dispatch (Phase 7.1: agentic is the only path)."""

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


def test_cli_always_runs_agentic(fake_db, monkeypatch):
    """CLI always uses the agentic path; no feature-flag branching remains."""
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        analyze_mod.execute(input=mock.MagicMock(), format="text")
        agentic.assert_called_once()


def test_cli_legacy_flag_is_no_op(fake_db, monkeypatch):
    """PERFXPERT_LEGACY=1 is unrecognized post-Phase-7.1 — still routes agentic."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    with mock.patch.object(analyze_mod, "_execute_agentic") as agentic:
        agentic.return_value = 0
        analyze_mod.execute(input=mock.MagicMock(), format="text")
        agentic.assert_called_once()


def test_legacy_symbols_are_absent():
    """The _execute_legacy function and ai_analysis package must be gone."""
    assert not hasattr(analyze_mod, "_execute_legacy"), (
        "_execute_legacy should have been deleted in Phase 7.1"
    )
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("perfxpert.ai_analysis")
