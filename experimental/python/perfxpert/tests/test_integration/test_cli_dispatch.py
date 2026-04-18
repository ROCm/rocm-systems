"""Tests for analyze.py CLI feature-flag dispatch."""

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


def test_cli_default_is_legacy(fake_db, monkeypatch):
    monkeypatch.delenv("PERFXPERT_USE_AGENTS", raising=False)
    with mock.patch.object(analyze_mod, "_execute_legacy") as legacy:
        with mock.patch.object(analyze_mod, "_execute_agentic", create=True) as agentic:
            legacy.return_value = 0
            analyze_mod.execute(input=mock.MagicMock(), format="text")
            legacy.assert_called_once()
            agentic.assert_not_called()


def test_cli_flag_on_routes_to_agentic(fake_db, monkeypatch):
    monkeypatch.setenv("PERFXPERT_USE_AGENTS", "1")
    with mock.patch.object(analyze_mod, "_execute_agentic", create=True) as agentic:
        with mock.patch.object(analyze_mod, "_execute_legacy") as legacy:
            agentic.return_value = 0
            analyze_mod.execute(input=mock.MagicMock(), format="text")
            agentic.assert_called_once()
            legacy.assert_not_called()
