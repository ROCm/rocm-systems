"""Tests for ai_analysis/api.py feature-flag dispatch (PERFXPERT_USE_AGENTS)."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert.ai_analysis import api


@pytest.fixture
def fake_db(tmp_path: Path):
    """Minimal rocpd-like DB fixture."""
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


def test_default_path_is_legacy(fake_db, monkeypatch):
    """Without PERFXPERT_USE_AGENTS, the legacy code path is used."""
    monkeypatch.delenv("PERFXPERT_USE_AGENTS", raising=False)

    with mock.patch.object(api, "_analyze_database_legacy", wraps=api._analyze_database_legacy) as legacy:
        with mock.patch.object(api, "_analyze_database_agentic", create=True) as agentic:
            try:
                api.analyze_database(database_path=fake_db)
            except Exception:
                # legacy path may raise on minimal fixture; what we verify is DISPATCH
                pass
            legacy.assert_called_once()
            agentic.assert_not_called()


def test_flag_on_routes_to_agentic(fake_db, monkeypatch):
    """With PERFXPERT_USE_AGENTS=1, the agentic path is used."""
    monkeypatch.setenv("PERFXPERT_USE_AGENTS", "1")

    with mock.patch.object(api, "_analyze_database_agentic", create=True) as agentic:
        with mock.patch.object(api, "_analyze_database_legacy", wraps=api._analyze_database_legacy) as legacy:
            agentic.return_value = mock.MagicMock()
            api.analyze_database(database_path=fake_db)
            agentic.assert_called_once()
            legacy.assert_not_called()


@pytest.mark.parametrize("value", ["0", "false", "False", ""])
def test_flag_off_values_route_to_legacy(value, fake_db, monkeypatch):
    monkeypatch.setenv("PERFXPERT_USE_AGENTS", value)
    with mock.patch.object(api, "_analyze_database_legacy", wraps=api._analyze_database_legacy) as legacy:
        try:
            api.analyze_database(database_path=fake_db)
        except Exception:
            pass
        legacy.assert_called_once()


def test_flag_truthy_values_route_to_agentic(fake_db, monkeypatch):
    for value in ["1", "true", "True", "yes"]:
        monkeypatch.setenv("PERFXPERT_USE_AGENTS", value)
        with mock.patch.object(api, "_analyze_database_agentic", create=True) as agentic:
            agentic.return_value = mock.MagicMock()
            api.analyze_database(database_path=fake_db)
            agentic.assert_called_once()
            agentic.reset_mock()
