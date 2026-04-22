"""Tests for ai_analysis/api.py feature-flag dispatch (Phase 6: PERFXPERT_LEGACY)."""

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


def test_default_path_is_agentic(fake_db, monkeypatch):
    """Without PERFXPERT_LEGACY, the agentic path is used (Phase 6 default)."""
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)

    with mock.patch.object(api, "_route_to_agents") as agentic:
        with mock.patch.object(api, "_route_to_legacy") as legacy:
            agentic.return_value = mock.MagicMock()
            api.analyze_database(database_path=fake_db)
            agentic.assert_called_once()
            legacy.assert_not_called()


def test_legacy_flag_on_routes_to_legacy(fake_db, monkeypatch):
    """With PERFXPERT_LEGACY=1, the legacy path is used."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")

    with mock.patch.object(api, "_route_to_legacy", wraps=api._route_to_legacy) as legacy:
        with mock.patch.object(api, "_route_to_agents") as agentic:
            try:
                api.analyze_database(database_path=fake_db)
            except Exception:
                # legacy path may raise on minimal fixture; what we verify is DISPATCH
                pass
            legacy.assert_called_once()
            agentic.assert_not_called()


@pytest.mark.parametrize("value", ["0", "false", "False", ""])
def test_legacy_flag_off_values_route_to_agentic(value, fake_db, monkeypatch):
    """Explicit falsy values for PERFXPERT_LEGACY route to agentic."""
    monkeypatch.setenv("PERFXPERT_LEGACY", value)
    with mock.patch.object(api, "_route_to_agents") as agentic:
        with mock.patch.object(api, "_route_to_legacy") as legacy:
            agentic.return_value = mock.MagicMock()
            api.analyze_database(database_path=fake_db)
            agentic.assert_called_once()
            legacy.assert_not_called()
