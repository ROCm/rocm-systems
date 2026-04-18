"""Feature-flag default behavior (Phase 6).

Post-Phase-6: PERFXPERT_USE_AGENTS=1 is default.
PERFXPERT_LEGACY=1 opts out to the soon-to-be-removed legacy path.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from perfxpert import analyze


def test_default_routes_to_agent_runtime(monkeypatch, tmp_path):
    """With no env vars set, analyze_database() routes into agent runtime, NOT legacy."""
    # Clear any inherited env
    monkeypatch.delenv("PERFXPERT_USE_AGENTS", raising=False)
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)

    from perfxpert.ai_analysis import api
    db = tmp_path / "fake.db"
    db.touch()

    # Spy the two possible routers
    with patch.object(api, "_route_to_agents", return_value=MagicMock()) as agent_route, \
         patch.object(api, "_route_to_legacy", return_value=MagicMock()) as legacy_route:
        api.analyze_database(database_path=db)

    agent_route.assert_called_once()
    legacy_route.assert_not_called()


def test_PERFXPERT_USE_AGENTS_1_is_redundant_but_harmless(monkeypatch, tmp_path):
    """Users who had PERFXPERT_USE_AGENTS=1 set pre-Phase-6 experience no change."""
    monkeypatch.setenv("PERFXPERT_USE_AGENTS", "1")
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)

    from perfxpert.ai_analysis import api
    db = tmp_path / "fake.db"
    db.touch()

    with patch.object(api, "_route_to_agents", return_value=MagicMock()) as agent_route, \
         patch.object(api, "_route_to_legacy", return_value=MagicMock()) as legacy_route:
        api.analyze_database(database_path=db)

    agent_route.assert_called_once()
    legacy_route.assert_not_called()


def test_PERFXPERT_USE_AGENTS_0_no_longer_disables_agentic_path(monkeypatch, tmp_path):
    """Phase 6: PERFXPERT_USE_AGENTS=0 is IGNORED. Only PERFXPERT_LEGACY=1 triggers legacy."""
    monkeypatch.setenv("PERFXPERT_USE_AGENTS", "0")
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)

    from perfxpert.ai_analysis import api
    db = tmp_path / "fake.db"
    db.touch()

    with patch.object(api, "_route_to_agents", return_value=MagicMock()) as agent_route, \
         patch.object(api, "_route_to_legacy", return_value=MagicMock()) as legacy_route:
        api.analyze_database(database_path=db)

    agent_route.assert_called_once()
    legacy_route.assert_not_called()


def test_PERFXPERT_LEGACY_1_routes_to_legacy(monkeypatch, tmp_path):
    """Explicit opt-in to legacy: env var PERFXPERT_LEGACY=1."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    monkeypatch.delenv("PERFXPERT_USE_AGENTS", raising=False)

    from perfxpert.ai_analysis import api
    db = tmp_path / "fake.db"
    db.touch()

    with patch.object(api, "_route_to_agents", return_value=MagicMock()) as agent_route, \
         patch.object(api, "_route_to_legacy", return_value=MagicMock()) as legacy_route:
        api.analyze_database(database_path=db)

    legacy_route.assert_called_once()
    agent_route.assert_not_called()


def test_PERFXPERT_LEGACY_1_emits_deprecation_warning(monkeypatch, tmp_path, capsys):
    """PERFXPERT_LEGACY=1 must print a visible deprecation warning to stderr."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")

    from perfxpert.ai_analysis import api
    # Reset the deprecation flag so it emits again in this test
    api._emit_legacy_deprecation_once._emitted = False
    db = tmp_path / "fake.db"
    db.touch()

    with patch.object(api, "_route_to_legacy", return_value=MagicMock()):
        api.analyze_database(database_path=db)

    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "DEPRECATED" in combined.upper()
    assert "PERFXPERT_LEGACY" in combined
    assert "vX.Y+1" in combined or "next minor release" in combined.lower() or \
           "will be removed" in combined.lower()


def test_both_flags_set_LEGACY_wins(monkeypatch, tmp_path):
    """Explicit opt-in to legacy takes precedence even if both env vars are set."""
    monkeypatch.setenv("PERFXPERT_USE_AGENTS", "1")
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")

    from perfxpert.ai_analysis import api
    db = tmp_path / "fake.db"
    db.touch()

    with patch.object(api, "_route_to_agents", return_value=MagicMock()) as agent_route, \
         patch.object(api, "_route_to_legacy", return_value=MagicMock()) as legacy_route:
        api.analyze_database(database_path=db)

    legacy_route.assert_called_once()
    agent_route.assert_not_called()
