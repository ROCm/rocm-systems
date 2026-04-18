"""Tests for ai_analysis/api.py feature-flag dispatch (Phase 6: PERFXPERT_LEGACY)."""

import json
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.agents import runtime, schemas
from perfxpert.ai_analysis import api
from perfxpert.providers._exceptions import AuthError, RateLimitError


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
        agentic.return_value = mock.MagicMock()
        api.analyze_database(database_path=fake_db)
        agentic.assert_called_once()


def test_route_to_agents_wraps_output_for_existing_serializers(fake_db):
    """Agentic API results must still round-trip through the legacy JSON serializer."""
    session = mock.Mock()
    session.run_analysis.return_value = schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.8,
        time_breakdown={
            "kernel_pct": 0.90,
            "memcpy_pct": 0.05,
            "api_pct": 0.03,
            "idle_pct": 0.02,
        },
        hot_kernels=[{"name": "matmul", "pct": 0.90, "duration_ns": 1000, "calls": 2}],
        counter_data_available=False,
    )
    session.run_recommendation.return_value = schemas.RecommendationOutput(
        recommendations=[
            {
                "name": "launch_bounds",
                "rationale": "Reduce VGPR pressure.",
                "expected_impact": 0.25,
                "risk": "low",
            }
        ],
        specialist_used="compute",
        plateau_detected=False,
    )

    with mock.patch.object(runtime, "build_session", return_value=session):
        result = api._route_to_agents(fake_db)

    payload = json.loads(result.to_json())
    assert payload["summary"]["primary_bottleneck"] == "compute"
    assert payload["hotspots"][0]["name"] == "matmul"
    assert payload["recommendations"][0]["issue"] == "launch_bounds"


def test_route_to_agents_preserves_auth_errors(fake_db):
    """Provider auth failures must keep the library's auth exception contract."""
    session = mock.Mock()
    session.run_analysis.side_effect = AuthError("openai", "bad key")

    with mock.patch.object(runtime, "build_session", return_value=session):
        with pytest.raises(api.LLMAuthenticationError):
            api._route_to_agents(fake_db, enable_llm=True, llm_provider="openai")


def test_route_to_agents_preserves_rate_limit_errors(fake_db):
    """Provider rate-limit failures must keep the library's rate-limit contract."""
    session = mock.Mock()
    session.run_analysis.side_effect = RateLimitError("openai", retry_after=30.0)

    with mock.patch.object(runtime, "build_session", return_value=session):
        with pytest.raises(api.LLMRateLimitError):
            api._route_to_agents(fake_db, enable_llm=True, llm_provider="openai")


def test_route_to_agents_wraps_non_provider_failures(fake_db):
    """Unrelated agentic failures should no longer masquerade as database corruption."""
    session = mock.Mock()
    session.run_analysis.side_effect = ValueError("boom")

    with mock.patch.object(runtime, "build_session", return_value=session):
        with pytest.raises(RuntimeError, match="Agentic analysis failed"):
            api._route_to_agents(fake_db)
