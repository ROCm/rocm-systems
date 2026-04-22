"""Tests for ai_analysis/api.py feature-flag dispatch (Phase 6: PERFXPERT_LEGACY)."""

import json
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.ai_analysis import api
from perfxpert.ai_analysis.exceptions import DatabaseNotFoundError
from perfxpert.agents import schemas


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


def _make_legacy_result(database_path: Path) -> api.AnalysisResult:
    recommendation = api.Recommendation(
        id="rec_001",
        priority="high",
        category="compute",
        title="Use MFMA intrinsics",
        description="Legacy recommendation payload",
        estimated_impact="High",
    )
    result = api.AnalysisResult(
        metadata=api.AnalysisMetadata(rocpd_version="0.1.0", database_file=str(database_path)),
        profiling_info=api.ProfilingInfo(
            total_duration_ns=123,
            profiling_mode="thread_trace",
            analysis_tier=3,
            gpus=[],
        ),
        summary=api.AnalysisSummary(
            overall_assessment="legacy",
            primary_bottleneck="latency",
            confidence=0.42,
            key_findings=["legacy finding"],
        ),
        execution_breakdown=api.ExecutionBreakdown(
            kernel_time_ns=60,
            kernel_time_pct=60.0,
            memcpy_time_ns=20,
            memcpy_time_pct=20.0,
            api_overhead_pct=10.0,
            idle_time_pct=10.0,
        ),
        recommendations=api.RecommendationSet(high_priority=[recommendation]),
        warnings=[],
    )
    result._raw = {
        "time_breakdown": {
            "total_runtime": 100,
            "total_kernel_time": 60,
            "total_memcpy_time": 20,
            "kernel_percent": 60.0,
            "memcpy_percent": 20.0,
            "overhead_percent": 10.0,
            "kernel_pct": 0.6,
            "memcpy_pct": 0.2,
            "api_pct": 0.1,
            "idle_pct": 0.1,
        },
        "hotspots": [{"name": "legacy_kernel", "calls": 3, "total_duration": 60}],
        "memory_analysis": {},
        "recommendations_raw": [{"category": "compute", "issue": "legacy payload"}],
        "hardware_counters": {
            "has_counters": True,
            "metrics": {},
            "counters": {
                "SQ_WAVES": {
                    "sample_count": 1,
                    "avg_value": 1.0,
                    "min_value": 1.0,
                    "max_value": 1.0,
                    "total_value": 1.0,
                }
            },
        },
        "database_path": str(database_path),
        "att_trace": {"has_att_data": True, "summary": {"kernel_count": 1}},
        "att_analysis": {"has_att_data": True, "summary": {"kernel_count": 1}},
    }
    return result


def test_agentic_route_preserves_legacy_recommendations_and_att(fake_db, monkeypatch):
    legacy_result = _make_legacy_result(fake_db)
    session = mock.MagicMock()
    session.run_analysis.return_value = schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.91,
        time_breakdown={
            "kernel_pct": 0.82,
            "memcpy_pct": 0.08,
            "api_pct": 0.05,
            "idle_pct": 0.05,
        },
        hot_kernels=[{"name": "agentic_kernel", "pct": 0.82, "duration_ns": 1200}],
        counter_data_available=True,
    )

    monkeypatch.setattr("perfxpert.agents.runtime.build_session", lambda **_: session)
    monkeypatch.setattr(api, "_route_to_legacy", lambda *args, **kwargs: legacy_result)

    result = api._route_to_agents(database_path=fake_db, att_dir="/tmp/att")

    assert result.summary.primary_bottleneck == "compute"
    assert result.summary.confidence == 0.91
    assert result.recommendations.high_priority[0].id == "rec_001"
    assert result._raw["recommendations_raw"][0]["category"] == "compute"
    assert result.profiling_info.analysis_tier == 3
    assert result.profiling_info.profiling_mode == "thread_trace"
    assert result._raw["att_trace"]["has_att_data"] is True
    assert result.execution_breakdown.kernel_time_ns == 82
    doc = json.loads(result.to_json())
    assert doc["summary"]["primary_bottleneck"] == "compute"
    assert doc["execution_breakdown"]["kernel_time_pct"] == 82.0


def test_agentic_route_maps_missing_database_to_public_error(fake_db, monkeypatch):
    session = mock.MagicMock()
    session.run_analysis.side_effect = FileNotFoundError("missing db")

    monkeypatch.setattr("perfxpert.agents.runtime.build_session", lambda **_: session)

    with pytest.raises(DatabaseNotFoundError, match="Database file not found"):
        api._route_to_agents(database_path=fake_db)
