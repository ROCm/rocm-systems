"""Regression tests for current analysis payload field names."""

import importlib.util
from pathlib import Path

from perfxpert.agents import analysis as analysis_module
from perfxpert.agents import schemas
from perfxpert.analysis.payload import build_analysis_payload
from perfxpert.analysis.recommendations import generate_recommendations
from perfxpert.connection import PerfxpertConnection


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_legacy_ai_analysis_bridge_is_absent() -> None:
    """The deleted bridge that mismatched LLM payload fields is not importable."""
    assert importlib.util.find_spec("perfxpert.ai_analysis") is None


def test_hotspot_payload_uses_current_kernel_metric_fields() -> None:
    """Current hotspots use the field names consumed by recommendations."""
    conn = PerfxpertConnection(str(FIXTURE_DIR / "compute_bound.db"))

    payload = build_analysis_payload(conn, top_kernels=1)

    hotspot = payload["hotspots"][0]
    assert hotspot["calls"] > 0
    assert hotspot["percent_of_total"] > 0.0
    assert "dispatch_count" not in hotspot
    assert "pct_total_time" not in hotspot


def test_analysis_agent_payload_passes_current_hotspot_fields(monkeypatch) -> None:
    """The agent/LLM path receives facts with the current hotspot field names."""
    hotspot = {
        "name": "tiny_kernel",
        "calls": 1001,
        "total_duration_ns": 5_000_000,
        "avg_duration_ns": 4_995,
        "percent_of_total": 72.5,
    }
    facts = {
        "time_breakdown": {
            "total_kernel_time": 5_000_000,
            "kernel_pct": 72.5,
            "memcpy_pct": 0.0,
            "api_overhead_pct": 25.0,
        },
        "hot_kernels": [hotspot],
        "metrics_for_classifier": {
            "api_overhead_pct": 0.25,
            "memcpy_pct": 0.0,
            "avg_kernel_duration_us": 4.995,
            "total_kernel_calls": 1001,
        },
        "counter_data_available": False,
        "db_error": None,
    }
    captured = {}

    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda *args, **kwargs: facts,
    )

    def fake_run_agent(agent, input_payload, **kwargs):
        captured["input_payload"] = input_payload
        return {
            "structured_output": {
                "primary_bottleneck": "latency",
                "confidence": 0.75,
                "time_breakdown": input_payload["facts"]["time_breakdown"],
                "hot_kernels": input_payload["facts"]["hot_kernels"],
                "counter_data_available": False,
            }
        }

    monkeypatch.setattr(analysis_module, "run_agent", fake_run_agent)

    output = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        provider="openai",
        airgap=False,
    )

    agent_hotspot = captured["input_payload"]["facts"]["hot_kernels"][0]
    assert agent_hotspot["calls"] == 1001
    assert agent_hotspot["percent_of_total"] == 72.5
    assert "dispatch_count" not in agent_hotspot
    assert "pct_total_time" not in agent_hotspot
    assert output.hot_kernels[0]["calls"] == 1001
    assert output.hot_kernels[0]["percent_of_total"] == 72.5


def test_recommendations_consume_current_calls_field() -> None:
    """Launch-overhead recommendation depends on the current `calls` field."""
    recs = generate_recommendations(
        time_breakdown={"total_kernel_time": 5_000_000},
        hotspots=[{"name": "tiny_kernel", "calls": 1001, "percent_of_total": 72.5}],
        memory_analysis={},
    )

    launch_recs = [rec for rec in recs if rec.get("category") == "Launch Overhead"]
    assert launch_recs
    assert "1001 launches" in launch_recs[0]["issue"]


def test_recommendations_consume_current_percent_field() -> None:
    """Pragma recommendation depends on the current `percent_of_total` field."""
    recs = generate_recommendations(
        time_breakdown={},
        hotspots=[{"name": "hot_kernel", "calls": 1, "percent_of_total": 25.0}],
        memory_analysis={},
        hardware_counters={
            "has_counters": True,
            "metrics": {"gpu_utilization_percent": 95.0},
        },
    )

    assert any(rec.get("subtype") == "pragma" for rec in recs)
