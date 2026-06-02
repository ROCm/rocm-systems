"""Tests for the _TraceAnalysisAdapter shim in agents/analysis.py.

The shim is always active in production because perfxpert.tools.trace_analysis
does not exist. Every agent test mocks the tool directly; these tests exercise
the shim's delegation to analyze.py.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal PerfxpertConnection stand-in
# ---------------------------------------------------------------------------

class _FakeConn:
    """Stand-in for PerfxpertConnection — accepted by monkeypatched functions."""
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shim_delegates_compute_time_breakdown_to_analyze_module(monkeypatch):
    """Monkeypatch perfxpert.analyze.compute_time_breakdown; call the shim;
    assert the returned dict is derived from that patched function's output."""
    import perfxpert.agents.analysis as analysis_mod

    # The shim is a _TraceAnalysisAdapter instance bound to `trace_analysis`.
    # We need to confirm it uses analyze.compute_time_breakdown internally.
    # Patch the analyze module function AND PerfxpertConnection constructor.
    fake_breakdown_result = {
        "kernel_percent": 72.5,
        "memcpy_percent": 18.3,
        "overhead_percent": 9.2,
    }

    import perfxpert.analyze as analyze_mod_ref
    import perfxpert.connection as conn_mod

    monkeypatch.setattr(analyze_mod_ref, "compute_time_breakdown", lambda conn: fake_breakdown_result)
    monkeypatch.setattr(conn_mod, "PerfxpertConnection", _FakeConn)

    # Also stub _check_counters_available so it doesn't try to open a real DB.
    # The shim's time_breakdown method calls it internally (defined at module closure).
    # Patch via the analysis module namespace.
    monkeypatch.setattr(analysis_mod, "_check_counters_available", lambda db_path: False)

    result = analysis_mod.trace_analysis.time_breakdown("fake.db")

    assert result["kernel_pct"] == pytest.approx(72.5)
    assert result["memcpy_pct"] == pytest.approx(18.3)
    assert result["api_pct"] == pytest.approx(9.2)
    assert result["idle_pct"] == 0.0
    assert result["counter_data_available"] is False


def test_shim_delegates_identify_hotspots_to_analyze_module(monkeypatch):
    """Monkeypatch perfxpert.analyze.identify_hotspots; call the shim's hotspots();
    assert the return value matches the patched function's output."""
    import perfxpert.agents.analysis as analysis_mod
    import perfxpert.analyze as analyze_mod_ref
    import perfxpert.connection as conn_mod

    fake_hotspots = [
        {"name": "heavy_kernel", "pct": 0.75},
        {"name": "light_kernel", "pct": 0.15},
    ]
    monkeypatch.setattr(analyze_mod_ref, "identify_hotspots", lambda conn, top_n=10: fake_hotspots)
    monkeypatch.setattr(conn_mod, "PerfxpertConnection", _FakeConn)

    result = analysis_mod.trace_analysis.hotspots("fake.db", top_n=5)

    assert result == fake_hotspots
    assert len(result) == 2
    assert result[0]["name"] == "heavy_kernel"


def test_shim_used_when_trace_analysis_module_absent(monkeypatch):
    """Simulate absent perfxpert.tools.trace_analysis and reload the module;
    assert that trace_analysis is a _TraceAnalysisAdapter instance (the shim)."""
    # Hide the trace_analysis module so the ImportError branch executes.
    saved = sys.modules.pop("perfxpert.tools.trace_analysis", None)
    # Also hide the key so the try-block import fails rather than finds a cached value.
    # Insert a sentinel that causes ImportError when imported.
    sys.modules["perfxpert.tools.trace_analysis"] = None  # type: ignore[assignment]

    try:
        # Remove the cached analysis module so the try/except re-runs on reload.
        saved_analysis = sys.modules.pop("perfxpert.agents.analysis", None)
        import perfxpert.agents.analysis as fresh_mod
        importlib.reload(fresh_mod)

        ta = fresh_mod.trace_analysis
        # Must be the shim: has both expected method names
        assert hasattr(ta, "time_breakdown"), "shim must expose time_breakdown"
        assert hasattr(ta, "hotspots"), "shim must expose hotspots"
        # And must NOT be a real module (the shim is an instance, not a module)
        assert not isinstance(ta, type(sys)), "_TraceAnalysisAdapter shim should not be a module"
    finally:
        # Restore original state
        sys.modules.pop("perfxpert.tools.trace_analysis", None)
        if saved is not None:
            sys.modules["perfxpert.tools.trace_analysis"] = saved
        # Reload to restore to normal state
        import perfxpert.agents.analysis as analysis_mod
        importlib.reload(analysis_mod)
