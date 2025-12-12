"""Isolation tests for Analysis agent (Layer 1)."""

import sys
import types

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import analysis as analysis_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import FakeProviderResponse


def test_analysis_agent_builds():
    agent = analysis_module.build_analysis_agent()
    assert agent.name == "Analysis"
    assert agent.layer == 1


def test_analysis_tool_count_exactly_five():
    agent = analysis_module.build_analysis_agent()
    assert len(agent.tools) == 5


def test_analysis_no_execution_tools():
    agent = analysis_module.build_analysis_agent()
    forbidden = {"patch.apply", "patch.revert", "compile.build", "profile.run", "anchors.check"}
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden), "Analysis must have zero execution tools"


def test_analysis_has_no_allowed_handoffs():
    """Layer 1 agents return to Root — they don't fan out (only Recommendation does)."""
    agent = analysis_module.build_analysis_agent()
    assert agent.allowed_handoffs == ()


def test_analysis_classifies_compute_bound(fake_provider, monkeypatch):
    """Given a high-VALU kernel, LLM produces a compute classification."""
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "primary_bottleneck": "compute",
            "confidence": 0.88,
            "time_breakdown": {"kernel_pct": 0.90, "memcpy_pct": 0.05, "api_pct": 0.03, "idle_pct": 0.02},
            "hot_kernels": [{"name": "[KERNEL_1]", "pct": 0.75}],
            "counter_data_available": True,
        },
    )
    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda db, top_n=10: {
            "time_breakdown": {"kernel_pct": 0.90, "memcpy_pct": 0.05, "api_pct": 0.03, "idle_pct": 0.02},
            "hot_kernels": [{"name": "[KERNEL_1]", "pct": 0.75}],
            "metrics_for_classifier": {"valu_util_pct": 0.85},
            "counter_data_available": True,
        },
    )
    # Tools are stubbed via monkeypatch; the agent trusts LLM's synthesis
    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db", top_kernels=10),
        provider="anthropic",
    )
    assert isinstance(result, schemas.AnalysisOutput)
    assert result.primary_bottleneck == "compute"
    assert result.confidence == 0.88


def test_analysis_classifies_memory_bound(fake_provider, monkeypatch):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "primary_bottleneck": "memory_transfer",
            "confidence": 0.80,
            "time_breakdown": {"kernel_pct": 0.55, "memcpy_pct": 0.40, "api_pct": 0.03, "idle_pct": 0.02},
            "hot_kernels": [{"name": "[KERNEL_1]", "pct": 0.40}],
            "counter_data_available": False,
        },
    )
    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda db, top_n=10: {
            "time_breakdown": {"kernel_pct": 0.55, "memcpy_pct": 0.40, "api_pct": 0.03, "idle_pct": 0.02},
            "hot_kernels": [{"name": "[KERNEL_1]", "pct": 0.40}],
            "metrics_for_classifier": {"memcpy_pct": 0.40},
            "counter_data_available": False,
        },
    )
    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        provider="anthropic",
    )
    assert result.primary_bottleneck == "memory_transfer"


def test_analysis_airgap_uses_deterministic_classifier(monkeypatch):
    """Airgap mode must still produce a classification — from
    bottleneck.classify_from_metrics (pure rule).
    """
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    # Stub the time breakdown + hotspots tools
    monkeypatch.setattr(
        analysis_module, "_collect_deterministic_metrics",
        lambda db, top_n=10, min_duration=0.0: {
            "time_breakdown": {"kernel_pct": 0.90, "memcpy_pct": 0.05, "api_pct": 0.03, "idle_pct": 0.02},
            "hot_kernels": [],
            "metrics_for_classifier": {"valu_util_pct": 0.85},
            "counter_data_available": True,
        },
    )
    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        airgap=True,
    )
    # The rule-based classifier in bottleneck.classify_from_metrics should
    # classify as compute given valu_util_pct=0.85
    assert result.primary_bottleneck in ("compute", "mixed")


def test_analysis_airgap_prefers_runtime_overhead_when_it_dominates(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda db, top_n=10, min_duration=0.0: {
            "time_breakdown": {
                "kernel_pct": 8.9,
                "memcpy_pct": 4.5,
                "api_pct": 86.6,
                "idle_pct": 0.0,
            },
            "hot_kernels": [],
            "metrics_for_classifier": {
                "memcpy_pct": 0.045,
                "api_overhead_pct": 0.866,
                "valu_util_pct": 1.0,
                "arithmetic_intensity_above_ridge": 1,
                "gpu_util_pct": 1.0,
            },
            "counter_data_available": True,
        },
    )

    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        airgap=True,
    )

    assert result.primary_bottleneck == "latency"


def test_analysis_airgap_rejects_unknown_rule_bottleneck(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda db, top_n=10: {
            "time_breakdown": {"kernel_pct": 0.90, "memcpy_pct": 0.05, "api_pct": 0.03, "idle_pct": 0.02},
            "hot_kernels": [],
            "metrics_for_classifier": {},
            "counter_data_available": True,
        },
    )
    monkeypatch.setattr(
        analysis_module.bottleneck,
        "classify_from_metrics",
        lambda metrics: {"type": "unknown", "confidence": 0.2},
    )

    with pytest.raises(ValueError, match="unknown"):
        analysis_module.run_analysis(
            schemas.AnalysisInput(database_path="fake.db"),
            airgap=True,
        )


def test_analysis_propagates_counter_availability(fake_provider, monkeypatch):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "primary_bottleneck": "mixed",
            "confidence": 0.5,
            "time_breakdown": {"kernel_pct": 0.5, "memcpy_pct": 0.2, "api_pct": 0.2, "idle_pct": 0.1},
            "hot_kernels": [],
            "counter_data_available": False,
        },
    )
    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda db, top_n=10: {
            "time_breakdown": {"kernel_pct": 0.5, "memcpy_pct": 0.2, "api_pct": 0.2, "idle_pct": 0.1},
            "hot_kernels": [],
            "metrics_for_classifier": {},
            "counter_data_available": False,
        },
    )
    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        provider="anthropic",
    )
    assert result.counter_data_available is False


def test_extract_hw_metrics_uses_detected_arch_specs(monkeypatch):
    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            if isinstance(self._rows, list):
                return self._rows[0] if self._rows else None
            return self._rows

        def fetchall(self):
            if isinstance(self._rows, list):
                return self._rows
            return [self._rows]

    def _fake_execute_statement(_conn, query):
        if "COUNT(*) FROM pmc_events" in query:
            return _Cursor((1,))
        if "GROUP BY counter_name" in query:
            return _Cursor(
                [
                    ("GRBM_GUI_ACTIVE", 100.0, 100.0),
                    ("GRBM_COUNT", 100.0, 100.0),
                    ("FETCH_SIZE", 600.0, 600.0),
                    ("WRITE_SIZE", 400.0, 400.0),
                    ("SQ_INSTS_VALU_MFMA", 35.0, 35.0),
                ]
            )
        if "MAX(end) - MIN(start) FROM kernels" in query:
            return _Cursor((1000,))
        if "FROM rocpd_info_agent" in query:
            return _Cursor(("gfx950",))
        raise AssertionError(f"unexpected query: {query}")

    fake_connection = types.SimpleNamespace(
        PerfxpertConnection=lambda db: object(),
        execute_statement=_fake_execute_statement,
    )
    monkeypatch.setitem(sys.modules, "perfxpert.connection", fake_connection)

    metrics = analysis_module._extract_hw_metrics("fake.db")

    assert metrics["arithmetic_intensity_above_ridge"] == 0
    assert metrics["arithmetic_intensity_below_ridge"] == 1
    assert metrics["hbm_bw_utilization"] == pytest.approx(0.000125, rel=0.01)


def test_extract_hw_metrics_leaves_arch_sensitive_fields_unknown_when_gpu_missing(monkeypatch):
    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            if isinstance(self._rows, list):
                return self._rows[0] if self._rows else None
            return self._rows

        def fetchall(self):
            if isinstance(self._rows, list):
                return self._rows
            return [self._rows]

    def _fake_execute_statement(_conn, query):
        if "COUNT(*) FROM pmc_events" in query:
            return _Cursor((1,))
        if "GROUP BY counter_name" in query:
            return _Cursor(
                [
                    ("FETCH_SIZE", 600.0, 600.0),
                    ("WRITE_SIZE", 400.0, 400.0),
                    ("SQ_INSTS_VALU_MFMA", 35.0, 35.0),
                ]
            )
        if "FROM rocpd_info_agent" in query:
            return _Cursor((None,))
        raise AssertionError(f"unexpected query: {query}")

    fake_connection = types.SimpleNamespace(
        PerfxpertConnection=lambda db: object(),
        execute_statement=_fake_execute_statement,
    )
    monkeypatch.setitem(sys.modules, "perfxpert.connection", fake_connection)

    metrics = analysis_module._extract_hw_metrics("fake.db")

    assert metrics["hbm_bw_utilization"] is None
    assert metrics["arithmetic_intensity_above_ridge"] is None
    assert metrics["arithmetic_intensity_below_ridge"] is None
