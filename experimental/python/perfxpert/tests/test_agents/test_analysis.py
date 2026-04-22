"""Isolation tests for Analysis agent (Layer 1)."""

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import analysis as analysis_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import FakeProviderResponse


def _fake_facts(
    *,
    kernel_pct: float = 0.90,
    memcpy_pct: float = 0.05,
    api_pct: float = 0.03,
    counter_data_available: bool = False,
):
    return {
        "time_breakdown": {
            "kernel_pct": kernel_pct,
            "memcpy_pct": memcpy_pct,
            "api_pct": api_pct,
            "idle_pct": max(0.0, 1.0 - kernel_pct - memcpy_pct - api_pct),
        },
        "hot_kernels": [{"name": "[KERNEL_1]", "pct": kernel_pct}],
        "metrics_for_classifier": {},
        "counter_data_available": counter_data_available,
    }


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
    assert agent.allowed_handoffs == []


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
        lambda db, top_n=10: _fake_facts(counter_data_available=True),
    )
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
        lambda db, top_n=10: _fake_facts(kernel_pct=0.55, memcpy_pct=0.40),
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
        lambda db, top_n=10: {
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
        lambda db, top_n=10: _fake_facts(kernel_pct=0.5, memcpy_pct=0.2, api_pct=0.2, counter_data_available=False),
    )
    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        provider="anthropic",
    )
    assert result.counter_data_available is False


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (
            {
                "time_breakdown": {"kernel_pct": 0.20, "memcpy_pct": 0.40, "api_pct": 0.05, "idle_pct": 0.0},
                "hot_kernels": [],
                "metrics_for_classifier": {"memcpy_pct": 0.40},
                "counter_data_available": False,
            },
            "memory_transfer",
        ),
        (
            {
                "time_breakdown": {"kernel_pct": 0.10, "memcpy_pct": 0.05, "api_pct": 0.50, "idle_pct": 0.0},
                "hot_kernels": [{"name": "tiny", "calls": 1200, "duration_ns": 6_000_000}],
                "metrics_for_classifier": {
                    "api_overhead_pct": 0.50,
                    "avg_kernel_duration_us": 5.0,
                    "total_kernel_calls": 1200,
                },
                "counter_data_available": False,
            },
            "latency",
        ),
        (
            {
                "time_breakdown": {"kernel_pct": 0.85, "memcpy_pct": 0.05, "api_pct": 0.05, "idle_pct": 0.0},
                "hot_kernels": [{"name": "matmul", "calls": 50, "duration_ns": 5_000_000}],
                "metrics_for_classifier": {},
                "counter_data_available": True,
            },
            "compute",
        ),
    ],
)
def test_analysis_airgap_uses_legacy_compatible_trace_fallback(monkeypatch, facts, expected):
    monkeypatch.setattr(
        analysis_module,
        "_collect_deterministic_metrics",
        lambda db, top_n=10: facts,
    )
    result = analysis_module.run_analysis(
        schemas.AnalysisInput(database_path="fake.db"),
        airgap=True,
    )
    assert result.primary_bottleneck == expected
