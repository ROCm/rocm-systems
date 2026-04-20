"""Smoke tests for the 7 agent MCP tool wrappers.

Each tool in ``perfxpert.tools.agents.*`` should:
  1. Be annotated with ``@tool_class(ToolClass.READ_ONLY)``.
  2. Run end-to-end in airgap mode without a live provider.
  3. Return a dict matching the documented agent output schema keys.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from perfxpert.tools._class import ToolClass
from perfxpert.tools.agents import (
    agent_analysis,
    agent_compute_specialist,
    agent_correctness,
    agent_latency_specialist,
    agent_memory_specialist,
    agent_recommendation,
    agent_root,
)


# ---------------------------------------------------------------------------
# Common airgap-mode fixture inputs
# ---------------------------------------------------------------------------


_HOT_KERNELS = [
    {"name": "matmul", "pct": 80.0, "duration_ns": 1_000_000},
]


@pytest.fixture(autouse=True)
def _enable_airgap(monkeypatch):
    """All tests in this module run deterministically in airgap mode."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.delenv("PERFXPERT_LLM_FALLBACK_CHAIN", raising=False)


# ---------------------------------------------------------------------------
# Tool-class annotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        agent_root,
        agent_analysis,
        agent_recommendation,
        agent_correctness,
        agent_compute_specialist,
        agent_memory_specialist,
        agent_latency_specialist,
    ],
)
def test_tool_is_read_only(fn) -> None:
    """Every agent tool must be exposed as READ_ONLY (spec §5.8)."""
    assert fn.__tool_class__ is ToolClass.READ_ONLY, fn


# ---------------------------------------------------------------------------
# Root — end-to-end airgap happy path (no DB required, no source).
# ---------------------------------------------------------------------------


def test_agent_root_airgap_returns_root_output_shape() -> None:
    result = agent_root(user_query="why slow?", airgap=True)
    for key in (
        "narrative",
        "recommendations",
        "primary_bottleneck",
        "warnings",
        "metadata",
    ):
        assert key in result, f"agent_root missing {key!r}: {sorted(result)}"
    assert isinstance(result["recommendations"], list)
    assert isinstance(result["warnings"], list)
    assert isinstance(result["metadata"], dict)


def test_agent_root_honors_airgap_flag() -> None:
    """airgap=True flows through build_session so no provider fires."""
    result = agent_root(
        user_query="anything", provider="anthropic", airgap=True
    )
    # Shape guarantee even when an airgap fallback narrative is used.
    assert "primary_bottleneck" in result


# ---------------------------------------------------------------------------
# Layer-1 agents (Analysis / Recommendation / Correctness)
# ---------------------------------------------------------------------------


def test_agent_analysis_airgap(tmp_path) -> None:
    """Analysis requires a database_path; in airgap mode the agent
    framework short-circuits so a stub path is accepted — the wrapper's
    job is to coerce the dict → Pydantic input and dispatch."""
    db = tmp_path / "fake.db"
    db.write_bytes(b"")  # existence only; airgap path does not read.
    result = agent_analysis(
        input={"database_path": str(db), "top_kernels": 5},
        airgap=True,
    )
    for key in (
        "primary_bottleneck",
        "confidence",
        "time_breakdown",
        "hot_kernels",
        "counter_data_available",
    ):
        assert key in result, f"agent_analysis missing {key!r}"


def test_agent_recommendation_airgap() -> None:
    findings = {
        "primary_bottleneck": "compute",
        "confidence": 0.9,
        "time_breakdown": {
            "kernel_pct": 80.0,
            "memcpy_pct": 5.0,
            "api_pct": 10.0,
            "idle_pct": 5.0,
        },
        "hot_kernels": _HOT_KERNELS,
        "counter_data_available": False,
    }
    result = agent_recommendation(
        input={"findings": findings},
        airgap=True,
    )
    for key in ("recommendations", "specialist_used", "plateau_detected"):
        assert key in result, f"agent_recommendation missing {key!r}"
    assert isinstance(result["recommendations"], list)


def test_agent_correctness_airgap() -> None:
    gate_verdict = {"status": "pass", "failing_gate": None, "detail": "all gates pass"}
    result = agent_correctness(
        input={"gate_verdict": gate_verdict, "kernel_name": "matmul"},
        airgap=True,
    )
    for key in ("verdict", "action", "narrative"):
        assert key in result, f"agent_correctness missing {key!r}"


# ---------------------------------------------------------------------------
# Layer-2 specialists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,input_payload",
    [
        (
            agent_compute_specialist,
            {"gfx_id": "gfx942", "hot_kernels": _HOT_KERNELS},
        ),
        (
            agent_memory_specialist,
            {"gfx_id": "gfx942", "hot_kernels": _HOT_KERNELS},
        ),
        (
            agent_latency_specialist,
            {
                "gfx_id": "gfx942",
                "hot_kernels": _HOT_KERNELS,
                "api_overhead_pct": 15.0,
            },
        ),
    ],
)
def test_specialist_airgap_returns_techniques_shape(
    fn, input_payload: Dict[str, Any]
) -> None:
    result = fn(input=input_payload, airgap=True)
    for key in ("techniques", "confidence", "citations"):
        assert key in result, f"{fn.__name__} missing {key!r}"
    assert isinstance(result["techniques"], list)
    assert isinstance(result["citations"], list)
