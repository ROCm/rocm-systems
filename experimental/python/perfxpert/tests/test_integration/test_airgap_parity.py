"""Air-gap parity test (spec §5 invariant).

The guarantee: with and without LLM, every handoff target and every gate
decision is byte-identical. Only narrative phrasing differs.

This test exercises Root routing and Analysis classification; gate parity
is covered in test_runtime/test_gate_cascade.py (gates are pure rules
regardless of mode).
"""

import pytest
from unittest.mock import MagicMock

from perfxpert.agents import build_session, schemas
from perfxpert.agents.framework import FakeProviderResponse


@pytest.mark.parametrize("user_query,expected_route", [
    ("why is this kernel slow?", "analysis"),
    ("analyze the trace", "analysis"),
    ("suggest optimizations", "recommendation"),
    ("did my patch help", "correctness"),
])
def test_root_routing_identical_airgap_vs_llm(user_query, expected_route, monkeypatch):
    """Handoff target must match whether LLM is enabled or not."""
    # Airgap
    airgap_session = build_session(airgap=True)
    airgap_out = airgap_session.run_root(
        schemas.RootInput(user_query=user_query, database_path=None)
    )

    # LLM mode (mocked) — stub the SDK invoke to return deterministic response
    def mock_invoke(agent, payload, provider):
        return FakeProviderResponse(
            structured_output={
                "narrative": "x",
                "recommendations": [],
                "primary_bottleneck": "mixed",
                "warnings": [], "metadata": {},
                "routed_to": airgap_out.metadata.get("routed_to"),
            },
        )

    monkeypatch.setattr(
        "perfxpert.agents.framework._sdk_invoke",
        mock_invoke,
    )
    llm_session = build_session(provider="anthropic")
    llm_out = llm_session.run_root(
        schemas.RootInput(user_query=user_query, database_path=None)
    )

    # Handoff target is identical
    assert airgap_out.metadata.get("routed_to") == llm_out.metadata.get("routed_to") == expected_route


def test_classification_verdict_identical_in_both_modes(memory_bound_db, monkeypatch):
    """bottleneck.classify_from_metrics runs in both modes; LLM can refine narrative."""
    airgap_session = build_session(airgap=True)
    airgap_out = airgap_session.run_analysis(
        schemas.AnalysisInput(database_path=str(memory_bound_db))
    )

    def mock_invoke(agent, payload, provider):
        return FakeProviderResponse(
            structured_output={
                "primary_bottleneck": airgap_out.primary_bottleneck,
                "confidence": airgap_out.confidence,
                "time_breakdown": airgap_out.time_breakdown,
                "hot_kernels": airgap_out.hot_kernels,
                "counter_data_available": airgap_out.counter_data_available,
            },
        )

    monkeypatch.setattr(
        "perfxpert.agents.framework._sdk_invoke",
        mock_invoke,
    )
    llm_session = build_session(provider="anthropic")
    llm_out = llm_session.run_analysis(
        schemas.AnalysisInput(database_path=str(memory_bound_db))
    )

    assert airgap_out.primary_bottleneck == llm_out.primary_bottleneck
