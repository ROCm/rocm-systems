"""Air-gap parity test (spec §5 invariant).

The guarantee: with and without an LLM, every handoff target, every gate
decision, and every ranked recommendation is identical. Only narrative
phrasing and the separately-carried exploratory lane differ.

Parity is only worth asserting against a model that is *trying* to break it.
A mock that echoes the air-gap answer back proves nothing — it passes whether
or not the runtime consults the model. So the model here returns hostile
output on every deterministic field, and parity must survive it.

Gate parity is covered in test_runtime/test_gate_cascade.py (gates are pure
rules regardless of mode).
"""

import pytest

from perfxpert.agents import build_session, schemas
from perfxpert.agents.framework import FakeProviderResponse

# Values a model would emit if it were trying to redirect routing, inflate
# its own confidence, invent findings, or reorder ranked advice.
HOSTILE = {
    "routed_to": "correctness",
    "primary_bottleneck": "compute_bound",
    "confidence": 1.0,
    "time_breakdown": {"kernel_pct": 100.0},
    "hot_kernels": [{"name": "[INVENTED]", "pct": 0.99}],
    "counter_data_available": True,
    "techniques": [{"name": "invented_technique", "expected_impact": 0.99}],
    "citations": ["fabricated"],
    "recommendations": [{"name": "invented_recommendation", "expected_impact": 0.99}],
    "specialist_used": "compute",
    "plateau_detected": True,
    "wall_delta_pct": -99.0,
    "verdict": "improved",
    "kernel_deltas": {"regressions": [], "improvements": [{"name": "[INVENTED]"}]},
    "narrative": "the model's own words",
}


@pytest.fixture
def hostile_llm(monkeypatch):
    """Every LLM call returns output designed to break parity.

    Records invocations so each test can prove the live path was really
    taken. Without that check a provider-resolution change could silently
    route "live" sessions back to air-gap, and every parity test here would
    keep passing while asserting nothing.
    """
    invoked = []

    def mock_invoke(agent, payload, provider):
        invoked.append(agent.name)
        return FakeProviderResponse(structured_output=dict(HOSTILE))

    monkeypatch.setattr("perfxpert.agents.framework._sdk_invoke", mock_invoke)
    return invoked


def _both_modes(method, payload, invoked):
    """Run the same call air-gapped and live; return (airgap_out, live_out)."""
    airgap_session = build_session(airgap=True)
    airgap = getattr(airgap_session, method)(payload)

    before = len(invoked)
    live_session = build_session(provider="anthropic")
    live = getattr(live_session, method)(payload)

    assert airgap_session.airgap is True
    assert live_session.airgap is False, "live session fell back to air-gap"
    assert len(invoked) > before, (
        f"{method} never reached the model, so this proves nothing about parity"
    )
    return airgap, live


def _assert_fields_identical(airgap, live, fields):
    for field in fields:
        assert getattr(airgap, field) == getattr(live, field), (
            f"{field} diverged between air-gap and live"
        )


@pytest.mark.parametrize("user_query,expected_route", [
    ("why is this kernel slow?", "analysis"),
    ("analyze the trace", "analysis"),
    ("suggest optimizations", "recommendation"),
    ("did my patch help", "correctness"),
])
def test_root_routing_identical_under_hostile_model(
    user_query, expected_route, hostile_llm
):
    """Handoff target is a rule decision; a model asking for a different
    specialist does not get one."""
    payload = schemas.RootInput(user_query=user_query, database_path=None)
    airgap_out, live_out = _both_modes("run_root", payload, hostile_llm)

    assert airgap_out.metadata.get("routed_to") == expected_route
    assert live_out.metadata.get("routed_to") == expected_route


def test_classification_identical_under_hostile_model(memory_bound_db, hostile_llm):
    """Bottleneck classification decides which specialist runs. A model that
    claims a different bottleneck, or perfect confidence, changes neither."""
    payload = schemas.AnalysisInput(database_path=str(memory_bound_db))
    airgap_out, live_out = _both_modes("run_analysis", payload, hostile_llm)

    _assert_fields_identical(
        airgap_out,
        live_out,
        (
            "primary_bottleneck",
            "confidence",
            "time_breakdown",
            "hot_kernels",
            "counter_data_available",
        ),
    )
    assert live_out.primary_bottleneck != HOSTILE["primary_bottleneck"]


@pytest.mark.parametrize("method,payload_cls,kwargs", [
    ("run_compute_specialist", "ComputeSpecialistInput", {}),
    ("run_memory_specialist", "MemorySpecialistInput", {}),
    ("run_latency_specialist", "LatencySpecialistInput", {}),
])
def test_specialist_vetted_lane_identical_under_hostile_model(
    method, payload_cls, kwargs, hostile_llm
):
    """The vetted lane is catalog-ranked. A model cannot substitute its own
    techniques, raise the confidence, or attach citations to them."""
    payload = getattr(schemas, payload_cls)(
        gfx_id="gfx942", hot_kernels=[{"name": "[K1]", "pct": 0.4}], **kwargs
    )
    airgap_out, live_out = _both_modes(method, payload, hostile_llm)

    _assert_fields_identical(airgap_out, live_out, ("techniques", "confidence", "citations"))
    names = [t.get("name") for t in live_out.techniques]
    assert "invented_technique" not in names
    assert live_out.citations == airgap_out.citations


def test_recommendation_ranking_identical_under_hostile_model(
    memory_bound_db, hostile_llm
):
    """Ranked advice, the chosen specialist, and plateau detection are all
    rule outputs."""
    findings = build_session(airgap=True).run_analysis(
        schemas.AnalysisInput(database_path=str(memory_bound_db))
    )
    payload = schemas.RecommendationInput(findings=findings, gfx_id="gfx942")
    airgap_out, live_out = _both_modes("run_recommendation", payload, hostile_llm)

    _assert_fields_identical(
        airgap_out, live_out, ("recommendations", "specialist_used", "plateau_detected")
    )
    names = [r.get("name") for r in live_out.recommendations]
    assert "invented_recommendation" not in names


def test_exploratory_lane_is_the_only_permitted_divergence(
    memory_bound_db, hostile_llm, monkeypatch
):
    """Parity is a claim about the vetted lane. Proposals may appear live and
    never air-gapped — that is the point of the lane — so the invariant is
    that they arrive *beside* identical vetted output, not instead of it."""
    monkeypatch.setenv("PERFXPERT_AGENT_CREATIVITY", "exploratory")

    findings = build_session(airgap=True).run_analysis(
        schemas.AnalysisInput(database_path=str(memory_bound_db))
    )
    payload = schemas.RecommendationInput(findings=findings, gfx_id="gfx942")
    airgap_out, live_out = _both_modes("run_recommendation", payload, hostile_llm)

    assert airgap_out.recommendations == live_out.recommendations
    assert airgap_out.exploratory_proposals == []
    for proposal in live_out.exploratory_proposals:
        assert proposal.status == "exploratory"
        assert proposal.confidence <= 0.5
