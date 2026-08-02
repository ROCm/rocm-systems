"""Two-lane behaviour for the Layer-2 specialists (RFC 0001, Phase 11C).

The vetted lane is deterministic and the model cannot reach it. The
exploratory lane is where a model contributes, and everything in it is
labelled unproven and bound to evidence from the run that produced it.
"""

import pytest

from perfxpert.agents import compute_specialist as cs_module
from perfxpert.agents import latency_specialist as ls_module
from perfxpert.agents import memory_specialist as ms_module
from perfxpert.agents import recommendation as rec_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import FakeProviderResponse

CATALOG = [
    {"name": "coalesce_loads", "expected_impact": 0.5, "effort_factor": 1.0, "risk": "low"},
    {"name": "use_lds_tiling", "expected_impact": 0.3, "effort_factor": 1.0, "risk": "low"},
]


@pytest.fixture
def exploratory(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AGENT_CREATIVITY", "exploratory")


@pytest.fixture
def memory_catalog(monkeypatch):
    monkeypatch.setattr(ms_module, "_fetch_catalog", lambda gfx_id: list(CATALOG))


def _memory_input():
    return schemas.MemorySpecialistInput(
        gfx_id="gfx942", hot_kernels=[{"name": "[K1]", "pct": 0.4}]
    )


def _draft(**overrides):
    payload = {
        "title": "Prefetch the paged region",
        "hypothesis": "Page faults dominate this kernel.",
        "mechanism": "Prefetch before the hot loop.",
        "target_kernel": "[K1]",
        "evidence": [
            {
                "kind": "tool",
                "ref": "unified_memory.analyze_paging",
                "observation": "paging_events non-zero",
            }
        ],
        "confidence": 0.4,
    }
    payload.update(overrides)
    return payload


# -- The vetted lane is frozen -------------------------------------------


def test_model_cannot_replace_memory_techniques(fake_provider, memory_catalog):
    """A model used to be able to swap the catalog ranking wholesale."""
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "techniques": [{"name": "invented_technique", "expected_impact": 0.99}],
            "confidence": 0.99,
            "citations": ["fabricated source"],
        },
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    names = [t["name"] for t in result.techniques]
    assert "invented_technique" not in names
    assert names == ["coalesce_loads", "use_lds_tiling"]
    assert result.confidence == 0.6
    assert result.citations == []


def test_model_cannot_reorder_the_vetted_lane(fake_provider, memory_catalog):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "techniques": [
                {"name": "use_lds_tiling", "expected_impact": 0.9},
                {"name": "coalesce_loads", "expected_impact": 0.1},
            ],
        },
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert [t["name"] for t in result.techniques] == ["coalesce_loads", "use_lds_tiling"]


def test_live_and_airgap_vetted_lanes_are_identical(fake_provider, memory_catalog):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"techniques": [{"name": "invented"}], "confidence": 0.95},
    )
    live = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    airgapped = ms_module.run_memory_specialist(_memory_input(), airgap=True)
    assert live.techniques == airgapped.techniques
    assert live.confidence == airgapped.confidence


@pytest.mark.parametrize(
    "module,runner,payload",
    [
        (
            cs_module,
            "run_compute_specialist",
            schemas.ComputeSpecialistInput(gfx_id="gfx942", hot_kernels=[{"name": "[K1]", "pct": 0.4}]),
        ),
        (
            ls_module,
            "run_latency_specialist",
            schemas.LatencySpecialistInput(gfx_id="gfx942", hot_kernels=[{"name": "[K1]", "pct": 0.4}]),
        ),
    ],
)
def test_other_specialists_also_freeze_the_vetted_lane(
    fake_provider, monkeypatch, module, runner, payload
):
    monkeypatch.setattr(module, "_fetch_catalog", lambda gfx_id: list(CATALOG))
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "techniques": [{"name": "invented_technique"}],
            "confidence": 0.99,
        },
    )
    result = getattr(module, runner)(payload, provider="anthropic")
    assert "invented_technique" not in [t["name"] for t in result.techniques]
    assert result.confidence == 0.6


# -- The exploratory lane is off by default ------------------------------


def test_no_proposals_without_the_configured_ceiling(fake_provider, memory_catalog):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft()]},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []


def test_no_proposals_in_airgap_even_when_enabled(exploratory, memory_catalog):
    result = ms_module.run_memory_specialist(_memory_input(), airgap=True)
    assert result.exploratory_proposals == []


# -- The exploratory lane when enabled -----------------------------------


def test_proposal_is_accepted_when_evidence_checks_out(
    exploratory, fake_provider, memory_catalog
):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft()]},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert len(result.exploratory_proposals) == 1
    proposal = result.exploratory_proposals[0]
    assert proposal.status == "exploratory"
    assert proposal.specialist == "memory"
    assert proposal.proposal_id.startswith("pxp-exp-")
    assert proposal.provenance.provider == "anthropic"


def test_over_confident_draft_is_dropped_not_quietly_clamped(
    exploratory, fake_provider, memory_catalog
):
    """The ceiling is refusal, not correction.

    Clamping 0.95 down to 0.5 would keep a proposal whose author misjudged how
    strong its evidence was, and present it at the same confidence as one that
    was honest about it. Nothing downstream could tell the two apart.
    """
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft(confidence=0.95)]},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []


def test_proposal_citing_an_uncalled_tool_is_dropped(
    exploratory, fake_provider, memory_catalog
):
    """Evidence must come from this run, not from the model's imagination."""
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft()]},
        tool_calls=[],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []


def test_proposal_naming_an_unmeasured_kernel_is_dropped(
    exploratory, fake_provider, memory_catalog
):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "exploratory_proposals": [_draft(target_kernel="[NEVER_PROFILED]")]
        },
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []


def test_sanitized_sdk_tool_names_still_match_the_manifest(
    exploratory, fake_provider, memory_catalog
):
    """The SDK reports unified_memory_analyze_paging, not the dotted name."""
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft()]},
        tool_calls=[{"name": "unified_memory_analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert len(result.exploratory_proposals) == 1


def test_proposals_never_leak_into_the_vetted_lane(
    exploratory, fake_provider, memory_catalog
):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft()]},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert len(result.exploratory_proposals) == 1
    assert [t["name"] for t in result.techniques] == ["coalesce_loads", "use_lds_tiling"]


def test_malformed_drafts_do_not_break_the_run(
    exploratory, fake_provider, memory_catalog
):
    """A bad proposal must not cost the user their vetted recommendations."""
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [{"garbage": True}]},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []
    assert len(result.techniques) == 2


def test_non_list_proposal_field_is_ignored(exploratory, fake_provider, memory_catalog):
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": "not a list"},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []


def test_output_validation_still_rejects_unknown_keys(
    exploratory, fake_provider, memory_catalog
):
    """Relaxing the draft field must not relax the rest of the schema."""
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"exploratory_proposals": [_draft()], "smuggled": "x"},
        tool_calls=[{"name": "unified_memory.analyze_paging", "arguments": {}}],
    )
    result = ms_module.run_memory_specialist(_memory_input(), provider="anthropic")
    assert result.exploratory_proposals == []


# -- Recommendation propagation ------------------------------------------


def _recommendation_input(bottleneck="memory_transfer"):
    return schemas.RecommendationInput(
        findings=schemas.AnalysisOutput(
            primary_bottleneck=bottleneck,
            confidence=0.8,
            time_breakdown={"kernel_pct": 0.5, "memcpy_pct": 0.4, "api_pct": 0.1, "idle_pct": 0.0},
            hot_kernels=[{"name": "[K1]", "pct": 0.4}],
            counter_data_available=True,
        ),
        gfx_id="gfx942",
    )


def test_recommendation_carries_proposals_through(monkeypatch):
    proposal = schemas.ExploratoryProposal(
        proposal_id="pxp-exp-abc", specialist="memory", title="t",
        hypothesis="h", mechanism="m", confidence=0.3,
    )
    monkeypatch.setattr(
        rec_module, "_run_specialist_memory",
        lambda payload, **kw: schemas.MemorySpecialistOutput(
            techniques=[{"name": "coalesce_loads"}],
            confidence=0.6,
            exploratory_proposals=[proposal],
        ),
    )
    result = rec_module.run_recommendation(_recommendation_input(), airgap=True)
    assert [p.proposal_id for p in result.exploratory_proposals] == ["pxp-exp-abc"]
    assert [r["name"] for r in result.recommendations] == ["coalesce_loads"]


def test_proposals_are_not_suppressed_by_seen_recommendation_hashes(monkeypatch):
    """The lanes hold different evidence, so neither may hide the other."""
    proposal = schemas.ExploratoryProposal(
        proposal_id="pxp-exp-abc", specialist="memory", title="coalesce_loads",
        hypothesis="h", mechanism="m", confidence=0.3,
    )
    monkeypatch.setattr(
        rec_module, "_run_specialist_memory",
        lambda payload, **kw: schemas.MemorySpecialistOutput(
            techniques=[{"name": "coalesce_loads"}],
            confidence=0.6,
            exploratory_proposals=[proposal],
        ),
    )
    payload = schemas.RecommendationInput(
        findings=_recommendation_input().findings,
        gfx_id="gfx942",
        seen_recommendation_hashes=[rec_module._hash_technique({"name": "coalesce_loads"})],
    )
    result = rec_module.run_recommendation(payload, airgap=True)
    assert result.recommendations == []
    assert len(result.exploratory_proposals) == 1


def test_data_insufficient_yields_no_proposals(monkeypatch, capsys):
    result = rec_module.run_recommendation(
        _recommendation_input(bottleneck="data_insufficient"), airgap=True
    )
    assert result.recommendations == []
    assert result.exploratory_proposals == []
