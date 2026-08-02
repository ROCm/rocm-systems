"""Tests for the exploratory lane (RFC 0001, Phase 11B).

The contract under test is that a model may propose but never promote: it
supplies a draft, and the runtime decides whether that draft becomes a
proposal, what identity it gets, and where it is recorded.
"""

import pytest
from pydantic import ValidationError

from perfxpert.agents import creativity, schemas
from perfxpert.agents.creativity import (
    CreativityTier,
    EvidenceManifest,
    ProposalRejected,
    build_proposal,
    build_proposals,
    compute_proposal_id,
    dedupe_proposals,
    resolve_tier,
    validate_draft,
)
from perfxpert.agents.framework import Agent, AgentCapability, AgentConstructionError


def _manifest(**kwargs):
    kwargs.setdefault("tool_calls", {"unified_memory.analyze_paging"})
    kwargs.setdefault("kernels", {"gemm_kernel"})
    return EvidenceManifest(**kwargs)


def _draft(**overrides):
    payload = {
        "title": "Prefetch the paged region",
        "hypothesis": "Page faults dominate this kernel's stall time.",
        "mechanism": "Issue an explicit prefetch before the hot loop.",
        "target_kernel": "gemm_kernel",
        "evidence": [
            {
                "kind": "tool",
                "ref": "unified_memory.analyze_paging",
                "observation": "paging_events is non-zero",
            }
        ],
        "expected_effects": [{"metric": "page_faults", "direction": "decrease"}],
        "confidence": 0.4,
    }
    payload.update(overrides)
    return schemas.ExploratoryProposalDraft(**payload)


# -- Tier resolution ------------------------------------------------------


@pytest.mark.parametrize(
    "configured,airgap,capability,expected",
    [
        (CreativityTier.STRICT, False, AgentCapability.ADDITIVE_EXPLORATION, CreativityTier.STRICT),
        (CreativityTier.STRICT, True, AgentCapability.ADDITIVE_EXPLORATION, CreativityTier.STRICT),
        (CreativityTier.EXPLORATORY, True, AgentCapability.ADDITIVE_EXPLORATION, CreativityTier.STRICT),
        (CreativityTier.EXPLORATORY, False, AgentCapability.CATALOG_ONLY, CreativityTier.STRICT),
        (CreativityTier.EXPLORATORY, False, AgentCapability.ADDITIVE_EXPLORATION, CreativityTier.EXPLORATORY),
    ],
)
def test_tier_lattice(configured, airgap, capability, expected):
    assert resolve_tier(configured, airgap=airgap, capability=capability) is expected


def test_airgap_always_stays_strict():
    """Decision parity between air-gap and live runs is an invariant."""
    for capability in AgentCapability:
        assert (
            resolve_tier(CreativityTier.EXPLORATORY, airgap=True, capability=capability)
            is CreativityTier.STRICT
        )


def test_default_configuration_is_strict():
    from perfxpert.config import PerfXpertConfig

    assert PerfXpertConfig().agent_creativity == "strict"


def test_unknown_creativity_value_fails_configuration():
    from perfxpert.config import PerfXpertConfig

    with pytest.raises(ValidationError):
        PerfXpertConfig(agent_creativity="unrestricted")


# -- Capability declaration ----------------------------------------------


def test_agents_default_to_catalog_only():
    agent = Agent(
        name="T", layer=2, fence_path=None, input_schema=dict, output_schema=dict
    )
    assert agent.capability is AgentCapability.CATALOG_ONLY


def test_layer2_may_declare_additive_exploration():
    agent = Agent(
        name="T", layer=2, fence_path=None, input_schema=dict, output_schema=dict,
        capability=AgentCapability.ADDITIVE_EXPLORATION,
    )
    assert agent.capability is AgentCapability.ADDITIVE_EXPLORATION


@pytest.mark.parametrize("layer", [0, 1])
def test_exploration_is_rejected_above_layer2(layer):
    """Root and the decision-makers route; an open channel there steers the run."""
    with pytest.raises(AgentConstructionError, match="additive_exploration"):
        Agent(
            name="T", layer=layer, fence_path=None, input_schema=dict,
            output_schema=dict, capability=AgentCapability.ADDITIVE_EXPLORATION,
        )


def test_only_layer2_specialists_opt_in():
    from perfxpert.agents import AGENT_BUILDERS

    for build in AGENT_BUILDERS:
        agent = build()
        if agent.capability is AgentCapability.ADDITIVE_EXPLORATION:
            assert agent.layer == 2, f"{agent.name} is layer {agent.layer}"


# -- Evidence binding -----------------------------------------------------


def test_proposal_must_cite_evidence():
    with pytest.raises(ProposalRejected, match="no evidence"):
        validate_draft(_draft(evidence=[]), manifest=_manifest())


def test_evidence_must_reference_a_tool_that_actually_ran():
    """The core anti-fabrication rule."""
    draft = _draft(
        evidence=[
            {"kind": "tool", "ref": "arch.lookup_peaks", "observation": "made up"}
        ]
    )
    with pytest.raises(ProposalRejected, match="not produced during this run"):
        validate_draft(draft, manifest=_manifest())


def test_target_kernel_must_have_been_measured():
    draft = _draft(target_kernel="kernel_that_does_not_exist")
    with pytest.raises(ProposalRejected, match="not among the measured kernels"):
        validate_draft(draft, manifest=_manifest())


def test_absent_target_kernel_is_allowed():
    validate_draft(_draft(target_kernel=None), manifest=_manifest())


def test_empty_manifest_rejects_everything():
    with pytest.raises(ProposalRejected):
        validate_draft(_draft(), manifest=EvidenceManifest())


def test_confidence_ceiling_is_enforced_by_the_schema():
    with pytest.raises(ValidationError):
        _draft(confidence=0.9)


def test_confidence_ceiling_is_also_enforced_at_validation():
    """Belt and braces: a draft built another way still cannot exceed it."""
    draft = _draft(confidence=0.5)
    object.__setattr__(draft, "__dict__", {**draft.__dict__, "confidence": 0.99})
    with pytest.raises(ProposalRejected, match="exceeds the exploratory ceiling"):
        validate_draft(draft, manifest=_manifest())


# -- Runtime-owned fields -------------------------------------------------


def test_draft_cannot_carry_runtime_owned_fields():
    """A model must not be able to assign identity, status or provenance."""
    for field in ("proposal_id", "status", "specialist", "provenance"):
        with pytest.raises(ValidationError):
            schemas.ExploratoryProposalDraft(
                title="t", hypothesis="h", mechanism="m", **{field: "x"}
            )


def test_runtime_stamps_identity_and_status():
    proposal = build_proposal(
        _draft(),
        specialist="memory",
        manifest=_manifest(),
        provenance=schemas.ProposalProvenance(provider="anthropic"),
    )
    assert proposal.proposal_id.startswith(creativity.PROPOSAL_ID_PREFIX)
    assert proposal.status == "exploratory"
    assert proposal.specialist == "memory"
    assert proposal.provenance.provider == "anthropic"


def test_proposal_id_is_content_addressed_and_stable():
    a = compute_proposal_id(_draft(), specialist="memory")
    b = compute_proposal_id(_draft(), specialist="memory")
    assert a == b


def test_proposal_id_changes_with_content():
    base = compute_proposal_id(_draft(), specialist="memory")
    assert compute_proposal_id(_draft(title="Something else"), specialist="memory") != base
    assert compute_proposal_id(_draft(), specialist="compute") != base


def test_status_cannot_be_relabelled_as_vetted():
    with pytest.raises(ValidationError):
        schemas.ExploratoryProposal(
            proposal_id="pxp-exp-1", status="recommended", specialist="memory",
            title="t", hypothesis="h", mechanism="m", confidence=0.3,
        )


def test_verification_cannot_opt_out_of_the_gates():
    with pytest.raises(ValidationError):
        schemas.VerificationPlan(requires_full_gate_cascade=False)


# -- Batch construction ---------------------------------------------------


def test_one_bad_draft_does_not_discard_the_others():
    good = _draft()
    bad = _draft(
        title="Fabricated",
        evidence=[{"kind": "tool", "ref": "never.called", "observation": "x"}],
    )
    accepted, rejected = build_proposals(
        [good, bad], specialist="memory", manifest=_manifest(),
        provenance=schemas.ProposalProvenance(),
    )
    assert [p.title for p in accepted] == ["Prefetch the paged region"]
    assert len(rejected) == 1
    assert "never.called" in rejected[0]


def test_proposal_count_is_capped():
    drafts = [_draft(title=f"Idea {i}") for i in range(6)]
    accepted, rejected = build_proposals(
        drafts, specialist="memory", manifest=_manifest(),
        provenance=schemas.ProposalProvenance(),
    )
    assert len(accepted) == creativity.MAX_PROPOSALS_PER_SPECIALIST
    assert any("over the" in r for r in rejected)


def test_malformed_draft_is_reported_not_raised():
    accepted, rejected = build_proposals(
        [{"title": "missing required fields"}], specialist="memory",
        manifest=_manifest(), provenance=schemas.ProposalProvenance(),
    )
    assert accepted == []
    assert "malformed draft" in rejected[0]


def test_identical_drafts_collapse_to_one_proposal():
    accepted, _ = build_proposals(
        [_draft(), _draft()], specialist="memory", manifest=_manifest(),
        provenance=schemas.ProposalProvenance(),
    )
    assert len(accepted) == 1


def test_dedupe_uses_server_id_only():
    """Title-based dedupe would let one specialist suppress another's proposal."""
    first = build_proposal(
        _draft(), specialist="memory", manifest=_manifest(),
        provenance=schemas.ProposalProvenance(),
    )
    same_title_other_content = build_proposal(
        _draft(hypothesis="A different hypothesis entirely."),
        specialist="memory", manifest=_manifest(),
        provenance=schemas.ProposalProvenance(),
    )
    kept = dedupe_proposals([first, same_title_other_content, first])
    assert len(kept) == 2


# -- Output contract ------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        schemas.ComputeSpecialistOutput,
        schemas.MemorySpecialistOutput,
        schemas.LatencySpecialistOutput,
    ],
)
def test_specialist_outputs_carry_an_empty_lane_by_default(model):
    out = model(techniques=[], confidence=0.5)
    assert out.exploratory_proposals == []


def test_recommendation_carries_an_empty_lane_by_default():
    out = schemas.RecommendationOutput(recommendations=[], specialist_used="none")
    assert out.exploratory_proposals == []


def test_diff_nests_the_lane_under_kernel_deltas():
    """Diff is at the 5-field cap, so the lane nests instead of adding a field."""
    out = schemas.DiffSpecialistOutput(
        wall_delta_pct=0.0, verdict="neutral", narrative="n", confidence=0.5,
    )
    assert "exploratory_proposals" not in out.kernel_deltas
    out2 = schemas.DiffSpecialistOutput(
        wall_delta_pct=0.0, verdict="neutral", narrative="n", confidence=0.5,
        kernel_deltas={"regressions": [], "improvements": [], "exploratory_proposals": []},
    )
    assert out2.kernel_deltas["exploratory_proposals"] == []


def test_diff_kernel_deltas_rejects_unknown_keys():
    """Nesting must not turn the field into an unvalidated grab bag."""
    with pytest.raises(ValidationError, match="unknown key"):
        schemas.DiffSpecialistOutput(
            wall_delta_pct=0.0, verdict="neutral", narrative="n", confidence=0.5,
            kernel_deltas={"regressions": [], "smuggled": []},
        )


def test_diff_validates_nested_proposal_items():
    with pytest.raises(ValidationError):
        schemas.DiffSpecialistOutput(
            wall_delta_pct=0.0, verdict="neutral", narrative="n", confidence=0.5,
            kernel_deltas={"exploratory_proposals": [{"not": "a proposal"}]},
        )


def test_output_field_caps_still_hold():
    """Adding the lane must not push any output past the 5-field cap."""
    for model in (
        schemas.ComputeSpecialistOutput,
        schemas.MemorySpecialistOutput,
        schemas.LatencySpecialistOutput,
        schemas.RecommendationOutput,
        schemas.DiffSpecialistOutput,
    ):
        assert len(model.model_fields) <= 5, model.__name__


# -- Session wiring -------------------------------------------------------


def test_session_defaults_to_strict(monkeypatch):
    from perfxpert.agents import runtime as runtime_module

    monkeypatch.delenv("PERFXPERT_AGENT_CREATIVITY", raising=False)
    session = runtime_module.build_session(airgap=True)
    assert session.creativity is CreativityTier.STRICT


def test_session_reads_the_configured_ceiling(monkeypatch):
    from perfxpert.agents import runtime as runtime_module

    monkeypatch.setenv("PERFXPERT_AGENT_CREATIVITY", "exploratory")
    session = runtime_module.build_session(provider="anthropic")
    assert session.creativity is CreativityTier.EXPLORATORY


def test_session_tier_for_agent_respects_capability(monkeypatch):
    from perfxpert.agents import runtime as runtime_module

    monkeypatch.setenv("PERFXPERT_AGENT_CREATIVITY", "exploratory")
    session = runtime_module.build_session(provider="anthropic")

    catalog_only = Agent(
        name="T", layer=2, fence_path=None, input_schema=dict, output_schema=dict
    )
    exploring = Agent(
        name="T", layer=2, fence_path=None, input_schema=dict, output_schema=dict,
        capability=AgentCapability.ADDITIVE_EXPLORATION,
    )
    assert session.tier_for(catalog_only) is CreativityTier.STRICT
    assert session.tier_for(exploring) is CreativityTier.EXPLORATORY


def test_airgap_session_stays_strict_even_when_configured_otherwise(monkeypatch):
    from perfxpert.agents import runtime as runtime_module

    monkeypatch.setenv("PERFXPERT_AGENT_CREATIVITY", "exploratory")
    session = runtime_module.build_session(airgap=True)
    exploring = Agent(
        name="T", layer=2, fence_path=None, input_schema=dict, output_schema=dict,
        capability=AgentCapability.ADDITIVE_EXPLORATION,
    )
    assert session.tier_for(exploring) is CreativityTier.STRICT
