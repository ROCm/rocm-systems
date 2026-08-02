"""Tests for perfxpert.fence._builder."""

import pytest

from perfxpert.fence import FenceBuilder


VALID_ROLES = [
    "root",
    "analysis",
    "recommendation",
    "correctness",
    "compute_specialist",
    "memory_specialist",
    "latency_specialist",
    "diff_specialist",
]


def test_all_roles_build_nonempty():
    fb = FenceBuilder()
    for role in VALID_ROLES:
        text = fb.build(role)
        assert text, f"empty fence for role={role}"
        assert len(text) > 100, f"fence for {role} suspiciously short"


def test_always_slice_always_present():
    fb = FenceBuilder()
    for role in VALID_ROLES:
        text = fb.build(role)
        assert "# PerfXpert Always Fence" in text


def test_role_specific_section_present():
    fb = FenceBuilder()
    text = fb.build("compute_specialist")
    assert "compute" in text.lower()


def test_unknown_role_raises():
    fb = FenceBuilder()
    with pytest.raises(KeyError):
        fb.build("bogus_role")


def test_caches_identical_inputs():
    fb = FenceBuilder()
    a = fb.build("analysis")
    b = fb.build("analysis")
    assert a == b


def test_gfx_id_specialization_included():
    fb = FenceBuilder()
    text = fb.build("analysis", gfx_id="gfx942")
    assert "MI300X" in text or "gfx942" in text


def test_bottleneck_specialization_included():
    fb = FenceBuilder()
    text = fb.build("recommendation", bottleneck="memory_transfer")
    assert "memory" in text.lower()


# -- The builder is the live composition path (Phase 11A) ------------------


def test_live_agent_prompt_is_the_builder_output():
    """Agents used to read one role file directly, bypassing composition."""
    from perfxpert.agents import AGENT_BUILDERS
    from perfxpert.fence._builder import compose_prompt
    from pathlib import Path

    for build in AGENT_BUILDERS:
        agent = build()
        role = Path(agent.fence_path).stem
        assert agent.fence_text == compose_prompt(role), (
            f"{agent.name} prompt diverges from the composed fence for {role!r}"
        )


def test_shared_fence_reaches_every_live_agent_exactly_once():
    from perfxpert.agents import AGENT_BUILDERS

    marker = "# PerfXpert Always Fence"
    for build in AGENT_BUILDERS:
        agent = build()
        assert agent.fence_text.count(marker) == 1, (
            f"{agent.name} contains the shared fence "
            f"{agent.fence_text.count(marker)} times (want exactly 1)"
        )


def test_every_agent_role_is_known_to_the_builder():
    from pathlib import Path

    from perfxpert.agents import AGENT_BUILDERS
    from perfxpert.fence._builder import known_role

    for build in AGENT_BUILDERS:
        agent = build()
        role = Path(agent.fence_path).stem
        assert known_role(role), f"builder does not know role {role!r} ({agent.name})"


# -- Two-lane contract (Phase 11C) -----------------------------------------


def test_exploration_capable_agents_are_told_about_both_lanes():
    """An agent that can propose must be told what a proposal is."""
    from perfxpert.agents import AGENT_BUILDERS
    from perfxpert.agents.framework import AgentCapability

    for build in AGENT_BUILDERS:
        agent = build()
        if agent.capability is not AgentCapability.ADDITIVE_EXPLORATION:
            continue
        assert "## Two lanes" in agent.fence_text, agent.name
        assert "exploratory_proposals" in agent.fence_text, agent.name


def test_catalog_only_agents_are_not_invited_to_propose():
    """Otherwise a model would emit a lane the runtime discards."""
    from perfxpert.agents import AGENT_BUILDERS
    from perfxpert.agents.framework import AgentCapability

    for build in AGENT_BUILDERS:
        agent = build()
        if agent.capability is AgentCapability.CATALOG_ONLY:
            assert "## Two lanes" not in agent.fence_text, agent.name


def test_fence_states_the_confidence_ceiling_and_proposal_cap():
    from perfxpert.agents import build_memory_specialist

    text = build_memory_specialist().fence_text
    assert "0.5" in text
    assert "at most 3" in text


def test_fence_forbids_claiming_runtime_owned_fields():
    from perfxpert.agents import build_memory_specialist

    text = build_memory_specialist().fence_text
    for owned in ("proposal_id", "status", "specialist", "provenance"):
        assert owned in text, owned
