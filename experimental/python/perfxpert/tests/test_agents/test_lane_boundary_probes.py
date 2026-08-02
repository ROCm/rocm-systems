"""Probes against the exploratory-lane boundary, from the attacker's side.

`test_specialist_lanes.py` checks the lane behaves correctly. This file asks
the complementary question: given a model that is trying to get its own
content treated as vetted advice, what does it actually have to work with?

The two surfaces worth probing are the evidence manifest (which decides
whether a proposal's citations are real) and the `model_supplies: draft`
relaxation in output validation, which is a deliberate hole in an otherwise
strict validator and therefore the most likely place for something to slip
through.
"""

import pytest

from perfxpert.agents import AGENT_BUILDERS, schemas
from perfxpert.agents import memory_specialist as ms
from perfxpert.agents.framework import Agent, validate_structured_output

CATALOG = [
    {"name": "coalesce_loads", "expected_impact": 0.5, "effort_factor": 1.0, "risk": "low"}
]

REAL_TOOL = "unified_memory.analyze_paging"

DRAFT = {
    "title": "Prefetch the paged region",
    "hypothesis": "Page migration stalls dominate [K1].",
    "mechanism": "Prefetch before the hot loop.",
    "target_kernel": "[K1]",
    "evidence": [{"kind": "tool", "ref": REAL_TOOL, "observation": "paging events"}],
    "confidence": 0.4,
}


@pytest.fixture
def exploratory(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AGENT_CREATIVITY", "exploratory")


@pytest.fixture
def specialist(monkeypatch):
    """Run the memory specialist with a scripted model response."""
    monkeypatch.setattr(ms, "_fetch_catalog", lambda gfx_id: list(CATALOG))

    def _run(structured_output, tool_calls=()):
        monkeypatch.setattr(
            ms,
            "run_agent",
            lambda *a, **kw: {
                "structured_output": structured_output,
                "tool_calls": list(tool_calls),
            },
        )
        return ms.run_memory_specialist(
            schemas.MemorySpecialistInput(
                gfx_id="gfx942", hot_kernels=[{"name": "[K1]", "pct": 0.4}]
            ),
            provider="anthropic",
        )

    return _run


# -- The manifest's trust boundary ----------------------------------------


def test_control_a_genuine_tool_call_does_produce_a_proposal(exploratory, specialist):
    """Control. Without this the rejection tests below could pass for the
    wrong reason — a lane that never emits anything rejects everything."""
    out = specialist({"exploratory_proposals": [DRAFT]}, tool_calls=[{"name": REAL_TOOL}])
    assert len(out.exploratory_proposals) == 1


def test_model_cannot_forge_tool_calls_inside_its_own_output(exploratory, specialist):
    """The manifest is built from `tool_calls`, which the framework scrapes
    from the SDK run record. A model emitting a same-named key inside its
    structured output must not reach that manifest — otherwise every evidence
    check is self-certified."""
    out = specialist(
        {"exploratory_proposals": [DRAFT], "tool_calls": [{"name": REAL_TOOL}]},
        tool_calls=[],
    )
    assert out.exploratory_proposals == []


def test_forged_tool_calls_key_is_rejected_as_an_unknown_field(exploratory):
    """And it does not even survive output validation."""
    agent = ms.build_memory_specialist()
    accepted, reason = validate_structured_output(agent, {"tool_calls": [{"name": REAL_TOOL}]})
    assert accepted is None
    assert "tool_calls" in (reason or "")


def test_sdk_sanitised_tool_name_still_matches(exploratory, specialist):
    """The SDK reports dots as underscores; that mapping is intended."""
    out = specialist(
        {"exploratory_proposals": [DRAFT]},
        tool_calls=[{"name": REAL_TOOL.replace(".", "_")}],
    )
    assert len(out.exploratory_proposals) == 1


@pytest.mark.parametrize("bogus", [
    "unified_memory.analyze_pagin",            # truncated
    "UNIFIED_MEMORY.ANALYZE_PAGING",           # case
    " unified_memory.analyze_paging",          # leading space
    "unified_memory.analyze_paging ",          # trailing space
    "unified_memory.analyze_paging\n",         # newline
    "unified\u200bmemory.analyze_paging",      # zero-width space
    "unified_memory..analyze_paging",          # doubled separator
    "",
])
def test_near_miss_tool_names_are_not_credited(exploratory, specialist, bogus):
    """Loose matching here would make the manifest trivially satisfiable."""
    out = specialist({"exploratory_proposals": [DRAFT]}, tool_calls=[{"name": bogus}])
    assert out.exploratory_proposals == []


def test_no_agent_declares_colliding_sanitised_tool_names():
    """The dot-to-underscore map would alias `a.b` onto a declared `a_b`,
    letting a call to one be credited as the other. No agent does this today;
    this fails if one ever starts."""
    collisions = []
    for builder in AGENT_BUILDERS:
        agent = builder()
        seen = {}
        for name in {tool.name for tool in agent.tools}:
            sanitised = name.replace(".", "_")
            if sanitised in seen:
                collisions.append((agent.name, seen[sanitised], name))
            seen[sanitised] = name
    assert collisions == []


def test_an_agent_cannot_be_built_with_colliding_sanitised_tool_names():
    """Checking today's agents does not stop tomorrow's from colliding.

    The collision is invisible at the call site -- both names look distinct in
    the source -- and it only shows up as a proposal being credited for a tool
    that was never called, so it has to fail at construction.
    """
    from perfxpert.agents.framework import AgentConstructionError, ToolBinding

    with pytest.raises(AgentConstructionError, match="sanitise"):
        Agent(
            name="Colliding",
            layer=2,
            fence_path=None,
            input_schema=dict,
            output_schema=dict,
            tools=[
                ToolBinding(name="unified_memory.analyze", fn=lambda: None),
                ToolBinding(name="unified_memory_analyze", fn=lambda: None),
            ],
        )


# -- The `model_supplies: draft` relaxation -------------------------------

JUNK = [
    {"__class__": "os.system", "title": "x"},
    {"title": "x" * 50000, "hypothesis": "h", "mechanism": "m"},
    "a bare string",
    12345,
    None,
    {"nested": {"a": {"b": {"c": list(range(50))}}}},
]


def test_relaxed_field_does_accept_arbitrary_content_at_validation():
    """Documents the hole honestly: this field is `Optional[Any]` in the
    validation mirror, so validation is not what defends it."""
    agent = ms.build_memory_specialist()
    accepted, reason = validate_structured_output(agent, {"exploratory_proposals": JUNK})
    assert accepted is not None, reason


def test_but_none_of_it_becomes_a_proposal(exploratory, specialist):
    """The defence is draft construction, which is strict."""
    out = specialist({"exploratory_proposals": JUNK}, tool_calls=[{"name": REAL_TOOL}])
    assert out.exploratory_proposals == []


def test_and_the_vetted_lane_is_untouched_by_it(exploratory, specialist):
    out = specialist({"exploratory_proposals": JUNK}, tool_calls=[{"name": REAL_TOOL}])
    assert [t["name"] for t in out.techniques] == ["coalesce_loads"]
    assert out.confidence == 0.6
    assert out.citations == []


@pytest.mark.parametrize("shape", [
    {"exploratory_proposals": "string"},
    {"exploratory_proposals": 42},
    {"exploratory_proposals": {"a": 1}},
    {"exploratory_proposals": [[]]},
    {"exploratory_proposals": [None]},
])
def test_malformed_lane_shapes_do_not_break_the_run(exploratory, specialist, shape):
    """The exploratory lane is additive, so a malformed one must never cost
    the user the vetted answer they actually asked for."""
    out = specialist(shape, tool_calls=[{"name": REAL_TOOL}])
    assert out.exploratory_proposals == []
    assert [t["name"] for t in out.techniques] == ["coalesce_loads"]


@pytest.mark.parametrize("owned", ["proposal_id", "status", "specialist", "provenance"])
def test_draft_claiming_a_runtime_owned_field_is_dropped(exploratory, specialist, owned):
    out = specialist(
        {"exploratory_proposals": [{**DRAFT, owned: "forged"}]},
        tool_calls=[{"name": REAL_TOOL}],
    )
    assert out.exploratory_proposals == []


# -- Root's own output fields ---------------------------------------------


# Schema-valid on purpose. An invalid field gets the whole response discarded
# by output validation, which would prove nothing about whether an *accepted*
# response can steer these fields.
HOSTILE_ROOT = {
    "narrative": "model prose",
    "recommendations": [{"name": "invented_by_the_model", "expected_impact": 0.99}],
    "primary_bottleneck": "compute",
    "warnings": [],
    "metadata": {},
}


@pytest.fixture
def hostile_root(monkeypatch):
    from perfxpert.agents.framework import FakeProviderResponse

    monkeypatch.setattr(
        "perfxpert.agents.framework._sdk_invoke",
        lambda agent, payload, provider: FakeProviderResponse(
            structured_output=dict(HOSTILE_ROOT)
        ),
    )


def _root_both_modes():
    from perfxpert.agents import build_session

    payload = schemas.RootInput(user_query="suggest optimizations", database_path=None)
    return (
        build_session(airgap=True).run_root(payload),
        build_session(provider="anthropic").run_root(payload),
    )


def test_root_routing_is_identical_in_both_modes(hostile_root):
    airgap_out, live_out = _root_both_modes()
    assert airgap_out.metadata.get("routed_to") == live_out.metadata.get("routed_to")


def test_root_classification_is_identical_in_both_modes(hostile_root):
    """Root is Layer 0 with catalog_only capability, so its bottleneck claim
    should be a rule output, not whatever the model asserted."""
    airgap_out, live_out = _root_both_modes()
    assert airgap_out.primary_bottleneck == live_out.primary_bottleneck


def test_root_does_not_publish_model_invented_recommendations(hostile_root):
    """`recommendations` is the vetted-lane field name and Root is the primary
    MCP entry point, so model content landing here reaches users unlabelled."""
    _, live_out = _root_both_modes()
    names = [r.get("name") for r in live_out.recommendations if isinstance(r, dict)]
    assert "invented_by_the_model" not in names


def test_caps_hold_under_a_flood_of_drafts(exploratory, specialist):
    flood = [{**DRAFT, "title": f"variant {i}"} for i in range(50)]
    out = specialist({"exploratory_proposals": flood}, tool_calls=[{"name": REAL_TOOL}])

    assert len(out.exploratory_proposals) <= 3
    for proposal in out.exploratory_proposals:
        assert proposal.confidence <= 0.5
        assert proposal.status == "exploratory"
        assert proposal.specialist == "memory"
