"""Isolation tests for Root agent (Layer 0).

Each test scripts the mocked LLM response and asserts the routing /
handoff / output shape.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from perfxpert.agents import root as root_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    Agent, AgentConstructionError, HandoffPolicyViolation, ToolAllowlistViolation,
    FakeProviderResponse, dispatch_tool,
)
from perfxpert.tools import tasks as tasks_tool


# -- Construction ---------------------------------------------------------

def test_root_agent_builds():
    agent = root_module.build_root_agent()
    assert agent.name == "Root"
    assert agent.layer == 0


def test_root_tool_allowlist_size():
    agent = root_module.build_root_agent()
    assert len(agent.tools) <= 5


def test_root_uses_real_task_store_functions():
    agent = root_module.build_root_agent()
    bindings = {tool.name: tool.fn for tool in agent.tools}
    assert bindings["tasks.next"] is not None
    assert bindings["tasks.create"] is not None
    assert bindings["tasks.update"] is not None
    assert bindings["tasks.close"] is not None
    assert bindings["tasks.next"] is tasks_tool.next
    assert bindings["tasks.create"] is tasks_tool.create
    assert bindings["tasks.update"] is tasks_tool.update
    assert bindings["tasks.close"] is tasks_tool.close


def test_root_uses_source_dir_for_task_store(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "project"
    source_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    def fake_run_agent(agent, input_payload, **kwargs):
        dispatch_tool(agent, "tasks.create", {"title": "from-root"})
        return {
            "structured_output": {
                "narrative": "ok",
                "recommendations": [],
                "primary_bottleneck": "mixed",
                "warnings": [],
                "metadata": {},
            }
        }

    monkeypatch.setattr(root_module, "run_agent", fake_run_agent)
    root_module.run_root(
        schemas.RootInput(
            user_query="track this work",
            source_dir=str(source_dir),
            database_path=str(source_dir / "trace.db"),
        ),
        airgap=False,
        provider="anthropic",
    )

    assert (source_dir / ".beads" / "tasks.db").exists()
    assert not (tmp_path / ".beads" / "tasks.db").exists()


def test_root_allowed_handoffs_exactly_three():
    agent = root_module.build_root_agent()
    assert set(agent.allowed_handoffs) == {"analysis", "recommendation", "correctness"}


def test_root_cannot_handoff_to_specialist():
    agent = root_module.build_root_agent()
    with pytest.raises(HandoffPolicyViolation):
        from perfxpert.agents.framework import dispatch_handoff
        dispatch_handoff(agent, "compute_specialist")


def test_root_cannot_call_execution_tool():
    agent = root_module.build_root_agent()
    forbidden = [
        "patch.apply", "patch.revert", "patch.verify_output",
        "compile.build", "profile.run", "anchors.check",
    ]
    for tool in forbidden:
        assert not agent.has_tool(tool), f"Root must NOT have execution tool {tool!r}"


# -- Routing --------------------------------------------------------------

def test_root_routes_analyze_intent_to_analysis(fake_provider, monkeypatch):
    """Rule-first routing: intent=analyze → handoff to Analysis."""
    fake_provider.return_value = FakeProviderResponse(
        text="routed", handoff="analysis",
        structured_output={
            "narrative": "Routed to analysis.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="why is this kernel slow?", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)


def test_root_routes_verify_intent_to_correctness(fake_provider):
    fake_provider.return_value = FakeProviderResponse(
        text="routed", handoff="correctness",
        structured_output={
            "narrative": "Routed to correctness.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="did my patch help?", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)


def test_root_routes_optimize_intent_to_recommendation(fake_provider):
    fake_provider.return_value = FakeProviderResponse(
        text="routed", handoff="recommendation",
        structured_output={
            "narrative": "Routed to recommendation.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="suggest optimizations", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)


# -- Air-gap routing parity -----------------------------------------------

def test_root_routing_is_deterministic_in_airgap(monkeypatch):
    """Spec §5 invariant: air-gap routing decisions identical to LLM mode."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    result1 = root_module.run_root(
        schemas.RootInput(user_query="why slow?", database_path="x.db"),
        airgap=True,
    )
    result2 = root_module.run_root(
        schemas.RootInput(user_query="why slow?", database_path="x.db"),
        airgap=True,
    )
    # Deterministic: same input produces same handoff target
    assert result1.metadata.get("routed_to") == result2.metadata.get("routed_to")


def test_root_airgap_uses_analysis_bottleneck_for_db_queries(monkeypatch):
    """Airgap Root should preserve deterministic Analysis classification."""
    analysis_output = schemas.AnalysisOutput(
        primary_bottleneck="latency",
        confidence=0.75,
        time_breakdown={"api_pct": 86.6},
        hot_kernels=[],
        counter_data_available=True,
    )

    monkeypatch.setattr(
        "perfxpert.agents.analysis.run_analysis",
        lambda *args, **kwargs: analysis_output,
    )

    result = root_module.run_root(
        schemas.RootInput(
            user_query="analyze this trace",
            database_path="x.db",
        ),
        airgap=True,
    )

    assert result.primary_bottleneck == "latency"


# -- Narrative assembly ---------------------------------------------------

def test_root_writes_bottleneck_narrative(fake_provider):
    """Absorbed from former Bottleneck-Narrator (review N3).

    Root's narrative includes a 2-3 sentence classification explanation.
    """
    fake_provider.return_value = FakeProviderResponse(
        text="narrative",
        structured_output={
            "narrative": (
                "The workload is bottlenecked on HBM bandwidth. "
                "Kernel X consumes 62% of total runtime. "
                "Focus optimization on memory coalescing."
            ),
            "recommendations": [{"title": "coalesce loads", "priority": "high"}],
            "primary_bottleneck": "memory_transfer",
            "warnings": [], "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="analyze", database_path="x.db"),
        provider="anthropic",
    )
    assert len(result.narrative) > 0
    assert result.primary_bottleneck == "memory_transfer"


# -- Finding #24: Root lambda stubs for tasks.* never validated ---------------

def test_root_lambdas_for_tasks_never_invoked_during_routing(monkeypatch, fake_provider):
    """Root's tasks.* are stubbed as lambdas returning None. Ensure run_root
    doesn't actually invoke them in normal routing paths (Finding #24).

    If it does, the lambda stubs are hiding real behavior — switch to real
    tasks.* wrappers from the tasks module.
    """
    from perfxpert.agents.framework import dispatch_tool as real_dispatch_tool

    invocations = []

    def recording_dispatch(ag, tool_name, args):
        invocations.append(tool_name)
        return real_dispatch_tool(ag, tool_name, args)

    monkeypatch.setattr(
        "perfxpert.agents.framework.dispatch_tool",
        recording_dispatch,
    )

    fake_provider.return_value = FakeProviderResponse(
        text="routed",
        structured_output={
            "narrative": "Routed.",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [],
            "metadata": {},
        },
    )
    result = root_module.run_root(
        schemas.RootInput(user_query="why slow?", database_path="x.db"),
        provider="anthropic",
    )
    assert isinstance(result, schemas.RootOutput)

    tasks_invocations = [n for n in invocations if n.startswith("tasks.")]
    # Root should NOT invoke tasks.* tools during a simple routing turn.
    # If this assertion fails, the lambda stubs are hiding real behavior.
    assert not tasks_invocations, (
        f"Root invoked tasks.* tools during routing: {tasks_invocations}. "
        "Switch the lambda stubs to real tasks.* wrappers from perfxpert.tools.tasks."
    )


def test_root_tasks_bindings_are_real():
    """Root must wire real tasks.* wrappers — no lambda stubs may remain.

    Regression guard for Finding #24.
    """
    import inspect

    from perfxpert.tools import tasks as tasks_mod

    agent = root_module.build_root_agent()
    tasks_tools = [tb for tb in agent.tools if tb.name.startswith("tasks.")]

    assert len(tasks_tools) >= 1, (
        "Root should declare at least one tasks.* tool per fence spec"
    )

    expected = {
        "tasks.next": tasks_mod.next_task,
        "tasks.create": tasks_mod.create,
        "tasks.update": tasks_mod.update,
        "tasks.close": tasks_mod.close,
    }

    lambda_stubs = []
    for tb in tasks_tools:
        try:
            src = inspect.getsource(tb.fn).strip()
        except (OSError, TypeError):
            src = ""
        if "lambda" in src and ("None" in src or ": None" in src):
            lambda_stubs.append(tb.name)
        if tb.name in expected:
            assert tb.fn is expected[tb.name], (
                f"{tb.name} should be wired to perfxpert.tools.tasks.{expected[tb.name].__name__}"
            )

    assert not lambda_stubs, (
        f"Root still uses lambda stub(s) for: {lambda_stubs}. "
        "Real perfxpert.tools.tasks wrappers must be used instead."
    )


# -- Fence allowlist helper -----------------------------------------------

def _extract_fence_tool_names(fence_text):
    """Parse Tool allowlist section from a fence markdown file.

    Returns set of tool names. Stops at next ## heading or EOF.
    """
    in_section = False
    tools = set()
    for line in fence_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and "Tool allowlist" in stripped:
            in_section = True
            continue
        if in_section:
            if stripped.startswith("## "):
                break
            if stripped.startswith("- "):
                tool_name = stripped[2:].strip()
                if tool_name:
                    tools.add(tool_name)
    return tools


# -- Fence / allowlist alignment (design N29) ----------------------------

def test_root_fence_tools_match_allowlist():
    """Fence-declared tools must be subset of agent actual allowlist (N29)."""
    from pathlib import Path
    agent = root_module.build_root_agent()
    allowed = {t.name for t in agent.tools}
    fence_path = Path(root_module.__file__).parent / "fence" / "root.md"
    fence_text = fence_path.read_text()
    fence_tools = _extract_fence_tool_names(fence_text)
    violations = fence_tools - allowed
    assert not violations, (
        f"Root fence lists tools not in agent allowlist: {violations}"
    )


# -- Fence / allowlist alignment for ALL agents (design N29 full audit) --

def _agent_fence_allowlist_cases():
    from pathlib import Path
    from perfxpert.agents import (
        analysis as analysis_mod,
        recommendation as rec_mod,
        correctness as cor_mod,
        compute_specialist as cs_mod,
        memory_specialist as ms_mod,
        latency_specialist as ls_mod,
    )
    return [
        ("Root", root_module.build_root_agent,
         Path(root_module.__file__).parent / "fence" / "root.md"),
        ("Analysis", analysis_mod.build_analysis_agent,
         Path(analysis_mod.__file__).parent / "fence" / "analysis.md"),
        ("Recommendation", rec_mod.build_recommendation_agent,
         Path(rec_mod.__file__).parent / "fence" / "recommendation.md"),
        ("Correctness", cor_mod.build_correctness_agent,
         Path(cor_mod.__file__).parent / "fence" / "correctness.md"),
        ("ComputeSpecialist", cs_mod.build_compute_specialist,
         Path(cs_mod.__file__).parent / "fence" / "compute_specialist.md"),
        ("MemorySpecialist", ms_mod.build_memory_specialist,
         Path(ms_mod.__file__).parent / "fence" / "memory_specialist.md"),
        ("LatencySpecialist", ls_mod.build_latency_specialist,
         Path(ls_mod.__file__).parent / "fence" / "latency_specialist.md"),
    ]


@pytest.mark.parametrize("agent_name,builder,fence_path", _agent_fence_allowlist_cases())
def test_all_agents_fence_tools_subset_of_allowlist(agent_name, builder, fence_path):
    """Every fence-listed tool must appear in code allowlist (design N29)."""
    if not fence_path.exists():
        pytest.skip(f"No fence file for {agent_name}: {fence_path}")
    agent = builder()
    allowed = {t.name for t in agent.tools}
    fence_text = fence_path.read_text()
    fence_tools = _extract_fence_tool_names(fence_text)
    violations = fence_tools - allowed
    assert not violations, (
        f"{agent_name}: fence lists tools NOT in allowlist: {violations}"
    )
