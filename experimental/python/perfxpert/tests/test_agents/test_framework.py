"""Tests for perfxpert.agents.framework — the SDK facade."""

import inspect
from functools import partial
from types import SimpleNamespace

import pytest

from perfxpert.agents import framework
from perfxpert.agents.framework import (
    Agent,
    AgentConstructionError,
    Handoff,
    ToolBinding,
    run_agent,
)

# -- AgentSpec / Agent construction ----------------------------------------


def test_agent_construction_enforces_tool_cap():
    """Spec §2: ≤ 5 tools per agent."""
    too_many_tools = [ToolBinding(name=f"tool_{i}", fn=lambda x: x) for i in range(6)]
    with pytest.raises(AgentConstructionError, match="tool"):
        Agent(
            name="Overloaded",
            layer=1,
            fence_path="does-not-matter.md",
            input_schema=dict,
            output_schema=dict,
            tools=too_many_tools,
        )


def test_agent_construction_accepts_5_tools():
    tools = [ToolBinding(name=f"tool_{i}", fn=lambda x: x) for i in range(5)]
    a = Agent(
        name="MaxTools",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=tools,
    )
    assert len(a.tools) == 5


def test_agent_normalizes_mutable_collections_to_immutable_sequences():
    tools = [ToolBinding(name="tool_0", fn=lambda x: x)]
    handoffs = ["analysis"]
    agent = Agent(
        name="Immutable",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=tools,
        allowed_handoffs=handoffs,
    )

    tools.append(ToolBinding(name="tool_1", fn=lambda x: x))
    handoffs.append("recommendation")

    assert agent.tools == (ToolBinding(name="tool_0", fn=tools[0].fn),)
    assert agent.allowed_handoffs == ("analysis",)


def test_agent_rejects_post_construction_mutation():
    agent = Agent(
        name="Frozen",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    with pytest.raises((AttributeError, TypeError)):
        agent.tools += (ToolBinding(name="tool_0", fn=lambda x: x),)


def test_agent_builder_outputs_immutable_sequences():
    from perfxpert.agents import analysis as analysis_module

    agent = analysis_module.build_analysis_agent()
    assert isinstance(agent.tools, tuple)
    assert isinstance(agent.allowed_handoffs, tuple)


def test_agent_construction_enforces_fence_line_cap(tmp_path):
    big_fence = tmp_path / "big.md"
    big_fence.write_text("\n".join(f"line {i}" for i in range(401)))
    with pytest.raises(AgentConstructionError, match="fence"):
        Agent(
            name="Bloated",
            layer=1,
            fence_path=str(big_fence),
            input_schema=dict,
            output_schema=dict,
            tools=[],
        )


def test_agent_construction_accepts_400_line_fence(tmp_path):
    fence = tmp_path / "ok.md"
    fence.write_text("\n".join(f"line {i}" for i in range(400)))
    a = Agent(
        name="OK",
        layer=1,
        fence_path=str(fence),
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )
    assert a.fence_line_count == 400


# -- Handoff whitelist -----------------------------------------------------


def test_handoff_rejects_layer2_to_layer2():
    """Spec §2 rule: no Layer-2 → Layer-2 handoffs."""
    with pytest.raises(AgentConstructionError, match="layer"):
        Handoff(
            source_layer=2,
            target_layer=2,
            source_name="compute_specialist",
            target_name="memory_specialist",
        )


def test_handoff_allows_root_to_layer1():
    h = Handoff(source_layer=0, target_layer=1, source_name="root", target_name="analysis")
    assert h.target_name == "analysis"


def test_handoff_allows_layer1_to_layer2_from_recommendation():
    h = Handoff(
        source_layer=1,
        target_layer=2,
        source_name="recommendation",
        target_name="compute_specialist",
    )
    assert h.source_name == "recommendation"


def test_handoff_rejects_upward():
    with pytest.raises(AgentConstructionError, match="downward"):
        Handoff(source_layer=2, target_layer=1, source_name="compute_specialist", target_name="recommendation")


def test_handoff_rejects_skip_root_to_layer2():
    with pytest.raises(AgentConstructionError, match="skip"):
        Handoff(source_layer=0, target_layer=2, source_name="root", target_name="compute_specialist")


# -- Tool dispatch guard ---------------------------------------------------


def test_tool_dispatch_blocks_out_of_allowlist(fake_provider):
    """Agent cannot call a tool not in its allowlist."""
    allowed = ToolBinding(name="analysis.time_breakdown", fn=lambda db: {})
    agent = Agent(
        name="Test",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[allowed],
    )
    # Simulate SDK producing a tool call the agent isn't allowed to make
    with pytest.raises(framework.ToolAllowlistViolation):
        framework.dispatch_tool(agent, "profile.run", {"cmd": "rm -rf /"})


# -- Airgap fallback -------------------------------------------------------


def test_run_agent_airgap_uses_template(monkeypatch, tmp_path):
    """With PERFXPERT_AIRGAP=1, no SDK call is made; templates drive output."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")

    fence = tmp_path / "x.md"
    fence.write_text("short fence")
    agent = Agent(
        name="T",
        layer=1,
        fence_path=str(fence),
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    # If run_agent tried to call the SDK we'd get an AttributeError; with
    # airgap the facade must bypass it entirely.
    result = run_agent(agent, input_payload={"user_query": "why slow?"}, airgap=True)
    assert result is not None
    assert "airgap" in result.get("_mode", "").lower() or result.get("airgap") is True


# -- Provider selection pass-through --------------------------------------


def test_run_agent_passes_provider_to_sdk(fake_provider):
    from perfxpert.agents.framework import FakeProviderResponse  # type: ignore

    fence = None
    agent = Agent(
        name="P",
        layer=1,
        fence_path=fence,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )
    fake_provider.return_value = FakeProviderResponse(text="ok", structured_output={"x": 1})

    run_agent(agent, input_payload={"user_query": "?"}, provider="anthropic")

    # Assert the facade forwarded "anthropic" to the SDK call
    called_args = fake_provider.call_args
    assert "anthropic" in str(called_args)


# -- Finding #12: Agent layer validation and frozen enforcement ------------


def test_agent_rejects_invalid_layer():
    """Agent with layer=3 must raise AgentConstructionError."""
    from perfxpert.agents.framework import AgentConstructionError

    with pytest.raises(AgentConstructionError, match="layer=3"):
        Agent(
            name="Bad",
            layer=3,
            fence_path=None,
            input_schema=dict,
            output_schema=dict,
        )


def test_agent_rejects_negative_layer():
    """Agent with layer=-1 must raise AgentConstructionError."""
    from perfxpert.agents.framework import AgentConstructionError

    with pytest.raises(AgentConstructionError, match="layer=-1"):
        Agent(
            name="Bad",
            layer=-1,
            fence_path=None,
            input_schema=dict,
            output_schema=dict,
        )


def test_agent_accepts_all_valid_layers():
    """Layers 0, 1, 2 must all construct without error."""
    for layer in (0, 1, 2):
        agent = Agent(
            name=f"Layer{layer}",
            layer=layer,
            fence_path=None,
            input_schema=dict,
            output_schema=dict,
        )
        assert agent.layer == layer


def test_agent_is_frozen():
    """Agent must be frozen — attribute assignment must raise FrozenInstanceError."""
    import dataclasses

    agent = Agent(
        name="Frozen",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        agent.name = "mutated"


# -- _sdk_invoke live-path wiring (B1) -------------------------------------


def test_sdk_invoke_is_not_unconditionally_stub():
    """Regression for review blocker B1: _sdk_invoke must not raise
    NotImplementedError for the live path. Tests that monkeypatch
    _sdk_invoke are unaffected (they replace the symbol)."""
    import inspect
    from perfxpert.agents import framework

    src = inspect.getsource(framework._sdk_invoke)
    assert "NotImplementedError" not in src, (
        "Live SDK path must not raise NotImplementedError — this gates "
        "tests/test_integration/test_llm_end_to_end.py from ever running."
    )


def test_sdk_invoke_wires_openai_agents_sdk(monkeypatch):
    """Build an Agent with a tool whose name contains a dot; assert the
    wiring calls the SDK Runner.run_sync with a sanitized tool name and
    the selected model, then coerces the result into FakeProviderResponse.
    """
    from perfxpert.agents import framework

    # Stub the SDK Agent, Runner, function_tool so we don't hit the network.
    captured = {}

    class _FakeSdkAgent:
        def __init__(self, *, name, instructions, tools, model):
            captured["agent_name"] = name
            captured["instructions"] = instructions
            captured["tools"] = list(tools)
            captured["model"] = model

    class _FakeRunResult:
        def __init__(self):
            self.final_output = {"narrative": "hello", "recommendations": []}
            self.new_items = []

    class _FakeRunner:
        @staticmethod
        def run_sync(*, starting_agent, input, max_turns, run_config):
            captured["input"] = input
            captured["max_turns"] = max_turns
            captured["run_config"] = run_config
            return _FakeRunResult()

    def _fake_function_tool(fn, *, name_override, strict_mode):
        return {"name": name_override, "fn": fn}

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda **kwargs: {"kwargs": kwargs})
    monkeypatch.setattr(framework, "sdk_function_tool", _fake_function_tool)

    from perfxpert.agents.framework import Agent, ToolBinding, _sdk_invoke, FakeProviderResponse

    def _noop(**kwargs):
        return None

    agent = Agent(
        name="T",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[ToolBinding(name="intent.classify", fn=_noop)],
    )

    resp = _sdk_invoke(agent, {"user_query": "?"}, provider="openai")

    assert isinstance(resp, FakeProviderResponse)
    assert resp.structured_output == {"narrative": "hello", "recommendations": []}
    # The SDK receives a sanitized tool name (dots → underscores)
    assert captured["tools"] == [{"name": "intent_classify", "fn": _noop}]
    # Default max_turns=10 when PERFXPERT_AGENTS_MAX_TURNS unset
    assert captured["max_turns"] == 10
    # Model resolved from _DEFAULT_MODELS["openai"]
    assert captured["model"] == "gpt-4o-mini"
    assert captured["run_config"] == {"kwargs": {}}


def test_translate_tools_accepts_partial_callables(monkeypatch):
    captured = {}

    def _fake_function_tool(fn, *, name_override, strict_mode):
        captured["callable_name"] = fn.__name__
        return {"name": name_override, "fn": fn}

    def _create_at(root, title):
        return f"{root}:{title}"

    monkeypatch.setattr(framework, "sdk_function_tool", _fake_function_tool)

    tool = ToolBinding(name="tasks.create", fn=partial(_create_at, "demo-app"))
    wrapped = framework._translate_tools([tool])

    assert captured["callable_name"] == "tasks_create"
    assert wrapped[0]["name"] == "tasks_create"
    assert wrapped[0]["fn"]("check") == "demo-app:check"
    assert str(inspect.signature(wrapped[0]["fn"])) == "(title)"


def test_sdk_invoke_raises_runtime_error_when_sdk_missing(monkeypatch):
    """When openai-agents is not installed, _sdk_invoke must raise RuntimeError
    with an actionable message — NOT NotImplementedError."""
    from perfxpert.agents import framework

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", False)
    monkeypatch.setattr(framework, "SdkAgent", None)
    monkeypatch.setattr(framework, "SdkRunner", None)
    monkeypatch.setattr(framework, "SdkRunConfig", None)

    from perfxpert.agents.framework import Agent, _sdk_invoke

    agent = Agent(
        name="T", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    with pytest.raises(RuntimeError, match="openai-agents"):
        _sdk_invoke(agent, "x", provider="openai")


def test_sdk_invoke_preserves_provider_taxonomy(monkeypatch):
    """ProviderError subclasses from the live SDK path must not be flattened."""
    from perfxpert.agents import framework
    from perfxpert.providers._exceptions import AuthError

    class _FakeSdkAgent:
        def __init__(self, **_kwargs):
            pass

    class _FakeRunner:
        @staticmethod
        def run_sync(**_kwargs):
            raise AuthError("openai", "bad key")

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda **kwargs: {"kwargs": kwargs})
    monkeypatch.setattr(framework, "sdk_function_tool", lambda fn, **kwargs: {"fn": fn, **kwargs})

    agent = Agent(
        name="T",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    with pytest.raises(AuthError):
        framework._sdk_invoke(agent, {"user_query": "?"}, provider="openai")


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "litellm/anthropic/claude-sonnet-4-20250514"),
        ("ollama", "litellm/ollama/llama3.1"),
        ("private", "private/gpt-4o-mini"),
    ],
)
def test_resolve_model_qualifies_non_openai_providers(provider, expected, monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_PRIVATE_MODEL", raising=False)
    monkeypatch.delenv(f"PERFXPERT_AGENTS_MODEL_{provider.upper()}", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_MODEL", raising=False)
    assert framework._resolve_model(provider) == expected


def test_resolve_model_preserves_explicit_prefixed_override(monkeypatch):
    monkeypatch.setenv(
        "PERFXPERT_AGENTS_MODEL_ANTHROPIC",
        "litellm/anthropic/claude-3-7-sonnet-latest",
    )
    assert framework._resolve_model("anthropic") == "litellm/anthropic/claude-3-7-sonnet-latest"


def test_resolve_model_uses_private_specific_model_env(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "gpt-5.3-codex")
    assert framework._resolve_model("private") == "private/gpt-5.3-codex"


def test_sdk_invoke_builds_litellm_route_for_anthropic(monkeypatch):
    captured = {}

    class _FakeSdkAgent:
        def __init__(self, *, name, instructions, tools, model):
            captured["model"] = model

    class _FakeRunResult:
        def __init__(self):
            self.final_output = {"narrative": "ok"}
            self.new_items = []

    class _FakeRunner:
        @staticmethod
        def run_sync(*, starting_agent, input, max_turns, run_config):
            captured["run_config"] = run_config
            return _FakeRunResult()

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda **kwargs: {"kwargs": kwargs})
    monkeypatch.setattr(framework, "sdk_function_tool", lambda fn, **kwargs: {"fn": fn, **kwargs})

    agent = Agent(
        name="T",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    framework._sdk_invoke(agent, {"user_query": "?"}, provider="anthropic")

    assert captured["model"] == "litellm/anthropic/claude-sonnet-4-20250514"
    assert "model_provider" in captured["run_config"]["kwargs"]


def test_sdk_invoke_private_provider_uses_custom_openai_base_url(monkeypatch):
    captured = {}

    class _FakeSdkAgent:
        def __init__(self, *, name, instructions, tools, model):
            captured["model"] = model

    class _FakeRunResult:
        def __init__(self):
            self.final_output = {"narrative": "ok"}
            self.new_items = []

    class _FakeRunner:
        @staticmethod
        def run_sync(*, starting_agent, input, max_turns, run_config):
            captured["run_config"] = run_config
            return _FakeRunResult()

    import agents.models.openai_provider as openai_provider_module
    import openai as openai_module

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

    class _FakeOpenAIProvider:
        def __init__(self, **kwargs):
            captured["provider_kwargs"] = kwargs

    import httpx as httpx_module
    monkeypatch.setattr(openai_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(openai_provider_module, "OpenAIProvider", _FakeOpenAIProvider)

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["http_client_kwargs"] = kwargs

    monkeypatch.setattr(httpx_module, "AsyncClient", _FakeAsyncClient)

    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.example/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "internal-xl")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_API_KEY", "sk-private")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_VERIFY_SSL", "false")
    monkeypatch.setenv(
        "PERFXPERT_LLM_PRIVATE_HEADERS",
        '{"Ocp-Apim-Subscription-Key": "secret", "api-version": "preview"}',
    )
    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda **kwargs: {"kwargs": kwargs})
    monkeypatch.setattr(framework, "sdk_function_tool", lambda fn, **kwargs: {"fn": fn, **kwargs})

    agent = Agent(
        name="T",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    framework._sdk_invoke(agent, {"user_query": "?"}, provider="private")

    assert captured["model"] == "private/internal-xl"
    assert captured["client_kwargs"]["api_key"] == "sk-private"
    assert captured["client_kwargs"]["base_url"] == "https://llm.example/v1"
    assert captured["client_kwargs"]["default_headers"] == {
        "Ocp-Apim-Subscription-Key": "secret",
        "api-version": "preview",
    }
    assert captured["client_kwargs"]["http_client"] is not None
    assert captured["http_client_kwargs"] == {"verify": False}
    assert "openai_client" in captured["provider_kwargs"]
    assert captured["provider_kwargs"]["use_responses"] is False
    assert "model_provider" in captured["run_config"]["kwargs"]


def test_sdk_invoke_opencode_dispatches_to_subprocess_provider(monkeypatch):
    captured = {}

    class _FakeOpencodeProvider:
        def complete(self, messages, *, system="", model=None, **_kwargs):
            captured["messages"] = messages
            captured["system"] = system
            captured["model"] = model
            return SimpleNamespace(
                content='{"narrative": "ok"}',
                provider="opencode",
                model=model or "opencode-default",
            )

    import perfxpert.providers.opencode_provider as opencode_provider_module

    monkeypatch.setenv("PERFXPERT_AGENTS_MODEL_OPENCODE", "github-copilot/gpt-5")
    monkeypatch.setattr(opencode_provider_module, "OpencodeProvider", _FakeOpencodeProvider)
    monkeypatch.setattr(framework, "_SDK_AVAILABLE", False)

    agent = Agent(
        name="T",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    response = framework._sdk_invoke(agent, {"user_query": "?"}, provider="opencode")

    assert captured["model"] == "github-copilot/gpt-5"
    assert captured["messages"][0]["role"] == "user"
    assert "user_query" in captured["messages"][0]["content"]
    assert "You are the T agent" in captured["system"]
    assert response.text == '{"narrative": "ok"}'
    assert response.structured_output == {"narrative": "ok"}


def test_sdk_invoke_maps_rate_limit_like_runtime_errors(monkeypatch):
    from perfxpert.providers._exceptions import RateLimitError

    class _FakeSdkAgent:
        def __init__(self, **_kwargs):
            pass

    class _FakeRunner:
        @staticmethod
        def run_sync(**_kwargs):
            raise RuntimeError("429 rate limit from upstream gateway")

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda **kwargs: {"kwargs": kwargs})
    monkeypatch.setattr(framework, "sdk_function_tool", lambda fn, **kwargs: {"fn": fn, **kwargs})

    agent = Agent(
        name="T",
        layer=1,
        fence_path=None,
        input_schema=dict,
        output_schema=dict,
        tools=[],
    )

    with pytest.raises(RateLimitError):
        framework._sdk_invoke(agent, {"user_query": "?"}, provider="openai")
