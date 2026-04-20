"""Tests for perfxpert.agents.framework — the SDK facade."""

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
            return _FakeRunResult()

    def _fake_function_tool(fn, *, name_override, strict_mode):
        return {"name": name_override, "fn": fn}

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda: object())
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


# -- Phase 8 — non-openai providers route through LitellmModel (B1) --------


def _install_fake_sdk(monkeypatch, captured):
    """Helper: stub the openai-agents SDK symbols so no network I/O occurs."""
    from perfxpert.agents import framework

    class _FakeSdkAgent:
        def __init__(self, *, name, instructions, tools, model):
            captured["agent_name"] = name
            captured["instructions"] = instructions
            captured["tools"] = list(tools)
            captured["model"] = model

    class _FakeRunResult:
        def __init__(self):
            self.final_output = {"ok": True}
            self.new_items = []

    class _FakeRunner:
        @staticmethod
        def run_sync(*, starting_agent, input, max_turns, run_config):
            captured["input"] = input
            return _FakeRunResult()

    def _fake_function_tool(fn, *, name_override, strict_mode):
        return {"name": name_override, "fn": fn}

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda: object())
    monkeypatch.setattr(framework, "sdk_function_tool", _fake_function_tool)


def test_sdk_invoke_routes_anthropic_through_litellm_model(monkeypatch):
    """Phase 8 — provider=anthropic must construct a LitellmModel wrapping
    the provider-prefixed name (``anthropic/claude-sonnet-4-5``), NOT a
    plain model string. Without this wrapper the openai-agents SDK
    silently routes every request to OpenAI's endpoint and fails with
    "requested model does not exist".
    """
    from agents.extensions.models.litellm_model import LitellmModel

    from perfxpert.agents.framework import Agent, _sdk_invoke

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-phase8")
    monkeypatch.delenv("PERFXPERT_AGENTS_MODEL_ANTHROPIC", raising=False)
    monkeypatch.delenv("PERFXPERT_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_MODEL", raising=False)

    captured: Dict[str, Any] = {}  # noqa: F821 — Any re-exposed below
    from typing import Any, Dict  # noqa: F401,E402 — keep self-contained

    captured = {}
    _install_fake_sdk(monkeypatch, captured)

    agent = Agent(
        name="A", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )

    _sdk_invoke(agent, {"q": "?"}, provider="anthropic")

    assert isinstance(captured["model"], LitellmModel), (
        "anthropic provider must hand the SDK a LitellmModel, not a plain string — "
        "otherwise the SDK routes to OpenAI's endpoint and rejects Anthropic model names."
    )
    assert captured["model"].model == "anthropic/claude-sonnet-4-5"
    assert captured["model"].api_key == "sk-ant-fake-phase8"


def test_sdk_invoke_keeps_openai_as_plain_string(monkeypatch):
    """openai provider must keep the legacy plain-string model, so the
    SDK uses its native OpenAI client (retries / tracing / etc.)."""
    from perfxpert.agents.framework import Agent, _sdk_invoke

    monkeypatch.delenv("PERFXPERT_AGENTS_MODEL_OPENAI", raising=False)
    monkeypatch.delenv("PERFXPERT_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_MODEL", raising=False)

    captured = {}
    _install_fake_sdk(monkeypatch, captured)

    agent = Agent(
        name="O", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    _sdk_invoke(agent, {}, provider="openai")

    assert captured["model"] == "gpt-4o-mini"
    assert isinstance(captured["model"], str)


def test_sdk_invoke_claude_code_alias_routes_through_anthropic(monkeypatch):
    """Phase 8 — provider=claude-code is a credential alias for anthropic.

    It must construct a LitellmModel with ``anthropic/<model>`` so the
    SDK reaches Anthropic's endpoint, and pull the API key from
    ANTHROPIC_API_KEY (claude-agent-sdk does not expose a credential
    lookup helper for the ``claude`` CLI's stored OAuth token).
    """
    from agents.extensions.models.litellm_model import LitellmModel

    from perfxpert.agents.framework import Agent, _sdk_invoke

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-cc")
    monkeypatch.delenv("PERFXPERT_AGENTS_MODEL_ANTHROPIC", raising=False)
    monkeypatch.delenv("PERFXPERT_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_MODEL", raising=False)

    captured = {}
    _install_fake_sdk(monkeypatch, captured)

    agent = Agent(
        name="CC", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    _sdk_invoke(agent, {"q": "?"}, provider="claude-code")

    assert isinstance(captured["model"], LitellmModel)
    assert captured["model"].model.startswith("anthropic/")
    assert captured["model"].api_key == "sk-ant-fake-cc"


# -- Fix 1 — SDK error classification ---------------------------------------


def _install_fake_sdk_with_runner_error(monkeypatch, error_to_raise):
    """Stub the openai-agents SDK so Runner.run_sync raises ``error_to_raise``."""
    from perfxpert.agents import framework

    class _FakeSdkAgent:
        def __init__(self, *, name, instructions, tools, model):
            pass

    class _FakeRunner:
        @staticmethod
        def run_sync(*, starting_agent, input, max_turns, run_config):
            raise error_to_raise

    def _fake_function_tool(fn, *, name_override, strict_mode):
        return {"name": name_override, "fn": fn}

    monkeypatch.setattr(framework, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(framework, "SdkAgent", _FakeSdkAgent)
    monkeypatch.setattr(framework, "SdkRunner", _FakeRunner)
    monkeypatch.setattr(framework, "SdkRunConfig", lambda: object())
    monkeypatch.setattr(framework, "sdk_function_tool", _fake_function_tool)


def test_sdk_invoke_raises_quota_exceeded_on_429_insufficient_quota(monkeypatch):
    """A 429 with ``insufficient_quota`` substring must surface as
    ``QuotaExceededError`` (not a bare ``RuntimeError``), so the CLI
    boundary can render the clean "top up / switch provider" message."""
    from perfxpert.agents.framework import Agent, _sdk_invoke
    from perfxpert.providers._exceptions import QuotaExceededError

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    # Simulate what the OpenAI SDK raises on a 429 insufficient_quota:
    # the class name contains "ratelimit" and the message contains the
    # insufficient_quota substring.
    class _FakeRateLimit(Exception):
        pass
    _FakeRateLimit.__name__ = "RateLimitError"
    err = _FakeRateLimit(
        "Error code: 429 — You exceeded your current quota, please check your "
        "plan and billing details. code=insufficient_quota"
    )
    _install_fake_sdk_with_runner_error(monkeypatch, err)

    agent = Agent(
        name="Q", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    with pytest.raises(QuotaExceededError) as excinfo:
        _sdk_invoke(agent, {"q": "?"}, provider="openai")
    # Preserves the raw first-line message for user display.
    assert "quota" in str(excinfo.value).lower()
    assert excinfo.value.provider == "openai"


def test_sdk_invoke_raises_authentication_error_on_401(monkeypatch):
    """A 401 / invalid_api_key must surface as ``AuthError`` so the CLI
    boundary can tell the user which env var to fix."""
    from perfxpert.agents.framework import Agent, _sdk_invoke
    from perfxpert.providers._exceptions import AuthError

    monkeypatch.setenv("OPENAI_API_KEY", "sk-bogus")

    class _FakeAuth(Exception):
        pass
    _FakeAuth.__name__ = "AuthenticationError"
    err = _FakeAuth("Error code: 401 — Incorrect API key provided: sk-bogus. invalid_api_key")
    _install_fake_sdk_with_runner_error(monkeypatch, err)

    agent = Agent(
        name="A", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    with pytest.raises(AuthError) as excinfo:
        _sdk_invoke(agent, {"q": "?"}, provider="openai")
    assert excinfo.value.provider == "openai"
    assert "auth" in str(excinfo.value).lower()


def test_sdk_invoke_raises_rate_limit_on_bare_429(monkeypatch):
    """A plain 429 ``rate_limit_exceeded`` (no quota substring) must surface
    as ``RateLimitError``, not ``QuotaExceededError`` — users should retry
    or use the fallback chain, not panic about their credit balance."""
    from perfxpert.agents.framework import Agent, _sdk_invoke
    from perfxpert.providers._exceptions import QuotaExceededError, RateLimitError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")

    class _FakeRateLimit(Exception):
        pass
    _FakeRateLimit.__name__ = "RateLimitError"
    # Classic provider-throttle response: 429 with rate_limit_exceeded and
    # explicitly NO quota vocabulary.
    err = _FakeRateLimit("Error code: 429 — rate_limit_exceeded: too many requests, retry in 60s")
    _install_fake_sdk_with_runner_error(monkeypatch, err)

    agent = Agent(
        name="R", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    with pytest.raises(RateLimitError) as excinfo:
        _sdk_invoke(agent, {"q": "?"}, provider="anthropic")
    assert not isinstance(excinfo.value, QuotaExceededError)
    assert excinfo.value.provider == "anthropic"


def test_sdk_invoke_raises_transient_on_5xx(monkeypatch):
    """A 503 / connection error / timeout must surface as ``TransientError``
    so the CLI prompts the user to retry rather than panic."""
    from perfxpert.agents.framework import Agent, _sdk_invoke
    from perfxpert.providers._exceptions import TransientError

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    err = Exception("Error code: 503 — service unavailable")
    _install_fake_sdk_with_runner_error(monkeypatch, err)

    agent = Agent(
        name="T", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    with pytest.raises(TransientError) as excinfo:
        _sdk_invoke(agent, {"q": "?"}, provider="openai")
    assert excinfo.value.provider == "openai"


def test_sdk_invoke_anthropic_does_not_double_prefix(monkeypatch):
    """If the user pins an already-prefixed model via PERFXPERT_ANTHROPIC_MODEL,
    we must not produce ``anthropic/anthropic/…`` nonsense."""
    from perfxpert.agents.framework import Agent, _sdk_invoke

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("PERFXPERT_ANTHROPIC_MODEL", "anthropic/claude-opus-4")
    monkeypatch.delenv("PERFXPERT_AGENTS_MODEL_ANTHROPIC", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_MODEL", raising=False)

    captured = {}
    _install_fake_sdk(monkeypatch, captured)

    agent = Agent(
        name="A2", layer=1, fence_path=None, input_schema=dict, output_schema=dict, tools=[]
    )
    _sdk_invoke(agent, {}, provider="anthropic")
    assert captured["model"].model == "anthropic/claude-opus-4"
