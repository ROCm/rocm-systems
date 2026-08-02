"""Framework facade over OpenAI Agents SDK.

Design-review N5: agents depend on this facade, NOT on the SDK directly.
This is the only module allowed to `import openai_agents` (enforced by
tests/test_agents/test_no_sdk_import_leak.py — CI).

Under PERFXPERT_AIRGAP=1 the facade short-circuits SDK calls and drives
responses from agents/templates/airgap_report.txt, preserving the air-gap
parity invariant (spec §5 — gate decisions identical, narrative differs).

Construction-time guardrails:
- ≤ 5 tools per agent (spec §2)
- ≤ 400 fence lines per agent (spec §2)
- Handoff whitelist (Root→L1, Recommendation→L2 only; no L2→L2; no upward)

Runtime guardrails:
- Tool dispatch rejects out-of-allowlist calls
- Token budget per agent (provided by agent definitions)
- Structured output validated against declared Pydantic schema
- Session turn budget (carried by RunPolicy, owned by agents.runtime)
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from perfxpert.providers._exceptions import (
    AuthError,
    FatalError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError,
    TransientError,
)

# SDK import is lazy + isolated to this file — never in agent modules.
#
# The live path uses the openai-agents package (imported as `agents`). We
# try both the legacy `openai_agents` alias and the canonical `agents`
# module so downstream code can detect availability regardless of the
# install variant.
try:  # pragma: no cover - exercised only when the SDK is installed
    from agents import (  # type: ignore[import-not-found]
        Agent as SdkAgent,
        Runner as SdkRunner,
        RunConfig as SdkRunConfig,
        function_tool as sdk_function_tool,
    )

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - branch only when SDK absent
    SdkAgent = None  # type: ignore[assignment]
    SdkRunner = None  # type: ignore[assignment]
    SdkRunConfig = None  # type: ignore[assignment]
    sdk_function_tool = None  # type: ignore[assignment]
    _SDK_AVAILABLE = False


_LOG = logging.getLogger(__name__)


# Per-provider default models. Can be overridden via PERFXPERT_LLM_MODEL env
# or the PERFXPERT_AGENTS_MODEL_<PROVIDER> env.
_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.1",
    "private": "gpt-4o-mini",
    "opencode": "gpt-4o-mini",
}


# -- Exceptions -----------------------------------------------------------


class AgentConstructionError(ValueError):
    """Raised when an agent / handoff violates a design-time constraint."""


class ToolAllowlistViolation(RuntimeError):
    """Raised at runtime when an agent attempts to call a tool outside its allowlist."""


class HandoffPolicyViolation(RuntimeError):
    """Raised when an agent attempts a handoff not on its whitelist."""


class SessionTurnBudgetExceeded(RuntimeError):
    """Raised when a session exhausts MAX_SESSION_LLM_TURNS."""


# -- Capabilities ---------------------------------------------------------


class AgentCapability(str, Enum):
    """What an agent may do with a model-generated open-set channel.

    Lives here rather than in ``creativity`` so ``Agent`` can validate it
    without the framework depending on the creativity module.
    """

    CATALOG_ONLY = "catalog_only"
    ADDITIVE_EXPLORATION = "additive_exploration"


# -- Run policy -----------------------------------------------------------


@dataclass
class RunPolicy:
    """Per-session limits carried across ``run_agent()`` calls.

    ``agents.runtime`` owns one of these per :class:`AnalysisSession`. The
    per-run turn cap (``PERFXPERT_AGENTS_MAX_TURNS``) bounds a single SDK
    conversation; this bounds the whole session, which is the limit
    ``gate_cascade.MAX_SESSION_LLM_TURNS`` always declared but never applied.
    """

    max_session_turns: int = 100
    turns_used: int = 0

    def consume_turn(self, agent_name: str) -> None:
        if self.turns_used >= self.max_session_turns:
            raise SessionTurnBudgetExceeded(
                f"session exhausted its {self.max_session_turns}-turn LLM budget "
                f"while dispatching {agent_name}"
            )
        self.turns_used += 1


# The session's policy travels in a context variable rather than through every
# run_* signature: agents are plain functions called from several entry points
# (batch CLI, library API, MCP server) and threading a parameter through all of
# them would put session bookkeeping in each agent's contract.
_ACTIVE_POLICY: ContextVar[Optional[RunPolicy]] = ContextVar(
    "perfxpert_run_policy", default=None
)


@contextlib.contextmanager
def active_policy(policy: Optional[RunPolicy]) -> Iterator[None]:
    """Apply ``policy`` to every ``run_agent()`` call in this context."""
    token = _ACTIVE_POLICY.set(policy)
    try:
        yield
    finally:
        _ACTIVE_POLICY.reset(token)


# -- Fence composition ----------------------------------------------------


def _compose_fence_text(fence_path: str) -> str:
    """Return the effective prompt text for ``fence_path``.

    A recognised role gets the shared ``always.md`` policy composed in, via
    the single builder path. Anything else (test fixtures pointing at a
    temporary file) is read verbatim, so ad-hoc fences behave as before.
    """
    role = Path(fence_path).stem
    try:
        from perfxpert.fence._builder import compose_prompt, known_role

        if known_role(role):
            return compose_prompt(role)
    except ImportError:  # pragma: no cover - fence package always ships
        _LOG.warning("framework: fence builder unavailable; using role file verbatim")
    return Path(fence_path).read_text()


# -- Data classes ---------------------------------------------------------


@dataclass(frozen=True)
class ToolBinding:
    """A tool registered with an agent."""

    name: str
    fn: Callable[..., Any]


@dataclass(frozen=True)
class FakeProviderResponse:
    """What _sdk_invoke returns under test mocks."""

    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    structured_output: Optional[Dict[str, Any]] = None
    handoff: Optional[str] = None


@dataclass(frozen=True)
class Agent:
    """An LLM-backed reasoning unit with a fence slice and tool allowlist.

    Construction validates design-time constraints (spec §2):
    - layer in (0, 1, 2)
    - len(tools) <= 5
    - fence line count <= 400
    - additive_exploration only on Layer 2
    """

    name: str
    layer: int                              # 0=Root, 1=DecisionMaker, 2=Specialist
    fence_path: Optional[str]               # None = no fence file (test/placeholder)
    input_schema: Type                      # Pydantic model (or dict for tests)
    output_schema: Type                     # Pydantic model (or dict for tests)
    tools: Tuple[ToolBinding, ...] = field(default_factory=tuple)
    allowed_handoffs: Tuple[str, ...] = field(default_factory=tuple)
    token_budget: int = 4096
    # Default is the restrictive value: an agent gains an open-set channel
    # only by asking for one.
    capability: AgentCapability = AgentCapability.CATALOG_ONLY
    fence_text: str = field(init=False, default="")
    fence_line_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        tools = tuple(self.tools)
        allowed_handoffs = tuple(self.allowed_handoffs)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "allowed_handoffs", allowed_handoffs)
        object.__setattr__(self, "capability", AgentCapability(self.capability))

        if self.layer not in (0, 1, 2):
            raise AgentConstructionError(
                f"Agent {self.name}: layer={self.layer} — must be 0 (Root), "
                "1 (DecisionMaker), or 2 (Specialist)"
            )

        if len(tools) > 5:
            raise AgentConstructionError(f"Agent {self.name}: {len(tools)} tools declared (cap is 5)")

        # The SDK reports tool calls with dots rewritten to underscores, and
        # the evidence manifest maps those back by sanitised name. Two tools
        # that sanitise alike would make that mapping ambiguous, so a proposal
        # could cite one while the other was the tool actually called.
        by_sanitized: Dict[str, str] = {}
        for tool in tools:
            key = tool.name.replace(".", "_")
            if key in by_sanitized and by_sanitized[key] != tool.name:
                raise AgentConstructionError(
                    f"Agent {self.name}: {by_sanitized[key]!r} and {tool.name!r} both "
                    f"sanitise to {key!r}, so tool calls between them are "
                    "indistinguishable in the SDK run record"
                )
            by_sanitized[key] = tool.name

        if self.capability is AgentCapability.ADDITIVE_EXPLORATION and self.layer != 2:
            raise AgentConstructionError(
                f"Agent {self.name}: additive_exploration requires layer=2, got "
                f"layer={self.layer} — Root and the decision-makers route and "
                "classify, so an open-set channel there would steer the run"
            )

        if self.fence_path is not None:
            text = _compose_fence_text(self.fence_path)
            n = text.count("\n") + 1
            if n > 400:
                raise AgentConstructionError(f"Agent {self.name}: fence has {n} lines (cap is 400)")
            object.__setattr__(self, "fence_text", text)
            object.__setattr__(self, "fence_line_count", n)

    def has_tool(self, tool_name: str) -> bool:
        return any(t.name == tool_name for t in self.tools)


@dataclass(frozen=True)
class Handoff:
    """A typed transfer of control from one agent to another.

    Rules enforced at construction (spec §2):
    - downward only: source_layer < target_layer
    - no skipping: target_layer == source_layer + 1
    - no Layer-2 → Layer-2
    """

    source_layer: int
    target_layer: int
    source_name: str
    target_name: str

    def __post_init__(self) -> None:
        if self.source_layer == 2 and self.target_layer == 2:
            raise AgentConstructionError("No layer-2 → layer-2 handoffs (spec §2)")
        if self.target_layer <= self.source_layer:
            raise AgentConstructionError(
                f"Handoffs must be downward (source {self.source_layer} → target {self.target_layer})"
            )
        if self.target_layer != self.source_layer + 1:
            raise AgentConstructionError(f"Cannot skip layers ({self.source_layer} → {self.target_layer})")


# -- Output validation ----------------------------------------------------
#
# Agents merge model output over a deterministic core, so a response that
# omits a field is normal and must stay usable. A response carrying unknown
# keys or wrong types is not: that is either a confused model or an injection
# attempt, and trusting it would let unvalidated content reach the caller.
#
# So the declared output schema is mirrored into an all-optional model that
# keeps ``extra="forbid"`` and the per-field types. Missing fields pass;
# unknown or mistyped fields fail and the whole response is discarded, which
# drops the agent onto its deterministic fallback.


@lru_cache(maxsize=32)
def _partial_model_for(schema: Type) -> Optional[Type[BaseModel]]:
    """Build the all-optional mirror of ``schema``, or None if not a model."""
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return None
    fields: Dict[str, Any] = {}
    for name, info in schema.model_fields.items():
        extra = info.json_schema_extra
        if isinstance(extra, dict) and extra.get("model_supplies") == "draft":
            # The declared type is what the runtime constructs, not what the
            # model sends. Checking the model's draft against it would reject
            # every well-formed response; the draft is validated separately.
            fields[name] = (Optional[Any], None)
        else:
            fields[name] = (Optional[info.annotation], None)
    return create_model(
        f"Partial{schema.__name__}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def validate_structured_output(
    agent: "Agent", structured_output: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return ``(accepted_output, rejection_reason)`` for a model response.

    ``accepted_output`` is None when the response is unusable; callers then
    fall back to their deterministic path.
    """
    if structured_output is None:
        return None, None
    if not isinstance(structured_output, dict):
        return None, f"expected a JSON object, got {type(structured_output).__name__}"

    partial = _partial_model_for(agent.output_schema)
    if partial is None:
        # Test doubles may declare a plain dict schema; nothing to check.
        return structured_output, None

    try:
        partial(**structured_output)
    except ValidationError as exc:
        return None, str(exc)
    return structured_output, None


# -- SDK abstraction ------------------------------------------------------


def _resolve_model(provider: str) -> str:
    """Pick the model string to hand to the openai-agents SDK for ``provider``.

    Precedence:
      1. ``PERFXPERT_AGENTS_MODEL_<PROVIDER>`` (e.g. ``..._OPENAI``)
      2. ``PERFXPERT_LLM_PRIVATE_MODEL`` for the private provider
      3. ``PERFXPERT_LLM_MODEL`` (cross-provider override)
      4. Built-in default from :data:`_DEFAULT_MODELS`
    """
    specific = os.environ.get(f"PERFXPERT_AGENTS_MODEL_{provider.upper()}")
    provider_specific = (
        os.environ.get("PERFXPERT_LLM_PRIVATE_MODEL")
        if provider == "private"
        else None
    )
    generic = os.environ.get("PERFXPERT_LLM_MODEL")
    model = (
        specific
        or provider_specific
        or generic
        or _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["openai"])
    )
    if "/" in model:
        return model
    if provider == "anthropic":
        return f"litellm/anthropic/{model}"
    if provider == "ollama":
        return f"litellm/ollama/{model}"
    if provider == "private":
        return f"private/{model}"
    return model


def _build_model_settings(agent: "Agent") -> Any:
    """Translate ``agent.token_budget`` into SDK ModelSettings.

    Returns None when the SDK is too old to expose ModelSettings, so an
    unsupported install degrades to the previous unbounded behaviour rather
    than failing the run.
    """
    try:
        from agents.model_settings import ModelSettings  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - depends on SDK version
        try:
            from agents import ModelSettings  # type: ignore[import-not-found]
        except ImportError:
            _LOG.debug("framework: SDK exposes no ModelSettings; token budget not applied")
            return None
    try:
        return ModelSettings(max_tokens=agent.token_budget)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("framework: could not apply token budget (%s)", exc)
        return None


def _build_sdk_run_config(provider: str) -> Any:
    """Build the SDK RunConfig with provider-specific routing when needed."""
    kwargs: Dict[str, Any] = {}

    if provider in {"anthropic", "ollama", "private"}:
        from agents.models.multi_provider import MultiProvider, MultiProviderMap  # type: ignore[import-not-found]

        provider_map = MultiProviderMap()

        if provider in {"anthropic", "ollama"}:
            try:
                from agents.extensions.models.litellm_provider import LitellmProvider  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "framework: LiteLLM provider support missing; install perfxpert[litellm] "
                    "or perfxpert[all]"
                ) from exc
            provider_map.add_provider(provider, LitellmProvider())
        elif provider == "private":
            from agents.models.openai_provider import OpenAIProvider  # type: ignore[import-not-found]
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
            import httpx

            from perfxpert.providers.private_provider import (
                _parse_headers,
                _private_url_from_env,
                _verify_ssl_from_env,
            )

            base_url = _private_url_from_env()
            if not base_url:
                raise AuthError(
                    "private",
                    "no endpoint configured (set PERFXPERT_LLM_PRIVATE_URL or PRIVATE_LLM_ENDPOINT)",
                )
            try:
                extra_headers = _parse_headers(os.environ.get("PERFXPERT_LLM_PRIVATE_HEADERS", ""))
            except ValueError as exc:
                raise AuthError("private", str(exc)) from exc
            api_key = os.environ.get("PERFXPERT_LLM_PRIVATE_API_KEY") or "dummy"
            openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=extra_headers or None,
                http_client=httpx.AsyncClient(verify=_verify_ssl_from_env()),
            )
            provider_map.add_provider(
                "private",
                OpenAIProvider(
                    openai_client=openai_client,
                    use_responses=False,
                ),
            )

        kwargs["model_provider"] = MultiProvider(provider_map=provider_map)

    return SdkRunConfig(**kwargs)


def _map_exception_by_message(provider: str, exc: BaseException) -> ProviderError | None:
    """Best-effort string fallback when the SDK wraps or erases concrete types."""
    msg = str(exc)
    low = msg.lower()
    if "insufficient_quota" in low or ("quota" in low and "exceed" in low):
        return QuotaExceededError(provider, message=msg)
    if "rate limit" in low or "429" in low or "too many requests" in low:
        return RateLimitError(provider, message=msg)
    if (
        "invalid_api_key" in low
        or "authentication" in low
        or "unauthorized" in low
        or "permission denied" in low
        or "401" in low
        or "403" in low
    ):
        return AuthError(provider, msg)
    if "timeout" in low or "timed out" in low:
        return TimeoutError(provider, 0.0, msg)
    if (
        "connection" in low
        or "temporarily unavailable" in low
        or "try again later" in low
        or "503" in low
        or "504" in low
        or "502" in low
    ):
        return TransientError(provider, kind="transport", message=msg)
    return None


def _normalize_provider_exception(provider: str, exc: BaseException) -> ProviderError | None:
    """Map SDK/backend exceptions into the shared provider taxonomy."""
    if isinstance(exc, ProviderError):
        return exc

    try:
        import openai as _openai_sdk  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional runtime dep
        _openai_sdk = None  # type: ignore[assignment]

    try:
        import anthropic as _anthropic_sdk  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional runtime dep
        _anthropic_sdk = None  # type: ignore[assignment]

    text = str(exc)
    if _openai_sdk is not None:
        if isinstance(exc, (_openai_sdk.AuthenticationError, _openai_sdk.PermissionDeniedError)):
            return AuthError(provider, text)
        if isinstance(exc, _openai_sdk.RateLimitError):
            if "quota" in text.lower():
                return QuotaExceededError(provider, message=text)
            retry = getattr(exc, "retry_after", 0.0) or 0.0
            return RateLimitError(provider, retry_after=retry, message=text)
        if isinstance(exc, _openai_sdk.APITimeoutError):
            return TimeoutError(provider, 0.0, text)
        if isinstance(exc, (_openai_sdk.APIConnectionError, _openai_sdk.InternalServerError)):
            return TransientError(provider, kind="sdk", message=text)
        if isinstance(exc, _openai_sdk.BadRequestError):
            return FatalError(provider, text)

    if _anthropic_sdk is not None:
        if isinstance(exc, (_anthropic_sdk.AuthenticationError, _anthropic_sdk.PermissionDeniedError)):
            return AuthError(provider, text)
        if isinstance(exc, _anthropic_sdk.RateLimitError):
            if "quota" in text.lower():
                return QuotaExceededError(provider, message=text)
            retry = getattr(exc, "retry_after", 0.0) or 0.0
            return RateLimitError(provider, retry_after=retry, message=text)
        if isinstance(exc, _anthropic_sdk.APITimeoutError):
            return TimeoutError(provider, 0.0, text)
        if isinstance(exc, (_anthropic_sdk.APIConnectionError, _anthropic_sdk.InternalServerError)):
            return TransientError(provider, kind="sdk", message=text)

    return _map_exception_by_message(provider, exc)


def _serialize_input(input_payload: Any) -> str:
    """Coerce an arbitrary payload into the string the SDK Runner expects."""
    if isinstance(input_payload, str):
        return input_payload
    try:
        return json.dumps(input_payload, default=str)
    except (TypeError, ValueError):
        return str(input_payload)


def _sanitize_tool_name(name: str) -> str:
    """Translate a perfxpert tool name (``intent.classify``) into the
    OpenAI-compatible pattern ``^[a-zA-Z0-9_-]+$`` by swapping ``.`` for ``_``.
    """
    return name.replace(".", "_")


def _prepare_tool_callable_for_sdk(agent: "Agent", tb: ToolBinding) -> Callable[..., Any]:
    """Wrap a binding so every SDK invocation re-checks the allowlist.

    The SDK is only handed the agent's declared tools, so this is a second
    line of defence rather than the primary one. It matters because it puts
    :func:`dispatch_tool`'s policy on the live path instead of leaving it as
    a helper that only tests call.
    """
    safe_name = _sanitize_tool_name(tb.name)

    def _sdk_tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        if not agent.has_tool(tb.name):
            raise ToolAllowlistViolation(
                f"Agent {agent.name} attempted to call {tb.name!r}; "
                f"allowlist: {[t.name for t in agent.tools]}"
            )
        return tb.fn(*args, **kwargs)

    _sdk_tool_wrapper.__name__ = safe_name
    _sdk_tool_wrapper.__qualname__ = safe_name
    _sdk_tool_wrapper.__doc__ = getattr(tb.fn, "__doc__", None)
    try:
        _sdk_tool_wrapper.__signature__ = inspect.signature(tb.fn)  # type: ignore[attr-defined]
    except (TypeError, ValueError):  # pragma: no cover - best effort
        pass
    return _sdk_tool_wrapper


def _translate_tools(agent: "Agent") -> List[Any]:
    """Wrap an agent's ToolBindings in openai-agents function_tool decorators.

    The SDK expects FunctionTool objects; we wrap each binding's callable in
    ``function_tool`` so the runtime exposes it as a callable tool. Dots in
    our internal names (``intent.classify``) are rewritten to underscores to
    satisfy the OpenAI ``^[a-zA-Z0-9_-]+$`` constraint.
    """
    if sdk_function_tool is None:
        return []
    wrapped: List[Any] = []
    for tb in agent.tools:
        try:
            sdk_callable = _prepare_tool_callable_for_sdk(agent, tb)
            wrapped.append(
                sdk_function_tool(
                    sdk_callable,
                    name_override=_sanitize_tool_name(tb.name),
                    # The SDK's introspection can be picky about third-party
                    # callables; relax strict mode so we don't 400 on schema
                    # shape differences between python-fn and sdk-schema.
                    strict_mode=False,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning(
                "framework: could not translate tool %r for SDK (%s); skipping",
                tb.name,
                exc,
            )
    return wrapped


def _extract_tool_calls(run_result: Any) -> List[Dict[str, Any]]:
    """Scrape tool-call metadata from the openai-agents RunResult.

    We return a list of ``{"name": str, "arguments": Any}`` dicts so the
    rest of the facade can stay SDK-agnostic. Best-effort — missing fields
    are tolerated because the SDK's RunItem shape evolves across releases.
    """
    tool_calls: List[Dict[str, Any]] = []
    items = getattr(run_result, "new_items", None) or []
    for item in items:
        item_type = getattr(item, "type", None)
        if item_type != "tool_call_item":
            continue
        raw = getattr(item, "raw_item", None)
        name = getattr(raw, "name", None) or getattr(item, "title", None) or "<unknown>"
        args = getattr(raw, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                pass
        tool_calls.append({"name": name, "arguments": args})
    return tool_calls


def _final_output_text(run_result: Any) -> str:
    """Return the final output as a string, whether it's a dict, model, or str."""
    final = getattr(run_result, "final_output", None)
    if final is None:
        return ""
    if isinstance(final, str):
        return final
    # Pydantic / dataclass / dict → best-effort JSON.
    try:
        if hasattr(final, "model_dump"):
            return json.dumps(final.model_dump(), default=str)
        if hasattr(final, "__dict__"):
            return json.dumps(final.__dict__, default=str)
        return json.dumps(final, default=str)
    except (TypeError, ValueError):
        return str(final)


def _final_output_structured(run_result: Any) -> Optional[Dict[str, Any]]:
    """If the final output is JSON/structured, return it as a dict."""
    final = getattr(run_result, "final_output", None)
    if isinstance(final, dict):
        return final
    if hasattr(final, "model_dump"):
        try:
            return final.model_dump()
        except Exception:  # pragma: no cover
            return None
    if isinstance(final, str):
        stripped = final.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except (TypeError, ValueError):
                return None
    return None


def _structured_from_text(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _opencode_invoke(agent: "Agent", input_payload: Any) -> FakeProviderResponse:
    from perfxpert.providers.opencode_provider import OpencodeProvider

    instructions = agent.fence_text or (
        f"You are the {agent.name} agent. "
        "Follow the JSON payload contract defined in the perfxpert fence."
    )
    model = os.environ.get("PERFXPERT_AGENTS_MODEL_OPENCODE") or os.environ.get(
        "PERFXPERT_LLM_MODEL"
    )
    response = OpencodeProvider().complete(
        [{"role": "user", "content": _serialize_input(input_payload)}],
        system=instructions,
        model=model,
    )
    return FakeProviderResponse(
        text=response.content,
        tool_calls=[],
        structured_output=_structured_from_text(response.content),
        handoff=None,
    )


def _sdk_invoke(agent: "Agent", input_payload: Any, provider: str) -> FakeProviderResponse:
    """Invoke the openai-agents SDK Runner and return a FakeProviderResponse.

    Tests monkeypatch this symbol to return a scripted ``FakeProviderResponse``;
    in production it builds an SDK Agent from our ``Agent`` metadata and calls
    :meth:`Runner.run_sync` synchronously.

    The return type remains ``FakeProviderResponse`` so agents/runtime wiring
    is identical across mocked + live paths.
    """
    if provider == "opencode":
        return _opencode_invoke(agent, input_payload)

    if not _SDK_AVAILABLE or SdkAgent is None or SdkRunner is None or SdkRunConfig is None:
        raise RuntimeError(
            "OpenAI Agents SDK not installed; run `pip install openai-agents` "
            "or set PERFXPERT_AIRGAP=1"
        )

    model = _resolve_model(provider)
    tools = _translate_tools(agent)
    instructions = agent.fence_text or (
        f"You are the {agent.name} agent. "
        "Follow the JSON payload contract defined in the perfxpert fence."
    )

    sdk_kwargs: Dict[str, Any] = {}
    model_settings = _build_model_settings(agent)
    if model_settings is not None:
        sdk_kwargs["model_settings"] = model_settings

    try:
        sdk_agent = SdkAgent(
            name=agent.name,
            instructions=instructions,
            tools=tools,
            model=model,
            **sdk_kwargs,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"framework: failed to construct SDK Agent: {exc}") from exc

    max_turns_env = os.environ.get("PERFXPERT_AGENTS_MAX_TURNS", "10")
    try:
        max_turns = max(1, int(max_turns_env))
    except ValueError:
        max_turns = 10

    input_str = _serialize_input(input_payload)

    try:
        run_config = _build_sdk_run_config(provider)
        run_result = SdkRunner.run_sync(
            starting_agent=sdk_agent,
            input=input_str,
            max_turns=max_turns,
            run_config=run_config,
        )
    except ProviderError:
        raise
    except Exception as exc:
        mapped = _normalize_provider_exception(provider, exc)
        if mapped is not None:
            raise mapped from exc
        raise RuntimeError(f"framework: SDK Runner.run_sync failed: {exc}") from exc

    return FakeProviderResponse(
        text=_final_output_text(run_result),
        tool_calls=_extract_tool_calls(run_result),
        structured_output=_final_output_structured(run_result),
        handoff=None,
    )


# -- Runtime --------------------------------------------------------------


def _airgap_enabled(explicit: Optional[bool]) -> bool:
    """Resolve airgap, letting the environment win.

    ``PERFXPERT_AIRGAP=1`` is set by whoever deployed this into a network the
    tool must not leave, and they cannot audit every call site. A caller
    passing ``airgap=False`` -- including one built from a model-influenced
    argument -- used to turn it back off, so the env var is a floor rather
    than a default.
    """
    env_forced = os.environ.get("PERFXPERT_AIRGAP", "0") == "1"
    return env_forced or explicit is True


def _render_airgap_template(agent: Agent, payload: Any) -> Dict[str, Any]:
    """Produce a deterministic response from templates/airgap_report.txt."""
    template_path = Path(__file__).parent / "templates" / "airgap_report.txt"
    if template_path.exists():
        template = template_path.read_text()
    else:
        template = "[airgap] agent={agent_name} payload={payload}"
    return {
        "_mode": "airgap",
        "airgap": True,
        "agent": agent.name,
        "narrative": template.format(agent_name=agent.name, payload=payload),
    }


def run_agent(
    agent: Agent,
    input_payload: Any,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
    policy: Optional[RunPolicy] = None,
) -> Dict[str, Any]:
    """Run an agent through the SDK (or airgap template).

    This is the single entry point agent modules call. Agents never
    import openai_agents; they compose ToolBinding/Handoff objects and
    let the facade handle execution.

    A response that fails :func:`validate_structured_output` is dropped:
    ``structured_output`` comes back None and ``output_rejected`` carries the
    reason, so the calling agent falls through to its deterministic path
    instead of forwarding unvalidated content.
    """
    if _airgap_enabled(airgap):
        return _render_airgap_template(agent, input_payload)

    effective_policy = policy if policy is not None else _ACTIVE_POLICY.get()
    if effective_policy is not None:
        effective_policy.consume_turn(agent.name)

    resp = _sdk_invoke(agent, input_payload, provider)
    accepted, rejection = validate_structured_output(agent, resp.structured_output)
    if rejection is not None:
        _LOG.warning(
            "framework: discarding %s output that failed %s validation: %s",
            agent.name,
            getattr(agent.output_schema, "__name__", agent.output_schema),
            rejection,
        )

    return {
        "text": resp.text,
        "tool_calls": resp.tool_calls,
        "structured_output": accepted,
        "output_rejected": rejection,
        "handoff": resp.handoff,
    }


def dispatch_tool(agent: Agent, tool_name: str, args: Dict[str, Any]) -> Any:
    """Validate + execute a tool call from within an agent run.

    Rejects calls for tools not in the agent's allowlist (ToolAllowlistViolation).
    """
    if not agent.has_tool(tool_name):
        raise ToolAllowlistViolation(
            f"Agent {agent.name} attempted to call {tool_name!r}; " f"allowlist: {[t.name for t in agent.tools]}"
        )
    binding = next(t for t in agent.tools if t.name == tool_name)
    return binding.fn(**args)


def dispatch_handoff(agent: Agent, target_name: str) -> None:
    """Validate a handoff target against the agent's whitelist."""
    if target_name not in agent.allowed_handoffs:
        raise HandoffPolicyViolation(
            f"Agent {agent.name} attempted handoff to {target_name!r}; " f"whitelist: {agent.allowed_handoffs}"
        )


__all__ = [
    "Agent",
    "Handoff",
    "ToolBinding",
    "RunPolicy",
    "FakeProviderResponse",
    "AgentConstructionError",
    "ToolAllowlistViolation",
    "HandoffPolicyViolation",
    "SessionTurnBudgetExceeded",
    "run_agent",
    "dispatch_tool",
    "dispatch_handoff",
    "validate_structured_output",
]
