"""Tests for perfxpert.providers._fallback.FallbackProvider.

The user wanted a way to escape unending client-side rate-limit retries.
FallbackProvider cascades to the next registered provider on
RateLimitError; the opencode patch (.patches/0011-rate-limit-retry-override.patch)
handles the escape hatch in the opencode process.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pytest

from perfxpert.providers import (
    AuthError,
    ENV_FALLBACK_CHAIN,
    FallbackProvider,
    Provider,
    ProviderChainExhausted,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    UnknownProvider,
    get_fallback_provider,
    parse_chain_env,
)


class _StubProvider(Provider):
    """Test double — records invocation, returns or raises as configured."""

    def __init__(
        self,
        *,
        name: str,
        result: Optional[ProviderResponse] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        self.name = name
        self._result = result
        self._raises = raises
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        dry_run: bool = False,
    ) -> Union[ProviderResponse, Any]:
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
                "dry_run": dry_run,
            }
        )
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


def _ok(name: str) -> ProviderResponse:
    return ProviderResponse(
        content="ok",
        provider=name,
        model="m",
        input_tokens=1,
        output_tokens=1,
    )


# ---------------------------------------------------------------------------
# parse_chain_env
# ---------------------------------------------------------------------------


def test_parse_chain_env_none_is_empty() -> None:
    assert parse_chain_env(None) == []


def test_parse_chain_env_empty_string_is_empty() -> None:
    assert parse_chain_env("") == []


def test_parse_chain_env_trims_and_splits() -> None:
    assert parse_chain_env("openai, anthropic,  ollama") == [
        "openai",
        "anthropic",
        "ollama",
    ]


def test_parse_chain_env_drops_blanks() -> None:
    assert parse_chain_env("openai,,anthropic,") == ["openai", "anthropic"]


# ---------------------------------------------------------------------------
# FallbackProvider.complete
# ---------------------------------------------------------------------------


def test_primary_success_returns_without_touching_fallback() -> None:
    primary = _StubProvider(name="primary", result=_ok("primary"))
    secondary = _StubProvider(name="secondary", result=_ok("secondary"))

    fp = FallbackProvider([primary, secondary])
    resp = fp.complete([{"role": "user", "content": "hi"}])
    assert resp.provider == "primary"
    assert len(primary.calls) == 1
    assert secondary.calls == []


def test_rate_limit_falls_over_to_secondary() -> None:
    primary = _StubProvider(
        name="primary", raises=RateLimitError("primary", retry_after=0.0)
    )
    secondary = _StubProvider(name="secondary", result=_ok("secondary"))

    fp = FallbackProvider([primary, secondary])
    resp = fp.complete([{"role": "user", "content": "hi"}])
    assert resp.provider == "secondary"
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1


def test_all_rate_limited_raises_chain_exhausted() -> None:
    primary = _StubProvider(
        name="primary", raises=RateLimitError("primary", retry_after=0.0)
    )
    secondary = _StubProvider(
        name="secondary", raises=RateLimitError("secondary", retry_after=0.0)
    )

    fp = FallbackProvider([primary, secondary])
    with pytest.raises(ProviderChainExhausted) as exc:
        fp.complete([{"role": "user", "content": "hi"}])
    # The last attempt's RateLimitError is preserved as __cause__.
    assert isinstance(exc.value.__cause__, RateLimitError)
    # ProviderChainExhausted is a ProviderError subclass — taxonomy honoured.
    assert isinstance(exc.value, ProviderError)
    # Exhaustion records every attempt in order.
    assert [name for name, _ in exc.value.attempts] == [
        "_StubProvider",
        "_StubProvider",
    ]
    assert all(isinstance(e, RateLimitError) for _, e in exc.value.attempts)


def test_auth_error_in_primary_propagates_immediately() -> None:
    """Non-rate-limit errors are NOT swallowed by fallback."""
    primary = _StubProvider(name="primary", raises=AuthError("primary", "nope"))
    secondary = _StubProvider(name="secondary", result=_ok("secondary"))

    fp = FallbackProvider([primary, secondary])
    with pytest.raises(AuthError):
        fp.complete([{"role": "user", "content": "hi"}])
    # secondary must NOT have been tried
    assert secondary.calls == []


def test_empty_chain_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        FallbackProvider([])


def test_forwards_kwargs_to_each_provider() -> None:
    primary = _StubProvider(
        name="primary", raises=RateLimitError("primary", retry_after=0.0)
    )
    secondary = _StubProvider(name="secondary", result=_ok("secondary"))

    fp = FallbackProvider([primary, secondary])
    fp.complete(
        [{"role": "user", "content": "hi"}],
        system="you are an AMD GPU expert",
        model="m-1",
        max_tokens=42,
        dry_run=False,
    )
    assert primary.calls[0]["system"] == "you are an AMD GPU expert"
    assert primary.calls[0]["model"] == "m-1"
    assert primary.calls[0]["max_tokens"] == 42
    assert secondary.calls[0]["system"] == "you are an AMD GPU expert"
    assert secondary.calls[0]["model"] == "m-1"
    assert secondary.calls[0]["max_tokens"] == 42


# ---------------------------------------------------------------------------
# get_fallback_provider — env integration
# ---------------------------------------------------------------------------


def test_get_fallback_provider_returns_none_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_FALLBACK_CHAIN, raising=False)
    assert get_fallback_provider() is None


def test_get_fallback_provider_returns_instance_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_FALLBACK_CHAIN, "anthropic,openai")
    fp = get_fallback_provider()
    assert isinstance(fp, FallbackProvider)
    # providers are resolved lazily — the instance holds the string chain
    assert fp._providers == ["anthropic", "openai"]  # type: ignore[attr-defined]


def test_get_fallback_provider_explicit_chain_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_FALLBACK_CHAIN, "openai")
    fp = get_fallback_provider(chain=["anthropic", "private"])
    assert isinstance(fp, FallbackProvider)
    assert fp._providers == ["anthropic", "private"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Error taxonomy — I5: unknown provider + chain exhaustion never leak KeyError
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_typed_error() -> None:
    """An unresolvable name must surface as UnknownProvider, never KeyError."""
    fp = FallbackProvider(["__definitely_unregistered_provider__"])
    with pytest.raises(ProviderChainExhausted) as exc:
        fp.complete([{"role": "user", "content": "hi"}])
    # Cause chain must include the UnknownProvider error, not KeyError.
    assert isinstance(exc.value.__cause__, UnknownProvider)
    assert not isinstance(exc.value.__cause__, KeyError)
    # Taxonomy sanity.
    assert isinstance(exc.value.__cause__, ProviderError)


def test_chain_exhaustion_raises_typed_error_with_history() -> None:
    """Every provider entry fails — all attempts are recorded in order."""
    fp = FallbackProvider(
        ["__missing_alpha__", "__missing_beta__", "__missing_gamma__"]
    )
    with pytest.raises(ProviderChainExhausted) as exc:
        fp.complete([{"role": "user", "content": "hi"}])
    assert len(exc.value.attempts) == 3
    assert [name for name, _ in exc.value.attempts] == [
        "__missing_alpha__",
        "__missing_beta__",
        "__missing_gamma__",
    ]
    assert all(isinstance(e, UnknownProvider) for _, e in exc.value.attempts)
    assert exc.value.providers == [
        "__missing_alpha__",
        "__missing_beta__",
        "__missing_gamma__",
    ]


def test_original_exceptions_preserved_in_attempts_list() -> None:
    """Mixed chain (rate-limited stub + unresolvable name) preserves each original."""
    primary = _StubProvider(
        name="primary", raises=RateLimitError("primary", retry_after=0.0)
    )
    fp = FallbackProvider([primary, "__unresolvable_xyz__"])
    with pytest.raises(ProviderChainExhausted) as exc:
        fp.complete([{"role": "user", "content": "hi"}])

    assert len(exc.value.attempts) == 2
    name_a, err_a = exc.value.attempts[0]
    name_b, err_b = exc.value.attempts[1]
    # First attempt: the _StubProvider instance hit RateLimitError.
    assert name_a == "_StubProvider"
    assert isinstance(err_a, RateLimitError)
    assert err_a.provider == "primary"
    # Second attempt: name-string failed to resolve.
    assert name_b == "__unresolvable_xyz__"
    assert isinstance(err_b, UnknownProvider)
    # __cause__ is set to the most recent exception (UnknownProvider).
    assert exc.value.__cause__ is err_b


def test_env_var_chain_of_unknown_names_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: PERFXPERT_LLM_FALLBACK_CHAIN=nonexistent1,nonexistent2 path."""
    monkeypatch.setenv(ENV_FALLBACK_CHAIN, "nonexistent1,nonexistent2")
    fp = get_fallback_provider()
    assert fp is not None
    with pytest.raises(ProviderChainExhausted) as exc:
        fp.complete([{"role": "user", "content": "hi"}])
    # Never bare KeyError.
    assert not isinstance(exc.value, KeyError)
    assert [name for name, _ in exc.value.attempts] == [
        "nonexistent1",
        "nonexistent2",
    ]
