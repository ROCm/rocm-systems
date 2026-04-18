"""Tests for perfxpert.providers.anthropic_provider — mocked anthropic SDK."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers import registry
from perfxpert.providers._exceptions import AuthError, DryRunResponse, RateLimitError
import perfxpert.providers.anthropic_provider as _anthmod


def _fake_anthropic_response(text="hi", inp=5, out=7, model="claude-3-sonnet"):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        model=model,
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out),
        stop_reason="end_turn",
    )


def test_import_registers_anthropic(monkeypatch):
    import importlib

    import perfxpert.providers.anthropic_provider as mod
    importlib.reload(mod)
    assert "anthropic" in registry.list_providers()


def test_dry_run_returns_singleton_no_network(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test-dry")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    mock_sdk = MagicMock()
    with patch.object(_anthmod, "_SDK", mock_sdk):
        provider = AnthropicProvider()
        result = provider.complete([{"role": "user", "content": "hi"}], dry_run=True)
        assert result is DryRunResponse
        # dry_run must not perform the network-bearing API call.
        mock_sdk.Anthropic.return_value.messages.create.assert_not_called()


def test_complete_returns_provider_response(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_response(
        text="hello world", inp=11, out=22, model="claude-3-5-sonnet-20241022"
    )
    mock_sdk = MagicMock()
    mock_sdk.Anthropic.return_value = fake_client
    with patch.object(_anthmod, "_SDK", mock_sdk):
        provider = AnthropicProvider()
        r = provider.complete(
            [{"role": "user", "content": "hi"}],
            system="you are helpful",
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        assert r.content == "hello world"
        assert r.provider == "anthropic"
        assert r.model == "claude-3-5-sonnet-20241022"
        assert r.input_tokens == 11
        assert r.output_tokens == 22
        assert r.total_tokens == 33


def test_env_var_precedence_perfxpert_over_anthropic(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-perfxpert")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fallback")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    mock_sdk = MagicMock()
    with patch.object(_anthmod, "_SDK", mock_sdk):
        AnthropicProvider()
        mock_sdk.Anthropic.assert_called_once()
        assert mock_sdk.Anthropic.call_args.kwargs["api_key"] == "sk-perfxpert"


def test_env_var_fallback_to_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fallback")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    mock_sdk = MagicMock()
    with patch.object(_anthmod, "_SDK", mock_sdk):
        AnthropicProvider()
        assert mock_sdk.Anthropic.call_args.kwargs["api_key"] == "sk-fallback"


def test_missing_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    with pytest.raises(AuthError) as exc:
        AnthropicProvider()
    assert "anthropic" in str(exc.value).lower()


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-env")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    mock_sdk = MagicMock()
    with patch.object(_anthmod, "_SDK", mock_sdk):
        AnthropicProvider(api_key="sk-explicit")
        assert mock_sdk.Anthropic.call_args.kwargs["api_key"] == "sk-explicit"


def test_anthropic_api_timeout_normalized_to_provider_timeout_error(monkeypatch):
    """anthropic.APITimeoutError must surface as providers._exceptions.TimeoutError.

    Finding #23: the normalization path in AnthropicProvider.complete was never
    exercised by a test. The except clause catches anthropic.APITimeoutError and
    re-raises as perfxpert.providers._exceptions.TimeoutError with provider='anthropic'.
    """
    import anthropic as _anthropic_sdk
    from perfxpert.providers._exceptions import TimeoutError as ProviderTimeoutError

    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    fake_client = MagicMock()
    # Build a real anthropic.APITimeoutError — it requires a `request` kwarg.
    req = MagicMock()
    fake_client.messages.create.side_effect = _anthropic_sdk.APITimeoutError(request=req)

    # Patch _SDK so our fake_client is returned, while preserving the real
    # exception classes so the except clauses fire correctly.
    mock_sdk = MagicMock()
    mock_sdk.Anthropic.return_value = fake_client
    mock_sdk.AuthenticationError = _anthropic_sdk.AuthenticationError
    mock_sdk.RateLimitError = _anthropic_sdk.RateLimitError
    mock_sdk.APITimeoutError = _anthropic_sdk.APITimeoutError
    mock_sdk.APIError = _anthropic_sdk.APIError
    with patch.object(_anthmod, "_SDK", mock_sdk):
        provider = AnthropicProvider()
        with pytest.raises(ProviderTimeoutError) as exc_info:
            provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "anthropic", (
        f"TimeoutError.provider must be 'anthropic', got {exc_info.value.provider!r}"
    )


def test_missing_sdk_raises_external_tool_missing(monkeypatch):
    """N28: absent anthropic SDK raises ExternalToolMissing, not ImportError."""
    from perfxpert.providers.anthropic_provider import AnthropicProvider
    from perfxpert.tools._tooldep import ExternalToolMissing

    with patch("perfxpert.providers.anthropic_provider.require_tool") as mock_rt:
        mock_rt.side_effect = ExternalToolMissing(
            name="anthropic", install_hint="pip install anthropic",
        )
        monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test")
        with pytest.raises(ExternalToolMissing) as exc_info:
            AnthropicProvider()
        assert exc_info.value.name == "anthropic"
        assert "pip install anthropic" in exc_info.value.install_hint
