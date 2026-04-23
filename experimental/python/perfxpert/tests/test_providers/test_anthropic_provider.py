"""Tests for perfxpert.providers.anthropic_provider — mocked anthropic SDK."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers import registry
from perfxpert.providers._exceptions import AuthError, DryRunResponse, RateLimitError


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

    with patch("perfxpert.providers.anthropic_provider.anthropic") as mock_sdk:
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
    with patch(
        "perfxpert.providers.anthropic_provider.anthropic.Anthropic",
        return_value=fake_client,
    ):
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

    with patch("perfxpert.providers.anthropic_provider.anthropic.Anthropic") as mock_ctor:
        AnthropicProvider()
        mock_ctor.assert_called_once()
        assert mock_ctor.call_args.kwargs["api_key"] == "sk-perfxpert"


def test_env_var_fallback_to_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fallback")
    from perfxpert.providers.anthropic_provider import AnthropicProvider

    with patch("perfxpert.providers.anthropic_provider.anthropic.Anthropic") as mock_ctor:
        AnthropicProvider()
        assert mock_ctor.call_args.kwargs["api_key"] == "sk-fallback"


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

    with patch("perfxpert.providers.anthropic_provider.anthropic.Anthropic") as mock_ctor:
        AnthropicProvider(api_key="sk-explicit")
        assert mock_ctor.call_args.kwargs["api_key"] == "sk-explicit"
