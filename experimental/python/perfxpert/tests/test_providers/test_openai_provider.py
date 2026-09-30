"""Tests for perfxpert.providers.openai_provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers._exceptions import AuthError, DryRunResponse
import perfxpert.providers.openai_provider as _oaimod


def _fake_response(text="hi", model="gpt-4o-mini", inp=3, out=4):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        model=model,
        usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out),
    )


def test_dry_run_returns_singleton(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-test")
    from perfxpert.providers.openai_provider import OpenAIProvider

    mock_sdk = MagicMock()
    with patch.object(_oaimod, "_SDK", mock_sdk):
        prov = OpenAIProvider()
        assert prov.complete([], dry_run=True) is DryRunResponse
        # dry_run must not perform the network-bearing API call.
        mock_sdk.OpenAI.return_value.chat.completions.create.assert_not_called()


def test_complete_returns_response(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-test")
    from perfxpert.providers.openai_provider import OpenAIProvider

    fake = MagicMock()
    fake.chat.completions.create.return_value = _fake_response(
        text="greetings", inp=8, out=12, model="gpt-4o"
    )
    mock_sdk = MagicMock()
    mock_sdk.OpenAI.return_value = fake
    with patch.object(_oaimod, "_SDK", mock_sdk):
        p = OpenAIProvider()
        r = p.complete(
            [{"role": "user", "content": "hi"}],
            system="sys",
            model="gpt-4o",
            max_tokens=100,
        )
        assert r.content == "greetings"
        assert r.provider == "openai"
        assert r.input_tokens == 8
        assert r.output_tokens == 12


def test_falls_back_to_max_tokens_on_bad_request(monkeypatch):
    """Older API path: max_completion_tokens unsupported \u2192 retry with max_tokens."""
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-test")
    import openai as real_openai

    from perfxpert.providers.openai_provider import OpenAIProvider

    fake = MagicMock()
    # Create a mock response with proper request attribute
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.request = MagicMock()
    err = real_openai.BadRequestError(
        message="unknown param: max_completion_tokens",
        response=mock_response,
        body={"error": {"code": "unknown_parameter"}},
    )
    fake.chat.completions.create.side_effect = [err, _fake_response()]
    mock_sdk = MagicMock()
    mock_sdk.OpenAI.return_value = fake
    mock_sdk.BadRequestError = real_openai.BadRequestError
    with patch.object(_oaimod, "_SDK", mock_sdk):
        p = OpenAIProvider()
        r = p.complete([{"role": "user", "content": "hi"}], max_tokens=256)
        assert r.content == "hi"
        # First call used max_completion_tokens; second fell back to max_tokens
        call1, call2 = fake.chat.completions.create.call_args_list
        assert "max_completion_tokens" in call1.kwargs
        assert "max_tokens" in call2.kwargs


def test_env_precedence(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-perfxpert")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
    from perfxpert.providers.openai_provider import OpenAIProvider

    mock_sdk = MagicMock()
    with patch.object(_oaimod, "_SDK", mock_sdk):
        OpenAIProvider()
        assert mock_sdk.OpenAI.call_args.kwargs["api_key"] == "sk-perfxpert"


def test_missing_key_raises_auth(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from perfxpert.providers.openai_provider import OpenAIProvider

    with pytest.raises(AuthError):
        OpenAIProvider()


def test_registered_under_openai(monkeypatch):
    from perfxpert.providers import registry
    import perfxpert.providers.openai_provider  # noqa: F401

    assert "openai" in registry.list_providers()


def test_missing_sdk_raises_external_tool_missing(monkeypatch):
    """N28: absent openai SDK raises ExternalToolMissing, not ImportError."""
    from perfxpert.providers.openai_provider import OpenAIProvider
    from perfxpert.tools._tooldep import ExternalToolMissing

    with patch("perfxpert.providers.openai_provider.require_tool") as mock_rt:
        mock_rt.side_effect = ExternalToolMissing(
            name="openai", install_hint="pip install openai",
        )
        monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-test")
        with pytest.raises(ExternalToolMissing) as exc_info:
            OpenAIProvider()
        assert exc_info.value.name == "openai"
        assert "pip install openai" in exc_info.value.install_hint
