"""Cross-provider error taxonomy + legacy env var deprecation warnings."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers._exceptions import (
    AuthError,
    ProviderError,
    RateLimitError,
    TimeoutError as PTO,
    _legacy_env_warn,
)


def test_legacy_env_warn_emits_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _legacy_env_warn("ROCINSIGHT_LLM_ANTHROPIC_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "ROCINSIGHT_LLM_ANTHROPIC_KEY" in str(w.message)
            and "PERFXPERT_LLM_ANTHROPIC_KEY" in str(w.message)
            for w in caught
        )


def test_rocinsight_legacy_env_honored_with_warning(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ROCINSIGHT_LLM_ANTHROPIC_KEY", "sk-legacy-roc")
    from perfxpert.providers.anthropic_provider import AnthropicProvider
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch("perfxpert.providers.anthropic_provider.anthropic.Anthropic") as ctor:
            AnthropicProvider()
            assert ctor.call_args.kwargs["api_key"] == "sk-legacy-roc"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_rocpd_legacy_env_honored_with_warning(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ROCPD_LLM_OPENAI_KEY", "sk-legacy-rocpd")
    from perfxpert.providers.openai_provider import OpenAIProvider
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch("perfxpert.providers.openai_provider.openai.OpenAI") as ctor:
            OpenAIProvider()
            assert ctor.call_args.kwargs["api_key"] == "sk-legacy-rocpd"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_anthropic_auth_error_normalized(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test")
    import anthropic as real

    from perfxpert.providers.anthropic_provider import AnthropicProvider
    fake = MagicMock()
    fake.messages.create.side_effect = real.AuthenticationError(
        message="bad key",
        response=MagicMock(status_code=401),
        body={"error": {"message": "bad"}},
    )
    with patch(
        "perfxpert.providers.anthropic_provider.anthropic.Anthropic",
        return_value=fake,
    ):
        p = AnthropicProvider()
        with pytest.raises(AuthError):
            p.complete([{"role": "user", "content": "x"}])


def test_openai_rate_limit_normalized(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-test")
    import openai as real

    from perfxpert.providers.openai_provider import OpenAIProvider
    fake = MagicMock()
    fake.chat.completions.create.side_effect = real.RateLimitError(
        message="slow down",
        response=MagicMock(status_code=429),
        body={"error": {"message": "rl"}},
    )
    with patch(
        "perfxpert.providers.openai_provider.openai.OpenAI",
        return_value=fake,
    ):
        with pytest.raises(RateLimitError):
            OpenAIProvider().complete([{"role": "user", "content": "x"}])
