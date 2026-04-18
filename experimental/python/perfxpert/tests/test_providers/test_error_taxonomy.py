"""Cross-provider error taxonomy + env-var alias deprecation warnings.

Regression guards: the provider layer still honors the pre-rename
API-key env var alias `ROCPD_LLM_*`, emitting a DeprecationWarning
that points users to the canonical `PERFXPERT_LLM_*` names.
"""

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
import perfxpert.providers.anthropic_provider as _anthmod
import perfxpert.providers.openai_provider as _oaimod


def test_legacy_env_warn_emits_deprecation():
    # Regression guard — canonical PERFXPERT_LLM_* is the target.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _legacy_env_warn("ROCPD_LLM_ANTHROPIC_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "ROCPD_LLM_ANTHROPIC_KEY" in str(w.message)
            and "PERFXPERT_LLM_ANTHROPIC_KEY" in str(w.message)
            for w in caught
        )


def test_rocpd_legacy_env_honored_with_warning(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ROCPD_LLM_OPENAI_KEY", "sk-legacy-rocpd")
    from perfxpert.providers.openai_provider import OpenAIProvider
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mock_sdk = MagicMock()
        with patch.object(_oaimod, "_SDK", mock_sdk):
            OpenAIProvider()
            assert mock_sdk.OpenAI.call_args.kwargs["api_key"] == "sk-legacy-rocpd"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_anthropic_auth_error_normalized(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test")
    import anthropic as real

    from perfxpert.providers.anthropic_provider import AnthropicProvider
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = real.AuthenticationError(
        message="bad key",
        response=MagicMock(status_code=401),
        body={"error": {"message": "bad"}},
    )
    mock_sdk = MagicMock()
    mock_sdk.Anthropic.return_value = fake_client
    mock_sdk.AuthenticationError = real.AuthenticationError
    mock_sdk.RateLimitError = real.RateLimitError
    mock_sdk.APITimeoutError = real.APITimeoutError
    mock_sdk.APIError = real.APIError
    with patch.object(_anthmod, "_SDK", mock_sdk):
        p = AnthropicProvider()
        with pytest.raises(AuthError):
            p.complete([{"role": "user", "content": "x"}])


def test_openai_rate_limit_normalized(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-test")
    import openai as real

    from perfxpert.providers.openai_provider import OpenAIProvider
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = real.RateLimitError(
        message="slow down",
        response=MagicMock(status_code=429),
        body={"error": {"message": "rl"}},
    )
    mock_sdk = MagicMock()
    mock_sdk.OpenAI.return_value = fake_client
    mock_sdk.AuthenticationError = real.AuthenticationError
    mock_sdk.RateLimitError = real.RateLimitError
    mock_sdk.APITimeoutError = real.APITimeoutError
    mock_sdk.APIError = real.APIError
    mock_sdk.BadRequestError = real.BadRequestError
    with patch.object(_oaimod, "_SDK", mock_sdk):
        with pytest.raises(RateLimitError):
            OpenAIProvider().complete([{"role": "user", "content": "x"}])
