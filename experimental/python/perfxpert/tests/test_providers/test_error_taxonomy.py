"""Cross-provider error taxonomy + env-var alias deprecation warnings.

Regression guards: the provider layer still honors pre-rename
API-key env vars (regression guard for back-compat contract), emitting
a DeprecationWarning that points users to the canonical
`PERFXPERT_LLM_*` names. These aliases are an active back-compat
contract and deliberately outlived the Phase 7.1 ai_analysis module
removal — they concern user credentials, not code paths. The pre-rename
names (regression guard):

- `ROCINSIGHT_LLM_*` → `PERFXPERT_LLM_*` (regression guard)
- `ROCPD_LLM_*` → `PERFXPERT_LLM_*` (regression guard)
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
    # Regression guard — API-key alias back-compat contract.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _legacy_env_warn("ROCINSIGHT_LLM_ANTHROPIC_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY")  # regression guard
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "ROCINSIGHT_LLM_ANTHROPIC_KEY" in str(w.message)  # regression guard
            and "PERFXPERT_LLM_ANTHROPIC_KEY" in str(w.message)
            for w in caught
        )


def test_prerename_anthropic_env_honored_with_warning(monkeypatch):
    # Regression guard — API-key alias back-compat contract.
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ROCINSIGHT_LLM_ANTHROPIC_KEY", "sk-prerename-roc")  # regression guard
    from perfxpert.providers.anthropic_provider import AnthropicProvider
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mock_sdk = MagicMock()
        with patch.object(_anthmod, "_SDK", mock_sdk):
            AnthropicProvider()
            assert mock_sdk.Anthropic.call_args.kwargs["api_key"] == "sk-prerename-roc"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


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
