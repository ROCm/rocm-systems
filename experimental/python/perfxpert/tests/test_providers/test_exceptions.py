"""Tests for perfxpert.providers._exceptions — error taxonomy + DryRunResponse."""

import pytest

from perfxpert.providers._exceptions import (
    AuthError,
    DryRunResponse,
    FatalError,
    ProviderChainExhausted,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError as ProviderTimeoutError,
    TransientError,
)


def test_provider_error_is_exception():
    assert issubclass(ProviderError, Exception)


def test_auth_error_inherits_provider_error():
    assert issubclass(AuthError, ProviderError)
    e = AuthError("anthropic")
    assert "anthropic" in str(e)


def test_rate_limit_error_carries_retry_after():
    e = RateLimitError("openai", retry_after=30.0)
    assert e.provider == "openai"
    assert e.retry_after == 30.0


def test_rate_limit_error_default_retry_after_zero():
    e = RateLimitError("openai")
    assert e.retry_after == 0.0


def test_quota_exceeded_error_tracks_provider_and_model():
    e = QuotaExceededError("openai", model="gpt-4o", message="quota reached")
    assert e.provider == "openai"
    assert e.model == "gpt-4o"
    assert e.raw_message == "quota reached"


def test_transient_error_tracks_kind():
    e = TransientError("anthropic", kind="api_unavailable", message="try again")
    assert e.provider == "anthropic"
    assert e.kind == "api_unavailable"
    assert e.raw_message == "try again"


def test_fatal_error_tracks_raw_message():
    e = FatalError("private", "schema mismatch")
    assert e.provider == "private"
    assert e.raw_message == "schema mismatch"


def test_timeout_error_carries_timeout_seconds():
    e = ProviderTimeoutError("ollama", timeout_seconds=60.0)
    assert e.provider == "ollama"
    assert e.timeout_seconds == 60.0


def test_provider_chain_exhausted_tracks_attempted_providers_and_last_error():
    last_error = RateLimitError("openai", retry_after=30.0)
    e = ProviderChainExhausted(["anthropic", "openai"], last_error=last_error)
    assert e.providers == ("anthropic", "openai")
    assert e.last_error is last_error


def test_dry_run_response_is_singleton():
    from perfxpert.providers._exceptions import DryRunResponse as A
    from perfxpert.providers._exceptions import DryRunResponse as B
    assert A is B


def test_dry_run_response_fields():
    assert DryRunResponse.content == ""
    assert DryRunResponse.provider == "dry_run"
    assert DryRunResponse.model == "dry_run"
    assert DryRunResponse.input_tokens == 0
    assert DryRunResponse.output_tokens == 0
    assert DryRunResponse.total_tokens == 0


def test_public_exports_available_from_package():
    from perfxpert.providers import (
        AuthError,
        DryRunResponse,
        FatalError,
        ProviderChainExhausted,
        ProviderError,
        QuotaExceededError,
        RateLimitError,
        TransientError,
        TimeoutError,
    )
    assert ProviderError is not None
    assert DryRunResponse is not None
