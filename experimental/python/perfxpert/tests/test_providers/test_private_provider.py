"""Tests for perfxpert.providers.private_provider (OpenAI-compatible endpoints)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers._exceptions import AuthError, DryRunResponse


def _fake_chat_response(text="priv", model="internal-xl", inp=3, out=5):
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "model": model,
        "usage": {"prompt_tokens": inp, "completion_tokens": out},
    }
    return m


def test_dry_run_no_network(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.corp.internal/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "internal-xl")
    from perfxpert.providers.private_provider import PrivateProvider

    with patch("perfxpert.providers.private_provider.httpx.post") as mp:
        assert PrivateProvider().complete([], dry_run=True) is DryRunResponse
        mp.assert_not_called()


def test_missing_url_raises_auth(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_PRIVATE_URL", raising=False)
    from perfxpert.providers.private_provider import PrivateProvider

    with pytest.raises(AuthError):
        PrivateProvider()


def test_posts_to_chat_completions(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.corp.internal/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "internal-xl")
    from perfxpert.providers.private_provider import PrivateProvider

    with patch(
        "perfxpert.providers.private_provider.httpx.post",
        return_value=_fake_chat_response(),
    ) as mp:
        PrivateProvider().complete([{"role": "user", "content": "hi"}])
        url = mp.call_args.args[0]
        assert url == "https://llm.corp.internal/v1/chat/completions"


def test_default_api_key_dummy(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.corp.internal/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "internal-xl")
    monkeypatch.delenv("PERFXPERT_LLM_PRIVATE_API_KEY", raising=False)
    from perfxpert.providers.private_provider import PrivateProvider

    with patch(
        "perfxpert.providers.private_provider.httpx.post",
        return_value=_fake_chat_response(),
    ) as mp:
        PrivateProvider().complete([{"role": "user", "content": "hi"}])
        headers = mp.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer dummy"


def test_custom_headers_merged(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.corp.internal/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "internal-xl")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_API_KEY", "key-abc")
    monkeypatch.setenv(
        "PERFXPERT_LLM_PRIVATE_HEADERS",
        json.dumps({"X-Tenant": "amd-perf", "X-Trace": "1"}),
    )
    from perfxpert.providers.private_provider import PrivateProvider

    with patch(
        "perfxpert.providers.private_provider.httpx.post",
        return_value=_fake_chat_response(),
    ) as mp:
        PrivateProvider().complete([{"role": "user", "content": "hi"}])
        headers = mp.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer key-abc"
        assert headers["X-Tenant"] == "amd-perf"
        assert headers["X-Trace"] == "1"


def test_verify_ssl_disabled_when_env_zero(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://internal.corp/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "m")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_VERIFY_SSL", "0")
    from perfxpert.providers.private_provider import PrivateProvider

    with patch(
        "perfxpert.providers.private_provider.httpx.post",
        return_value=_fake_chat_response(),
    ) as mp:
        PrivateProvider().complete([{"role": "user", "content": "hi"}])
        assert mp.call_args.kwargs["verify"] is False


def test_response_parsed(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.corp.internal/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "internal-xl")
    from perfxpert.providers.private_provider import PrivateProvider

    with patch(
        "perfxpert.providers.private_provider.httpx.post",
        return_value=_fake_chat_response(text="hello", inp=9, out=13),
    ):
        r = PrivateProvider().complete([{"role": "user", "content": "hi"}])
        assert r.content == "hello"
        assert r.provider == "private"
        assert r.input_tokens == 9
        assert r.output_tokens == 13


def test_registered(monkeypatch):
    from perfxpert.providers import registry
    import perfxpert.providers.private_provider  # noqa: F401

    assert "private" in registry.list_providers()


# -- _parse_headers (Finding #7) -------------------------------------------


def test_parse_headers_empty_returns_empty():
    from perfxpert.providers.private_provider import _parse_headers

    assert _parse_headers("") == {}


def test_parse_headers_valid_json_dict():
    from perfxpert.providers.private_provider import _parse_headers

    result = _parse_headers('{"X-Tenant": "amd", "X-Version": "1"}')
    assert result == {"X-Tenant": "amd", "X-Version": "1"}


def test_parse_headers_invalid_json_raises_value_error():
    from perfxpert.providers.private_provider import _parse_headers

    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_headers("not-json{{{")


def test_parse_headers_non_dict_json_raises_value_error():
    from perfxpert.providers.private_provider import _parse_headers

    with pytest.raises(ValueError, match="JSON object"):
        _parse_headers('["X-Tenant", "amd"]')


def test_invalid_headers_env_raises_on_provider_init(monkeypatch):
    """PrivateProvider.__init__ must propagate ValueError from _parse_headers."""
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.corp.internal/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_HEADERS", "this is not json")
    from perfxpert.providers.private_provider import PrivateProvider

    with pytest.raises(ValueError, match="invalid JSON"):
        PrivateProvider()
