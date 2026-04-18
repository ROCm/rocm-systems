"""Tests for perfxpert.providers.ollama_provider."""

from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers._exceptions import DryRunResponse, ProviderError, TimeoutError as PTO


def _fake_response(text="ola", model="llama3", inp=4, out=5):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "message": {"content": text},
        "model": model,
        "prompt_eval_count": inp,
        "eval_count": out,
    }
    m.raise_for_status.return_value = None
    return m


def test_dry_run_no_network():
    from perfxpert.providers.ollama_provider import OllamaProvider
    with patch("perfxpert.providers.ollama_provider.httpx.post") as mock_post:
        p = OllamaProvider()
        assert p.complete([], dry_run=True) is DryRunResponse
        mock_post.assert_not_called()


def test_default_url_localhost(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_LOCAL_URL", raising=False)
    from perfxpert.providers.ollama_provider import OllamaProvider
    with patch(
        "perfxpert.providers.ollama_provider.httpx.post",
        return_value=_fake_response(),
    ) as mock_post:
        OllamaProvider().complete([{"role": "user", "content": "hi"}])
        url = mock_post.call_args.args[0]
        assert url == "http://localhost:11434/api/chat"


def test_custom_url_from_env(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_LOCAL_URL", "http://gpu-box:11434")
    from perfxpert.providers.ollama_provider import OllamaProvider
    with patch(
        "perfxpert.providers.ollama_provider.httpx.post",
        return_value=_fake_response(),
    ) as mock_post:
        OllamaProvider().complete([{"role": "user", "content": "hi"}])
        assert mock_post.call_args.args[0] == "http://gpu-box:11434/api/chat"


def test_complete_response_shape():
    from perfxpert.providers.ollama_provider import OllamaProvider
    with patch(
        "perfxpert.providers.ollama_provider.httpx.post",
        return_value=_fake_response(text="llama-out", inp=7, out=11),
    ):
        r = OllamaProvider().complete(
            [{"role": "user", "content": "hi"}],
            system="be brief",
            model="llama3:8b",
            max_tokens=256,
        )
        assert r.content == "llama-out"
        assert r.provider == "ollama"
        assert r.model == "llama3"
        assert r.input_tokens == 7
        assert r.output_tokens == 11


def test_timeout_maps_to_timeout_error():
    import httpx

    from perfxpert.providers.ollama_provider import OllamaProvider
    with patch(
        "perfxpert.providers.ollama_provider.httpx.post",
        side_effect=httpx.TimeoutException("slow"),
    ):
        with pytest.raises(PTO):
            OllamaProvider().complete([{"role": "user", "content": "x"}])


def test_http_error_maps_to_provider_error():
    import httpx

    from perfxpert.providers.ollama_provider import OllamaProvider
    mock_resp = MagicMock(status_code=500, text="server crashed")
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=mock_resp
    )
    with patch("perfxpert.providers.ollama_provider.httpx.post", return_value=mock_resp):
        with pytest.raises(ProviderError):
            OllamaProvider().complete([{"role": "user", "content": "x"}])


def test_registered():
    from perfxpert.providers import registry
    import perfxpert.providers.ollama_provider  # noqa: F401
    assert "ollama" in registry.list_providers()
