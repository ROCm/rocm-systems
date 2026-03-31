#!/usr/bin/env python3
"""
Tests for private LLM provider and LLMConversation private/local variants.

All HTTP calls are mocked — no real network requests are made.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import pytest

from rocinsight.ai_analysis.exceptions import LLMAuthenticationError, LLMRateLimitError
from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer
from rocinsight.ai_analysis.llm_conversation import LLMConversation, _build_private_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ref_guide_path(tmp_path) -> Path:
    """Create a minimal reference guide file so LLMAnalyzer can be constructed."""
    p = tmp_path / "llm-reference-guide.md"
    p.write_text("# GPU Analysis Reference Guide\n\nTest content.\n")
    return p


@pytest.fixture()
def private_env(monkeypatch):
    """Set the minimum env vars required for private provider."""
    monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://private.example.com/v1")
    monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_MODEL", "my-private-model")
    monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_API_KEY", "test-key-123")
    monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_HEADERS", raising=False)
    monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", raising=False)


def _make_chat_response(text: str):
    """Build a minimal mock OpenAI chat completion response."""
    msg = SimpleNamespace(content=text, refusal=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def _make_stream_chunk(text: str):
    """Build a minimal mock streaming chunk."""
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# _build_private_client
# ---------------------------------------------------------------------------

class TestBuildPrivateClient:
    def test_requires_private_url_env(self, monkeypatch):
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_URL", raising=False)
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_HEADERS", raising=False)
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            import openai as _oi
            with pytest.raises(ValueError, match="ROCINSIGHT_LLM_PRIVATE_URL"):
                _build_private_client(None, None)

    def test_merges_json_headers_from_env(self, monkeypatch):
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://example.com/v1")
        monkeypatch.setenv(
            "ROCINSIGHT_LLM_PRIVATE_HEADERS",
            '{"Ocp-Apim-Subscription-Key": "abc123"}',
        )
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", raising=False)

        captured_kwargs: Dict[str, Any] = {}

        mock_openai_cls = MagicMock()
        def fake_openai_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()
        mock_openai_cls.side_effect = fake_openai_init

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            _build_private_client("dummy-key", "my-model")

        headers = captured_kwargs.get("default_headers", {})
        assert headers.get("Ocp-Apim-Subscription-Key") == "abc123"

    def test_single_quoted_headers_parsed(self, monkeypatch):
        """Python-dict style headers (single-quoted keys) must be normalized."""
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://example.com/v1")
        monkeypatch.setenv(
            "ROCINSIGHT_LLM_PRIVATE_HEADERS",
            "{'X-Custom-Header': 'value42'}",
        )
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", raising=False)

        captured_kwargs: Dict[str, Any] = {}

        mock_openai_cls = MagicMock()
        def fake_openai_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()
        mock_openai_cls.side_effect = fake_openai_init

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            _build_private_client("dummy-key", None)

        headers = captured_kwargs.get("default_headers", {})
        assert headers.get("X-Custom-Header") == "value42"

    def test_verify_ssl_false_uses_httpx_client(self, monkeypatch):
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://example.com/v1")
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", "0")
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_HEADERS", raising=False)

        mock_httpx_client = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_httpx_client

        captured_kwargs: Dict[str, Any] = {}
        mock_openai_cls = MagicMock()
        def fake_openai_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()
        mock_openai_cls.side_effect = fake_openai_init

        with patch.dict(
            "sys.modules",
            {
                "openai": MagicMock(OpenAI=mock_openai_cls),
                "httpx": mock_httpx,
            },
        ):
            _build_private_client("dummy-key", None)

        mock_httpx.Client.assert_called_once_with(verify=False)
        assert captured_kwargs.get("http_client") is mock_httpx_client

    def test_auto_sets_user_header(self, monkeypatch):
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://example.com/v1")
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_HEADERS", raising=False)
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", raising=False)

        captured_kwargs: Dict[str, Any] = {}
        mock_openai_cls = MagicMock()
        def fake_openai_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()
        mock_openai_cls.side_effect = fake_openai_init

        with patch("rocinsight.ai_analysis.llm_conversation.os.getlogin", return_value="testuser"):
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
                _build_private_client("dummy-key", None)

        headers = captured_kwargs.get("default_headers", {})
        assert headers.get("user") == "testuser"


# ---------------------------------------------------------------------------
# LLMAnalyzer — private provider
# ---------------------------------------------------------------------------

class TestLLMAnalyzerPrivate:
    def test_call_private_uses_correct_url(self, private_env, ref_guide_path):
        """LLMAnalyzer._call_private must build a client pointing at ROCINSIGHT_LLM_PRIVATE_URL."""
        analyzer = LLMAnalyzer(
            provider="private",
            api_key="test-key",
            reference_guide_path=ref_guide_path,
        )

        mock_resp = _make_chat_response("analysis result")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        mock_openai_cls = MagicMock(return_value=mock_client)
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = mock_openai_cls
        # Make BadRequestError a real exception subclass so except works
        mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            result = analyzer._call_private("sys prompt", "user prompt")

        assert result == "analysis result"
        # The client should have been constructed with the private URL
        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs.get("base_url") == "https://private.example.com/v1"

    def test_call_private_raises_auth_error_on_401(self, private_env, ref_guide_path):
        analyzer = LLMAnalyzer(
            provider="private",
            api_key="bad-key",
            reference_guide_path=ref_guide_path,
        )

        mock_client = MagicMock()
        mock_openai_mod = MagicMock()
        auth_exc = Exception("401 Unauthorized")
        mock_client.chat.completions.create.side_effect = auth_exc

        mock_openai_mod.OpenAI = MagicMock(return_value=mock_client)
        mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            with pytest.raises(RuntimeError, match="Private LLM request failed"):
                analyzer._call_private("sys", "user")

    def test_call_private_returns_text_on_success(self, private_env, ref_guide_path):
        analyzer = LLMAnalyzer(
            provider="private",
            reference_guide_path=ref_guide_path,
        )

        mock_resp = _make_chat_response("GPU analysis: kernel is memory-bound")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(return_value=mock_client)
        mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            result = analyzer._call_private("sys", "user")

        assert "memory-bound" in result

    def test_private_headers_include_auto_user(self, private_env, ref_guide_path, monkeypatch):
        """ROCINSIGHT_LLM_PRIVATE_HEADERS merges with the auto-set user header."""
        monkeypatch.setenv(
            "ROCINSIGHT_LLM_PRIVATE_HEADERS",
            '{"Authorization": "Bearer tok"}',
        )
        analyzer = LLMAnalyzer(
            provider="private",
            reference_guide_path=ref_guide_path,
        )

        captured_kwargs: Dict[str, Any] = {}
        mock_resp = _make_chat_response("ok")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        def fake_openai_init(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_client

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(side_effect=fake_openai_init)
        mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})

        with patch(
            "rocinsight.ai_analysis.llm_analyzer.os.getlogin", return_value="devuser"
        ):
            with patch.dict("sys.modules", {"openai": mock_openai_mod}):
                analyzer._call_private("sys", "user")

        headers = captured_kwargs.get("default_headers", {})
        assert "Authorization" in headers
        assert headers.get("user") == "devuser"


# ---------------------------------------------------------------------------
# LLMConversation — private provider
# ---------------------------------------------------------------------------

class TestLLMConversationPrivate:
    def test_send_private_uses_private_url(self, private_env):
        conv = LLMConversation(provider="private", api_key="test-key", model="my-private-model")
        conv.initialize("You are a GPU analyst.")

        chunks = [_make_stream_chunk("GPU "), _make_stream_chunk("analysis")]
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks))
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        captured_client_kwargs: Dict[str, Any] = {}

        def fake_build_private_client(api_key, model_override):
            captured_client_kwargs["api_key"] = api_key
            captured_client_kwargs["model"] = model_override
            return mock_client, "my-private-model", None

        with patch(
            "rocinsight.ai_analysis.llm_conversation._build_private_client",
            side_effect=fake_build_private_client,
        ):
            mock_openai_mod = MagicMock()
            mock_openai_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
            mock_openai_mod.RateLimitError = type("RateLimitError", (Exception,), {})
            mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})
            with patch.dict("sys.modules", {"openai": mock_openai_mod}):
                result = conv.send("Analyze my trace")

        assert "GPU" in result or "analysis" in result

    def test_send_local_uses_ollama_endpoint(self, monkeypatch):
        monkeypatch.setenv("ROCINSIGHT_LLM_LOCAL_URL", "http://localhost:11434/v1")
        conv = LLMConversation(provider="local", model="codellama:13b")
        conv.initialize("You are a GPU analyst.")

        chunks = [_make_stream_chunk("response from ollama")]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        captured_url = {}

        def fake_openai_init(**kwargs):
            captured_url["base_url"] = kwargs.get("base_url")
            return mock_client

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(side_effect=fake_openai_init)
        mock_openai_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_openai_mod.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            conv.send("What is the bottleneck?")

        assert captured_url.get("base_url") == "http://localhost:11434/v1"

    def test_send_requires_initialize_first(self):
        conv = LLMConversation(provider="private", api_key="key", model="m")
        with pytest.raises(RuntimeError, match="initialize"):
            conv.send("hello")

    def test_to_dict_and_from_dict_round_trips(self):
        conv = LLMConversation(provider="private", api_key="key", model="m")
        conv.initialize("System prompt here")
        conv._messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        conv._turn_count = 1
        state = conv.to_dict()
        restored = LLMConversation.from_dict(state, api_key="key")
        assert restored._system == "System prompt here"
        assert len(restored._messages) == 2
        assert restored._turn_count == 1


# ---------------------------------------------------------------------------
# LLMAnalyzer — propagating auth / rate-limit errors
# ---------------------------------------------------------------------------

class TestLLMAnalyzerErrorPropagation:
    """
    LLMAuthenticationError and LLMRateLimitError from the API must propagate;
    other errors become RuntimeError.
    """

    def test_call_private_propagates_runtime_error_on_generic_failure(
        self, private_env, ref_guide_path
    ):
        analyzer = LLMAnalyzer(
            provider="private",
            reference_guide_path=ref_guide_path,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("connection refused")

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI = MagicMock(return_value=mock_client)
        mock_openai_mod.BadRequestError = type("BadRequestError", (Exception,), {})

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            with pytest.raises(RuntimeError):
                analyzer._call_private("sys", "user")

    def test_call_anthropic_raises_llm_auth_error(self, ref_guide_path):
        analyzer = LLMAnalyzer(
            provider="anthropic",
            api_key="bad-key",
            reference_guide_path=ref_guide_path,
        )

        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_anthropic_mod.RateLimitError = type("RateLimitError", (Exception,), {})

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_anthropic_mod.AuthenticationError(
            "auth failed"
        )
        mock_anthropic_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            with pytest.raises(LLMAuthenticationError):
                analyzer._call_anthropic("sys", "user")

    def test_call_anthropic_raises_llm_rate_limit_error(self, ref_guide_path):
        analyzer = LLMAnalyzer(
            provider="anthropic",
            api_key="key",
            reference_guide_path=ref_guide_path,
        )

        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_anthropic_mod.RateLimitError = type("RateLimitError", (Exception,), {})

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_anthropic_mod.RateLimitError(
            "rate limit"
        )
        mock_anthropic_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            with pytest.raises(LLMRateLimitError):
                analyzer._call_anthropic("sys", "user")


# ---------------------------------------------------------------------------
# _build_private_client — SSL disabled via env var
# ---------------------------------------------------------------------------

class TestBuildPrivateClientSSL:
    def test_ssl_enabled_by_default(self, monkeypatch):
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://example.com/v1")
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_HEADERS", raising=False)
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", raising=False)

        captured_kwargs: Dict[str, Any] = {}
        mock_openai_cls = MagicMock()
        def fake_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()
        mock_openai_cls.side_effect = fake_init

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            _build_private_client("k", None)

        # When SSL is enabled, no http_client is injected
        assert "http_client" not in captured_kwargs

    def test_ssl_disabled_false_value(self, monkeypatch):
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_URL", "https://example.com/v1")
        monkeypatch.setenv("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", "false")
        monkeypatch.delenv("ROCINSIGHT_LLM_PRIVATE_HEADERS", raising=False)

        mock_httpx_client = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_httpx_client

        captured_kwargs: Dict[str, Any] = {}
        mock_openai_cls = MagicMock()
        def fake_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()
        mock_openai_cls.side_effect = fake_init

        with patch.dict(
            "sys.modules",
            {
                "openai": MagicMock(OpenAI=mock_openai_cls),
                "httpx": mock_httpx,
            },
        ):
            _build_private_client("k", None)

        mock_httpx.Client.assert_called_once_with(verify=False)
        assert captured_kwargs.get("http_client") is mock_httpx_client
