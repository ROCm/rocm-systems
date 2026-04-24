"""Unit tests for `perfxpert doctor` provider-env detection."""

from perfxpert import __main__ as perfxpert_main


class _Stream:
    def __init__(self, encoding):
        self.encoding = encoding


def test_status_symbol_falls_back_for_non_utf8_streams():
    assert perfxpert_main._status_symbol("ok", _Stream("cp1252")) == "OK"
    assert perfxpert_main._status_symbol("warn", _Stream("cp1252")) == "WARN"
    assert perfxpert_main._status_symbol("fail", _Stream("cp1252")) == "FAIL"


def test_status_symbol_keeps_unicode_for_utf8_streams():
    assert perfxpert_main._status_symbol("ok", _Stream("utf-8")) == "✓"


def test_check_llm_providers_accepts_canonical_env_names(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("PERFXPERT_LLM_LOCAL_URL", "http://localhost:11434")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://llm.example/v1")

    configured, unconfigured = perfxpert_main._check_llm_providers()

    assert configured == sorted(
        ["anthropic", "ollama", "opencode", "openai", "private"]
    )
    assert unconfigured == []


def test_check_llm_providers_accepts_compatibility_aliases(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("PRIVATE_LLM_ENDPOINT", "https://llm.example/v1")

    configured, unconfigured = perfxpert_main._check_llm_providers()

    assert "ollama" in configured
    assert "private" in configured
    assert "opencode" in configured
