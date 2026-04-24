"""Unit tests for `perfxpert doctor` provider-env detection."""

from perfxpert import __main__ as perfxpert_main


class _Stream:
    def __init__(self, encoding: str):
        self.encoding = encoding


def test_doctor_status_tokens_use_unicode_when_supported():
    assert perfxpert_main._doctor_status_tokens(_Stream("utf-8")) == (
        "✓",
        "⚠",
        "✗",
        "—",
    )


def test_doctor_status_tokens_fallback_to_ascii_for_cp1252():
    assert perfxpert_main._doctor_status_tokens(_Stream("cp1252")) == (
        "[OK]",
        "[WARN]",
        "[FAIL]",
        "-",
    )


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
