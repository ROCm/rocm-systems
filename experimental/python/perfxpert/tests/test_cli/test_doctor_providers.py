"""Unit tests for `perfxpert doctor` provider-env detection."""

from perfxpert import __main__ as perfxpert_main


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
