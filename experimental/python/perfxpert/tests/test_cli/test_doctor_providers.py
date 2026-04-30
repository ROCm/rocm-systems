"""Unit tests for `perfxpert doctor` provider-env detection."""

import io
import sys

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


def test_run_doctor_is_safe_for_ascii_stdout(monkeypatch):
    task_store_msg = (
        "Python task store ready\n"
        "  WARNING: PERFXPERT_TASK_ROOT not set — defaulting to /tmp/.perfxpert. "
        "Set PERFXPERT_TASK_ROOT to use a custom location."
    )
    monkeypatch.setattr(perfxpert_main, "_check_version", lambda: (True, "perfxpert 0.1"))
    monkeypatch.setattr(
        perfxpert_main,
        "_check_python_version",
        lambda: (True, "Python 3.10"),
    )
    monkeypatch.setattr(
        perfxpert_main,
        "_check_openai_agents",
        lambda: (True, "openai-agents ok"),
    )
    monkeypatch.setattr(
        perfxpert_main,
        "_check_mcp_server",
        lambda: (False, "MCP server FAILED"),
    )
    monkeypatch.setattr(perfxpert_main, "_check_task_store", lambda: (True, task_store_msg))
    monkeypatch.setattr(
        perfxpert_main,
        "_check_opencode_bundled",
        lambda: (True, "Bundled opencode ok"),
    )
    monkeypatch.setattr(
        perfxpert_main,
        "_check_opencode_bundled_config",
        lambda: (True, "Bundled config ok"),
    )
    monkeypatch.setattr(
        perfxpert_main,
        "_check_llm_providers",
        lambda: (["opencode"], ["anthropic"]),
    )

    buffer = io.BytesIO()
    stdout = io.TextIOWrapper(buffer, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)

    assert perfxpert_main._run_doctor() == 1
    stdout.flush()
    output = buffer.getvalue().decode("ascii")

    assert "PERFXPERT_TASK_ROOT not set - defaulting" in output
    assert "providers unconfigured (anthropic) - see README" in output
    assert "FAIL Issues found - see above" in output


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
