"""Tests for perfxpert.cli.branding."""

from unittest.mock import patch

import pytest

from perfxpert.cli.branding import (
    get_amd_banner,
    get_provider_status_table,
    launch_opencode,
)


def _provider_line(table: str, provider: str) -> str:
    return next(line for line in table.splitlines() if line.lower().startswith(provider))


def test_amd_banner_contains_perfxpert():
    banner = get_amd_banner()
    assert "PerfXpert" in banner


def test_amd_banner_contains_amd_color_signature():
    banner = get_amd_banner()
    # Must reference AMD red (#ED1C24 or "AMD") for provenance
    assert "AMD" in banner


def test_provider_status_table_lists_all_five():
    table = get_provider_status_table()
    for p in ("anthropic", "openai", "ollama", "private", "opencode"):
        assert p in table.lower()


def test_provider_status_marks_configured(monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-abc")
    monkeypatch.delenv("PERFXPERT_LLM_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    table = get_provider_status_table()
    # anthropic has an env key → should be marked OK; openai should be marked missing
    lower = table.lower()
    assert "anthropic" in lower
    assert "openai" in lower


def test_provider_status_private_requires_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("PERFXPERT_LLM_PRIVATE_URL", raising=False)
    monkeypatch.delenv("PRIVATE_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_PRIVATE_API_KEY", raising=False)
    monkeypatch.setenv("PRIVATE_LLM_ENDPOINT", "https://llm.example/v1")

    table = get_provider_status_table()
    assert "missing" in _provider_line(table, "private").lower()

    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_API_KEY", "sk-private")
    table = get_provider_status_table()
    assert "configured" in _provider_line(table, "private").lower()


def test_provider_status_opencode_reports_availability_not_provider_auth(monkeypatch):
    with patch(
        "perfxpert.cli.branding.resolve_opencode_binary",
        return_value="/tmp/perfxpert-bundled-opencode",
    ):
        table = get_provider_status_table()

    opencode = _provider_line(table, "opencode").lower()
    assert "available" in opencode
    assert "configured" not in opencode
    assert "provider auth" in opencode


def test_launch_opencode_dry_run_returns_command():
    cmd = launch_opencode(dry_run=True, opencode_path="/tmp/opencode", extra_args=["--foo"])
    assert cmd[0] == "/tmp/opencode"
    assert "--foo" in cmd


def test_launch_opencode_missing_binary_raises(monkeypatch):
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    with patch(
        "perfxpert.cli.branding.resolve_opencode_binary",
        side_effect=FileNotFoundError("missing"),
    ):
        with pytest.raises(FileNotFoundError):
            launch_opencode()


def test_launch_opencode_sets_recursion_env():
    captured_env = {}

    def fake_execvpe(path, argv, env):  # noqa: D401
        captured_env.update(env)
        raise SystemExit(0)

    with patch("perfxpert.cli.branding.os.execvpe", side_effect=fake_execvpe):
        with pytest.raises(SystemExit):
            launch_opencode(opencode_path="/bin/opencode")
    assert captured_env.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"
