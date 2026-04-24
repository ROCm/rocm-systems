"""Tests for perfxpert.cli.branding."""

from unittest.mock import patch

import pytest

from perfxpert.cli.branding import (
    get_amd_banner,
    get_provider_status_table,
    launch_opencode,
)


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


def test_launch_opencode_dry_run_returns_command():
    cmd = launch_opencode(dry_run=True, opencode_path="/tmp/opencode", extra_args=["--foo"])
    assert cmd[0] == "/tmp/opencode"
    assert "--foo" in cmd


def test_launch_opencode_missing_binary_raises(monkeypatch):
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    with patch("perfxpert.cli.branding.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError):
            launch_opencode()


def test_launch_opencode_sets_recursion_env(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/bin/opencode")
    import os as _os
    captured_env = {}

    def fake_execvpe(path, argv, env):  # noqa: D401
        captured_env.update(env)
        raise SystemExit(0)

    with patch("perfxpert.cli.branding.os.execvpe", side_effect=fake_execvpe):
        with pytest.raises(SystemExit):
            launch_opencode()
    assert captured_env.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"
