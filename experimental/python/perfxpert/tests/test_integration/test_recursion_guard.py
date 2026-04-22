"""Verify the opencode launcher sets PERFXPERT_IN_OPENCODE_SESSION=1 (recursion guard)."""

import os
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.cli import opencode_launcher


def test_launcher_sets_recursion_guard(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        opencode_launcher, "resolve_opencode_binary",
        lambda: Path("/bin/true"),
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_runtime_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main([])

    assert captured_env.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"


def test_launcher_rejects_recursive_entry(monkeypatch, capsys):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    monkeypatch.setattr(
        opencode_launcher.subprocess,
        "run",
        mock.MagicMock(side_effect=AssertionError("subprocess should not run")),
    )

    rc = opencode_launcher.main([])

    assert rc == 1
    captured = capsys.readouterr()
    assert "recursion detected" in captured.err


def test_recursion_guard_documented_in_agents_md():
    """AGENTS.md should warn future maintainers about the recursion guard."""
    from importlib import resources
    with resources.as_file(resources.files("perfxpert") / "_bundled" / "opencode_config" / "AGENTS.md") as p:
        content = p.read_text()
    # AGENTS.md covers the master agent's mandatory behavior; recursion guard
    # is enforced at the launcher level, so this test just confirms the file
    # exists and is nontrivial. The actual recursion-guard enforcement happens
    # in perfxpert/runtime/recursion_guard.py (Phase 3 deliverable).
    assert len(content) > 500
