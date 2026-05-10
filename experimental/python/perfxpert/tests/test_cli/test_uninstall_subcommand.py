"""Tests for `perfxpert-code uninstall <backend>` (Task 8).

Covers:

* `uninstall` is a perfxpert-owned dispatch subcommand (short-circuits
  before opencode resolution).
* Invokes the correct adapter's `uninstall()` for claude / gemini / codex.
* Missing backend name → exit 2 with usage message.
* `--yes` / `PERFXPERT_ASSUME_CONSENT=1` skips the prompt.
* Round-trip: install → uninstall → files removed (byte-identical
  restore when no pre-existing files).
* Refuses on marker drift (emits non-zero exit with
  `skipped_due_to_drift` listing).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from perfxpert.cli.opencode_launcher import (
    _PERFXPERT_DISPATCH_SUBCOMMANDS,
    _PERFXPERT_SUBCOMMANDS,
    main,
    route_subcommand,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setenv("PERFXPERT_SKIP_LIVE_CHECK", "1")
    return tmp_path


@pytest.fixture
def project_cwd(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = isolated_home / "proj"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# ---------------------------------------------------------------------------
# Dispatch plumbing.
# ---------------------------------------------------------------------------


def test_uninstall_is_perfxpert_owned_dispatch() -> None:
    assert "uninstall" in _PERFXPERT_DISPATCH_SUBCOMMANDS
    assert "uninstall" in _PERFXPERT_SUBCOMMANDS


def test_route_uninstall_is_perfxpert_kind() -> None:
    kind, _ = route_subcommand(["uninstall", "claude"])
    assert kind == "perfxpert"


def test_uninstall_missing_backend_name_returns_2(
    isolated_home: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = main(["uninstall"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "which backend" in err.lower()


def test_uninstall_unknown_backend_returns_2(
    isolated_home: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = main(["uninstall", "bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown backend" in err.lower()


def test_uninstall_codex_routes_to_adapter(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex uninstall now dispatches to CodexAdapter.uninstall, not a stub.

    The output must NOT contain the word "stub" or "ships in PR" (those
    belonged to the PR-1 placeholder message removed in Finding #1).
    """
    monkeypatch.setattr("shutil.which", lambda _: None)  # no codex binary
    # Redirect chdir target to a writable dir inside isolated_home.
    cwd = isolated_home / "codex-proj"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    import io

    fake_err = io.StringIO()
    monkeypatch.setattr("sys.stderr", fake_err)
    rc = main(["uninstall", "--yes", "codex"])
    err = fake_err.getvalue()
    # The adapter succeeds even without the binary (best-effort shell-out
    # plus tomlkit fallback); the key assertion is that we no longer
    # emit the stub message.
    assert rc == 0, f"rc={rc}, err={err!r}"
    assert "stub" not in err.lower()
    assert "ships in PR" not in err
    assert "Task 10" not in err


# ---------------------------------------------------------------------------
# Round-trip (claude).
# ---------------------------------------------------------------------------


def test_install_then_uninstall_round_trip_claude(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)  # no claude CLI
    # Import after monkeypatching to ensure structured-edit fallback path.
    from perfxpert.cli._backend.claude import ClaudeCodeAdapter

    adapter = ClaudeCodeAdapter()
    adapter.install(project_cwd, scope="project")
    mcp_config = project_cwd / ".mcp.json"
    pointer = project_cwd / "CLAUDE.local.md"
    agents_cache = project_cwd / ".perfxpert" / "AGENTS.md"
    assert mcp_config.is_file()
    assert pointer.is_file()
    assert agents_cache.is_file()

    # Now run the uninstall subcommand end-to-end.
    rc = main(["uninstall", "--yes", "claude"])
    assert rc == 0
    # Pointer + cache gone.
    assert not pointer.exists()
    assert not agents_cache.exists()


def test_uninstall_refuses_on_pointer_drift(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Pre-seed a user-edited pointer (no sentinel) so uninstall refuses it.
    (project_cwd / "CLAUDE.local.md").write_text(
        "user's own content — no perfxpert reference\n"
    )
    monkeypatch.setattr("shutil.which", lambda _: None)

    rc = main(["uninstall", "--yes", "claude"])
    # Non-zero exit to signal the partial uninstall.
    assert rc == 1
    err = capsys.readouterr().err
    assert "drift" in err.lower() or "Refused" in err


# ---------------------------------------------------------------------------
# Round-trip (gemini).
# ---------------------------------------------------------------------------


def test_uninstall_gemini_removes_perfxpert_entries(
    project_cwd: Path, isolated_home: Path
) -> None:
    from perfxpert.cli._backend.gemini import GeminiAdapter

    GeminiAdapter().install(project_cwd)
    settings = project_cwd / ".gemini" / "settings.json"
    data = json.loads(settings.read_text())
    assert "perfxpert" in data["mcpServers"]

    rc = main(["uninstall", "--yes", "gemini"])
    assert rc == 0
    data = json.loads(settings.read_text())
    assert "perfxpert" not in data.get("mcpServers", {})


# ---------------------------------------------------------------------------
# Confirmation flow.
# ---------------------------------------------------------------------------


def test_uninstall_refuses_without_yes_in_non_tty(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("PERFXPERT_ASSUME_CONSENT", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["uninstall", "claude"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--yes" in err or "ASSUME_CONSENT" in err


def test_uninstall_env_assume_consent_skips_prompt(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No --yes, but env set → no interactive prompt, proceeds.
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    # Pre-seed the pointer with our sentinel so uninstall has something
    # to do and doesn't error on drift.
    claude_dir = project_cwd / ".claude"
    claude_dir.mkdir()
    (claude_dir / "CLAUDE.md").write_text(
        "<!-- perfxpert-managed pointer file. -->\n@.perfxpert/AGENTS.md\n"
    )
    monkeypatch.setattr("shutil.which", lambda _: None)
    rc = main(["uninstall", "claude"])
    assert rc == 0
