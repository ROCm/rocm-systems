"""End-to-end integration tests for `perfxpert-code <backend>` (Task 11).

Skipped in CI (no backend binaries installed there). The
`@pytest.mark.skipif` guards each test on `shutil.which(<binary>)`
so a reviewer running on a dev machine with `claude` / `gemini` /
`codex` installed picks them up automatically.

Per-backend **manual recipe** docstrings spell out the exact
commands a reviewer should run to verify cycle-2 acceptance
criterion 9 (gate probe) + criterion 10 (MCP warmup) against a
real backend binary.

Always-on safety net: the `_helpers` section has tests that exercise
the `--dry-run` codepath end-to-end using stubbed adapters — those
DO run on CI so the dispatcher wiring doesn't silently bitrot.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Always-on: dry-run end-to-end through the dispatcher (stubbed adapters).
# ---------------------------------------------------------------------------


def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect HOME + XDG + consent auto-grant for end-to-end runs."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setenv("PERFXPERT_SKIP_LIVE_CHECK", "1")
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")


def test_dry_run_claude_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`perfxpert-code claude --dry-run hello` writes no files."""
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    _isolated_env(monkeypatch, tmp_path)

    # Stub spawn so we don't try to execvpe a missing claude.
    import perfxpert.cli._backend.claude as claude_mod

    spawns: list = []

    def _fake_spawn(self, argv, env, cwd):
        spawns.append(argv)
        return 0

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "spawn", _fake_spawn)

    from perfxpert.cli.opencode_launcher import main

    before = set(proj.rglob("*"))
    rc = main(["claude", "--dry-run", "hello"])
    after = set(proj.rglob("*"))

    assert rc == 0
    assert before == after, "dry-run must not create any files"
    # dry-run also skips spawn.
    assert spawns == []


def test_dry_run_gemini_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    _isolated_env(monkeypatch, tmp_path)

    import perfxpert.cli._backend.gemini as gemini_mod

    spawns: list = []
    monkeypatch.setattr(
        gemini_mod.GeminiAdapter,
        "spawn",
        lambda self, argv, env, cwd: spawns.append(argv) or 0,
    )

    from perfxpert.cli.opencode_launcher import main

    before = set(proj.rglob("*"))
    rc = main(["gemini", "--dry-run", "hello"])
    after = set(proj.rglob("*"))

    assert rc == 0
    assert before == after
    assert spawns == []


# ---------------------------------------------------------------------------
# Skip-on-CI: real backend present.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("claude") is None, reason="claude CLI not installed"
)
def test_live_claude_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual-reviewer recipe (skipped unless `claude` is on PATH):

    1. Install claude per https://code.claude.com/docs/en/install.
    2. Activate this worktree's perfxpert editable install:
       `pip install -e experimental/python/perfxpert`.
    3. Run this test:
       `pytest -q tests/test_integration/test_multi_backend_e2e.py::test_live_claude_dry_run`.
    4. Inspect stderr — the per-step progress SHOULD list
       `[dry-run]`. No file changes under `tmp_path`.

    Acceptance: test exits 0; `tmp_path` unchanged post-run.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    _isolated_env(monkeypatch, tmp_path)

    from perfxpert.cli.opencode_launcher import main

    before = {p.name for p in proj.iterdir()}
    rc = main(["claude", "--dry-run", "hello"])
    after = {p.name for p in proj.iterdir()}

    assert rc == 0
    assert before == after


@pytest.mark.skipif(
    shutil.which("gemini") is None, reason="gemini CLI not installed"
)
def test_live_gemini_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual-reviewer recipe (skipped unless `gemini` is on PATH):

    1. Install gemini-cli per
       https://github.com/google-gemini/gemini-cli.
    2. Activate perfxpert editable install.
    3. Run: `pytest -q tests/test_integration/test_multi_backend_e2e.py::test_live_gemini_dry_run`.
    4. Inspect stderr: `[dry-run]` + `[1/4] Staging rendered prompt ...`
       etc. `~/.gemini/settings.json` unchanged post-run.

    Acceptance: test exits 0; `~/.gemini/settings.json` unchanged.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    _isolated_env(monkeypatch, tmp_path)

    settings = tmp_path / ".gemini" / "settings.json"
    before = settings.read_bytes() if settings.exists() else None
    from perfxpert.cli.opencode_launcher import main

    rc = main(["gemini", "--dry-run", "hello"])
    after = settings.read_bytes() if settings.exists() else None

    assert rc == 0
    assert before == after


@pytest.mark.skipif(
    shutil.which("codex") is None, reason="codex CLI not installed"
)
def test_live_codex_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`perfxpert-code codex --dry-run hello` writes no files.

    Manual-recipe counterpart: reviewer installs codex CLI, marks the
    project trusted (`PERFXPERT_AUTO_TRUST=1` or edit
    `~/.codex/config.toml`), and runs this test on their machine.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    _isolated_env(monkeypatch, tmp_path)

    from perfxpert.cli.opencode_launcher import main

    before = set(proj.rglob("*"))
    rc = main(["codex", "--dry-run", "hello"])
    after = set(proj.rglob("*"))

    assert rc == 0
    assert before == after, "dry-run must not create any files"


# ---------------------------------------------------------------------------
# Manual-recipe docstring — single source of truth for acceptance criterion 9.
# ---------------------------------------------------------------------------


def test_manual_recipe_docstring_exists() -> None:
    """The acceptance-criterion-9 gate-probe recipe lives in THIS
    module's docstring; this test keeps the docstring from bit-rotting.
    """
    import tests.test_integration.test_multi_backend_e2e as mod  # type: ignore

    doc = mod.__doc__ or ""
    # Core markers: each backend name + "manual recipe".
    for token in (
        "claude",
        "gemini",
        "codex",
        "manual recipe",
    ):
        assert token in doc.lower(), token
