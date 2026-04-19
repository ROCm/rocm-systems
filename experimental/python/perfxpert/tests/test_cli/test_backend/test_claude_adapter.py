"""Tests for `perfxpert.cli._backend.claude.ClaudeCodeAdapter` (Task 4b).

Covers:

* check_available — missing binary, version-too-old, happy path.
* tool-name template (B1).
* plan — enumerates every target file.
* install primary path (shells out to `claude mcp add`).
* install fallback (structured edit of .mcp.json, project scope only).
* install refuses whole-file edit of ~/.claude.json (I4).
* install preserves existing mcpServers in .mcp.json.
* install is idempotent.
* install appends .mcp.json to .gitignore if present.
* install refuses to touch a git-tracked CLAUDE.md without --allow-flag.
* uninstall refuses on block drift.
* spawn uses execvpe (not subprocess).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Snapshot the real subprocess.run before any test monkey-patches it.
# Used inside fake _run callables to invoke git (which we DON'T want to
# intercept) without re-entering our own fake.
_REAL_RUN = subprocess.run

import pytest

from perfxpert.cli._backend.claude import (
    ClaudeCodeAdapter,
    SKIP_LIVE_CHECK_ENV,
)
from perfxpert.cli._backend.protocol import (
    BackendAdapter,
    BackendNotFound,
    ConfigClobber,
    ConsentDenied,
    InstallReport,
    Plan,
    UninstallReport,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME + XDG_CONFIG_HOME so tests never touch real home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setenv(SKIP_LIVE_CHECK_ENV, "1")
    return tmp_path


@pytest.fixture
def project_cwd(isolated_home: Path) -> Path:
    """A git-initialized fake project cwd."""
    cwd = isolated_home / "proj"
    cwd.mkdir()
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"],
        cwd=str(cwd),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=str(cwd), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(cwd), check=True)
    return cwd


# ---------------------------------------------------------------------------
# check_available.
# ---------------------------------------------------------------------------


def test_adapter_conforms_to_protocol() -> None:
    assert isinstance(ClaudeCodeAdapter(), BackendAdapter)


def test_tool_name_template_is_mcp_double_underscore() -> None:
    """B1: claude wire format is `mcp__perfxpert__<tool>`."""
    assert ClaudeCodeAdapter.tool_name_template == "mcp__perfxpert__{tool}"


def test_spawn_strategy_is_execvpe() -> None:
    """I1: claude is a TUI; must exec rather than shell out."""
    assert ClaudeCodeAdapter.spawn_strategy == "execvpe"


def test_check_available_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = ClaudeCodeAdapter()
    ok, reason = adapter.check_available()
    assert ok is False
    assert "not found on PATH" in reason


def test_check_available_version_too_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    class _FakeResult:
        returncode = 0
        stdout = "claude 1.0.0\n"
        stderr = ""

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run",
        lambda *a, **kw: _FakeResult(),
    )
    adapter = ClaudeCodeAdapter()
    ok, reason = adapter.check_available()
    assert ok is False
    assert "below" in reason


def test_check_available_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    class _FakeResult:
        returncode = 0
        stdout = "claude 2.1.59\n"
        stderr = ""

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run",
        lambda *a, **kw: _FakeResult(),
    )
    adapter = ClaudeCodeAdapter()
    ok, reason = adapter.check_available()
    assert ok is True
    # reason is the path or version detail.
    assert reason


# ---------------------------------------------------------------------------
# plan.
# ---------------------------------------------------------------------------


def test_plan_lists_targets(project_cwd: Path) -> None:
    adapter = ClaudeCodeAdapter()
    plan = adapter.plan(project_cwd)
    assert isinstance(plan, Plan)
    assert plan.backend == "claude"
    # Every file the installer will touch.
    target_names = {p.name for p in plan.targets}
    assert ".mcp.json" in target_names
    assert "CLAUDE.local.md" in target_names  # canonical local-override
    assert "AGENTS.md" in target_names


def test_plan_actions_mention_each_step(project_cwd: Path) -> None:
    adapter = ClaudeCodeAdapter()
    plan = adapter.plan(project_cwd)
    joined = "\n".join(plan.actions).lower()
    assert "register" in joined
    assert "pointer" in joined
    assert "verify" in joined


# ---------------------------------------------------------------------------
# install — primary shell-out path.
# ---------------------------------------------------------------------------


def _fake_claude_subprocess_factory(
    *,
    get_exit: int = 1,
    add_exit: int = 0,
    remove_exit: int = 0,
):
    """Return a `subprocess.run` replacement that mimics the claude CLI."""

    def _run(cmd, *args, **kwargs):
        class _R:
            def __init__(self, rc: int) -> None:
                self.returncode = rc
                self.stdout = b""
                self.stderr = b""

        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] == "git":
            # Real git — forward to the saved subprocess.run snapshot.
            return _REAL_RUN(cmd, *args, **kwargs)
        if isinstance(cmd, list) and "mcp" in cmd:
            verb = cmd[cmd.index("mcp") + 1]
            if verb == "get":
                return _R(get_exit)
            if verb == "add":
                return _R(add_exit)
            if verb == "remove":
                return _R(remove_exit)
        return _R(0)

    return _run


def test_install_shells_out_to_claude_mcp_add(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    called: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        class _R:
            def __init__(self, rc: int) -> None:
                self.returncode = rc
                self.stdout = b""
                self.stderr = b""

        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        called.append(list(cmd))
        # Simulate "not yet registered" then "add ok".
        if "get" in cmd:
            return _R(1)
        return _R(0)

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )

    adapter = ClaudeCodeAdapter()
    report = adapter.install(project_cwd, scope="project")
    assert isinstance(report, InstallReport)
    # Any call must have included `mcp add perfxpert`.
    add_calls = [c for c in called if "add" in c and "perfxpert" in c]
    assert add_calls, f"expected `claude mcp add perfxpert` call; got {called}"


def test_install_idempotent_via_mcp_get(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `claude mcp get perfxpert` returns 0, skip the add."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    seen_verbs: list[str] = []

    def _run(cmd, *args, **kwargs):
        class _R:
            returncode = 0
            stdout = b""
            stderr = b""

        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        if "mcp" in cmd:
            verb = cmd[cmd.index("mcp") + 1]
            seen_verbs.append(verb)
        return _R()

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    adapter = ClaudeCodeAdapter()
    adapter.install(project_cwd)
    # "get" should appear (the idempotency probe); "add" should NOT.
    assert "get" in seen_verbs
    assert "add" not in seen_verbs


# ---------------------------------------------------------------------------
# install — fallback structured edit.
# ---------------------------------------------------------------------------


def test_install_fallback_structured_edit_on_shellout_failure(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `claude mcp add` returns nonzero, we print-for-human and
    structured-edit .mcp.json (project scope)."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *args, **kwargs):
        class _R:
            def __init__(self, rc: int) -> None:
                self.returncode = rc
                self.stdout = b""
                self.stderr = b""

        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        return _R(1)  # every claude invocation fails

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    adapter = ClaudeCodeAdapter()
    adapter.install(project_cwd, scope="project")
    mcp_config = project_cwd / ".mcp.json"
    assert mcp_config.is_file()
    data = json.loads(mcp_config.read_text())
    assert data["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_install_user_scope_refuses_whole_file_edit(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I4: user scope has no cli → structured edit is refused."""
    monkeypatch.setattr("shutil.which", lambda _: None)  # no claude binary
    adapter = ClaudeCodeAdapter()
    with pytest.raises(BackendNotFound):
        adapter.install(project_cwd, scope="user")


def test_install_preserves_existing_mcp_servers(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing non-perfxpert entries in .mcp.json must survive."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    # Pre-existing config with another MCP server.
    mcp_config = project_cwd / ".mcp.json"
    mcp_config.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-bin", "args": []}}})
    )

    def _run(cmd, *args, **kwargs):
        class _R:
            returncode = 1  # force structured edit
            stdout = b""
            stderr = b""

        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        return _R()

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    adapter = ClaudeCodeAdapter()
    adapter.install(project_cwd)
    data = json.loads(mcp_config.read_text())
    assert data["mcpServers"]["other"]["command"] == "other-bin"
    assert data["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_install_raises_on_conflicting_entry(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ConfigClobber: an existing `perfxpert` entry with a different
    command must NOT be silently overwritten."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    mcp_config = project_cwd / ".mcp.json"
    mcp_config.write_text(
        json.dumps(
            {"mcpServers": {"perfxpert": {"command": "different-bin", "args": []}}}
        )
    )

    def _run(cmd, *args, **kwargs):
        class _R:
            returncode = 1  # force structured edit path
            stdout = b""
            stderr = b""

        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        return _R()

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    adapter = ClaudeCodeAdapter()
    with pytest.raises(ConfigClobber):
        adapter.install(project_cwd)


def test_install_adds_mcp_json_to_gitignore(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase-8 convenience: if .gitignore already exists, append .mcp.json."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    (project_cwd / ".gitignore").write_text("*.pyc\n")

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run",
        _fake_claude_subprocess_factory(),
    )

    ClaudeCodeAdapter().install(project_cwd)
    lines = (project_cwd / ".gitignore").read_text().splitlines()
    assert ".mcp.json" in lines


# ---------------------------------------------------------------------------
# install — git-tracked CLAUDE.local.md guardrail (I3).
# ---------------------------------------------------------------------------


def test_install_refuses_when_claude_md_tracked_without_flag(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I3: never touch a git-tracked CLAUDE.local.md without --allow flag."""
    # Create and commit a tracked CLAUDE.local.md at project root.
    tracked = project_cwd / "CLAUDE.local.md"
    tracked.write_text("user's own content\n")
    subprocess.run(["git", "add", "CLAUDE.local.md"], cwd=str(project_cwd), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=str(project_cwd), check=True
    )

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run",
        _fake_claude_subprocess_factory(),
    )

    adapter = ClaudeCodeAdapter()
    with pytest.raises(ConfigClobber, match="tracked"):
        adapter.install(project_cwd, allow_agents_md_append=False)


# ---------------------------------------------------------------------------
# install — consent.
# ---------------------------------------------------------------------------


def test_install_raises_consent_denied(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PERFXPERT_ASSUME_CONSENT", raising=False)
    # Force prompt to decline.
    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.prompt_consent_interactive",
        lambda *a, **kw: False,
    )
    adapter = ClaudeCodeAdapter()
    with pytest.raises(ConsentDenied):
        adapter.install(project_cwd)


# ---------------------------------------------------------------------------
# spawn.
# ---------------------------------------------------------------------------


def test_spawn_uses_execvpe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """I1 — assert we call `os.execvpe`, not `subprocess.run`."""
    called = {}

    def _fake_execvpe(name, argv, env):
        called["name"] = name
        called["argv"] = list(argv)
        called["env"] = dict(env)
        # execvpe would normally not return; raise to break out of the
        # test without actually replacing the process.
        raise RuntimeError("stopped")

    monkeypatch.setattr("os.execvpe", _fake_execvpe)
    monkeypatch.setattr("os.chdir", lambda _p: None)

    adapter = ClaudeCodeAdapter()
    with pytest.raises(RuntimeError, match="stopped"):
        adapter.spawn(["hello"], {"K": "V"}, tmp_path)

    assert called["name"] == "claude"
    assert called["argv"] == ["claude", "hello"]
    assert called["env"] == {"K": "V"}


# ---------------------------------------------------------------------------
# uninstall — drift detection.
# ---------------------------------------------------------------------------


def test_uninstall_refuses_on_pointer_drift(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the pointer file was edited (no longer contains the sentinel),
    uninstall must list it in `skipped_due_to_drift`."""
    # Simulate a user-edited pointer.
    pointer = project_cwd / ".claude" / "CLAUDE.md"
    pointer.parent.mkdir()
    pointer.write_text("user removed the perfxpert reference\n")

    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = ClaudeCodeAdapter()
    report = adapter.uninstall(project_cwd)
    assert isinstance(report, UninstallReport)
    assert pointer in report.skipped_due_to_drift


def test_uninstall_removes_sentinel_pointer(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: sentinel present → pointer removed."""
    pointer = project_cwd / ".claude" / "CLAUDE.md"
    pointer.parent.mkdir()
    pointer.write_text(
        "<!-- perfxpert-managed pointer file. -->\n@.perfxpert/AGENTS.md\n"
    )
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = ClaudeCodeAdapter()
    report = adapter.uninstall(project_cwd)
    assert pointer in report.paths_removed
    assert not pointer.exists()
