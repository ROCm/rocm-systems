"""Tests for `perfxpert.cli._backend.codex.CodexAdapter` (Task 10, PR 2).

Covers:

* `check_available` — missing binary (BackendNotFound).
* `_check_trust` — parses TRUSTED / UNTRUSTED / UNKNOWN from
  `~/.codex/config.toml`.
* `install()` — trust-gate decision tree:
    - non-interactive + untrusted → TrustRequired.
    - PERFXPERT_AUTO_TRUST=1 → auto-marks trusted.
    - declined prompt → falls back to user scope with warning.
* Lazy import of `tomlkit` — the primary `codex mcp add` path must
  NEVER import tomlkit at module-load time (supersedes cycle-2 I7;
  commit 3547736829 made tomlkit a required dep but the import is
  still deferred to keep the primary path import-cost-free).
* Never mutates tracked `AGENTS.md`; uses a discovered `AGENTS.override.md`.
* `spawn()` uses `os.execvpe`, not `subprocess.run`.
* `uninstall()` removes only our managed `[mcp_servers.perfxpert]`
  block; refuses on marker drift (different `command` value).
* Mock-backend `verify_mcp_live()` happy-path + failure path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from perfxpert.cli._backend.codex import (
    AUTO_TRUST_ENV,
    SKIP_LIVE_CHECK_ENV,
    CodexAdapter,
    TrustStatus,
)
from perfxpert.cli._backend.protocol import (
    BackendAdapter,
    ConfigClobber,
    ConsentDenied,
    InstallReport,
    LiveCheckReport,
    Plan,
    PartialInstall,
    TrustRequired,
    UninstallReport,
)


# Snapshot real subprocess.run so fake _run helpers can still invoke git.
_REAL_RUN = subprocess.run


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME + XDG_CONFIG_HOME; neutralize consent prompt.

    NOTE: does NOT set PERFXPERT_AUTO_TRUST — tests that need auto-
    trust set it themselves so untrust tests stay honest.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setenv(SKIP_LIVE_CHECK_ENV, "1")
    # Belt-and-braces: reset any env the harness may have inherited.
    monkeypatch.delenv(AUTO_TRUST_ENV, raising=False)
    monkeypatch.delenv("PERFXPERT_GATE_HOOK", raising=False)
    return tmp_path


@pytest.fixture
def project_cwd(isolated_home: Path) -> Path:
    cwd = isolated_home / "proj"
    cwd.mkdir()
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"],
        cwd=str(cwd),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@e.com"], cwd=str(cwd), check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=str(cwd), check=True
    )
    return cwd


def _write_user_codex_config(home: Path, contents: str) -> Path:
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(contents)
    return cfg


def _mark_trusted(home: Path, cwd: Path) -> None:
    """Pre-populate the trusted-projects table for `cwd`."""
    resolved = str(cwd.expanduser().resolve())
    _write_user_codex_config(
        home,
        f'[projects."{resolved}"]\ntrust_level = "trusted"\n',
    )


def _fake_codex_subprocess(
    *,
    list_exit: int = 0,
    list_stdout: bytes = b"perfxpert\n",
    add_exit: int = 0,
    remove_exit: int = 0,
):
    """Return a `subprocess.run` replacement that fakes the codex CLI."""

    def _run(cmd, *args, **kwargs):
        class _R:
            def __init__(
                self, rc: int, stdout: bytes = b"", stderr: bytes = b""
            ) -> None:
                self.returncode = rc
                self.stdout = stdout
                self.stderr = stderr

        # Pass real git through unchanged.
        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)

        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "--version":
            return _R(0, stdout=b"codex 0.7.0\n")
        if isinstance(cmd, list) and "mcp" in cmd:
            verb = cmd[cmd.index("mcp") + 1]
            if verb == "list":
                return _R(list_exit, stdout=list_stdout)
            if verb == "add":
                return _R(add_exit)
            if verb == "remove":
                return _R(remove_exit)
        return _R(0)

    return _run


# ---------------------------------------------------------------------------
# Protocol conformance.
# ---------------------------------------------------------------------------


def test_adapter_conforms_to_protocol() -> None:
    assert isinstance(CodexAdapter(), BackendAdapter)


def test_tool_name_template() -> None:
    assert CodexAdapter.tool_name_template == "mcp_perfxpert_{tool}"


def test_spawn_strategy_is_execvpe() -> None:
    assert CodexAdapter.spawn_strategy == "execvpe"


# ---------------------------------------------------------------------------
# check_available.
# ---------------------------------------------------------------------------


def test_check_available_missing_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing codex binary → (False, install hint)."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    ok, reason = CodexAdapter().check_available()
    assert ok is False
    assert "not found on PATH" in reason
    assert "codex" in reason.lower()


def test_check_available_version_too_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")

    class _R:
        returncode = 0
        stdout = "codex 0.6.0\n"
        stderr = ""

    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        lambda *a, **kw: _R(),
    )
    ok, reason = CodexAdapter().check_available()
    assert ok is False
    assert "below" in reason


def test_check_available_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")

    class _R:
        returncode = 0
        stdout = "codex 0.7.0\n"
        stderr = ""

    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        lambda *a, **kw: _R(),
    )
    ok, _ = CodexAdapter().check_available()
    assert ok is True


# ---------------------------------------------------------------------------
# _check_trust — parses trusted / untrusted / unknown from config.toml.
# ---------------------------------------------------------------------------


def test_check_trust_parses_trusted_and_untrusted(
    isolated_home: Path, project_cwd: Path
) -> None:
    """I-N3 / plan B3: read `[projects."<cwd>"].trust_level`."""
    resolved = str(project_cwd.expanduser().resolve())
    adapter = CodexAdapter()

    # (a) No config file → UNKNOWN.
    assert adapter._check_trust(project_cwd) == TrustStatus.UNKNOWN

    # (b) Trusted.
    _write_user_codex_config(
        isolated_home,
        f'[projects."{resolved}"]\ntrust_level = "trusted"\n',
    )
    assert adapter._check_trust(project_cwd) == TrustStatus.TRUSTED

    # (c) Untrusted.
    _write_user_codex_config(
        isolated_home,
        f'[projects."{resolved}"]\ntrust_level = "untrusted"\n',
    )
    assert adapter._check_trust(project_cwd) == TrustStatus.UNTRUSTED

    # (d) Different project → UNKNOWN.
    other = isolated_home / "other"
    other.mkdir()
    assert adapter._check_trust(other) == TrustStatus.UNKNOWN


def test_check_trust_tolerates_invalid_toml(
    isolated_home: Path, project_cwd: Path
) -> None:
    """Malformed ~/.codex/config.toml → UNKNOWN (fail-soft)."""
    _write_user_codex_config(isolated_home, "}}}not toml{{{")
    assert (
        CodexAdapter()._check_trust(project_cwd) == TrustStatus.UNKNOWN
    )


# ---------------------------------------------------------------------------
# plan.
# ---------------------------------------------------------------------------


def test_plan_lists_targets(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = CodexAdapter().plan(project_cwd)
    assert isinstance(plan, Plan)
    assert plan.backend == "codex"
    # config.toml + Codex-discovered prompt file.
    target_names = {p.name for p in plan.targets}
    assert "config.toml" in target_names
    assert "AGENTS.override.md" in target_names
    joined = "\n".join(plan.actions).lower()
    assert "mcp_servers" in joined
    assert "trust" in joined


# ---------------------------------------------------------------------------
# install — trust gate decision tree.
# ---------------------------------------------------------------------------


def _make_stdin_tty(monkeypatch: pytest.MonkeyPatch, is_tty: bool) -> None:
    """Force `sys.stdin.isatty()` to return the requested value."""

    class _FakeStdin:
        def isatty(self) -> bool:
            return is_tty

        def readline(self) -> str:  # for the interactive-prompt path
            return "\n"

    monkeypatch.setattr(sys, "stdin", _FakeStdin())


def test_install_raises_trust_required_when_non_interactive_untrusted(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Task 10 invariant #1: untrusted + non-TTY + PERFXPERT_ASSUME_CONSENT
    not set → TrustRequired.

    Consent cache is pre-populated (via grant_consent) so we reach the
    trust-gate branch; then we clear PERFXPERT_ASSUME_CONSENT and
    assert the trust-gate raises.
    """
    from perfxpert.cli._consent import file_set_hash, grant_consent

    # Pre-grant consent so the consent stage passes WITHOUT the env
    # var. The trust-gate should then raise because neither consent-
    # env nor auto-trust is set.
    config_toml = project_cwd / ".codex" / "config.toml"
    agents = project_cwd / "AGENTS.override.md"
    grant_consent(
        "codex",
        project_cwd,
        file_set_hash(
            (
                (config_toml, config_toml.exists(), False),
                (agents, agents.exists(), False),
            )
        ),
    )
    monkeypatch.delenv("PERFXPERT_ASSUME_CONSENT", raising=False)
    monkeypatch.delenv(AUTO_TRUST_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)  # no binary spawn
    _make_stdin_tty(monkeypatch, False)
    # No [projects."..."] table → UNKNOWN trust status.
    with pytest.raises(TrustRequired, match="not marked trusted"):
        CodexAdapter().install(project_cwd, scope="project")


def test_install_auto_trusts_with_env_var(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """PERFXPERT_AUTO_TRUST=1 → mark trusted, proceed with project scope."""
    monkeypatch.setenv(AUTO_TRUST_ENV, "1")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    CodexAdapter().install(project_cwd, scope="project")

    # The user config should now list the cwd as trusted.
    cfg = isolated_home / ".codex" / "config.toml"
    assert cfg.is_file()
    text = cfg.read_text()
    resolved = str(project_cwd.expanduser().resolve())
    assert resolved in text
    assert 'trust_level = "trusted"' in text


def test_install_falls_back_to_user_scope_on_still_untrusted(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Interactive + user declines trust → falls back to user scope.

    CONSENT is pre-granted (via PERFXPERT_ASSUME_CONSENT=1 from the
    fixture) so the trust-gate interactive prompt is what actually
    runs in this test.
    """
    # Interactive TTY that answers "n" to the trust-gate prompt.
    class _FakeStdin:
        def isatty(self) -> bool:
            return True

        def readline(self) -> str:
            return "n\n"

    monkeypatch.setattr(sys, "stdin", _FakeStdin())

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    report = CodexAdapter().install(project_cwd, scope="project")
    # Resolved scope should be `user` after the user declined trust.
    joined = "\n".join(report.actions).lower()
    assert "user" in joined
    # No project-scope .codex/config.toml should have been written.
    project_cfg = project_cwd / ".codex" / "config.toml"
    # If tomlkit fallback fired, the project cfg would exist.
    # On the happy codex-mcp-add path the shell-out writes to user cfg.
    assert not project_cfg.exists() or project_cfg.read_text().strip() == ""


def test_install_reprompts_consent_after_scope_falls_back_to_user(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    prompts: list[list[str]] = []
    answers = iter([True, False])

    class _FakeStdin:
        def isatty(self) -> bool:
            return True

        def readline(self) -> str:
            return "n\n"

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    monkeypatch.delenv("PERFXPERT_ASSUME_CONSENT", raising=False)
    def _fake_prompt(*args, **kwargs):
        plan_lines = kwargs.get("plan_lines")
        if plan_lines is None and len(args) >= 3:
            plan_lines = args[2]
        prompts.append(list(plan_lines))
        return next(answers)

    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.prompt_consent_interactive",
        _fake_prompt,
    )

    with pytest.raises(ConsentDenied):
        CodexAdapter().install(project_cwd, scope="project")

    assert len(prompts) == 2
    assert str(project_cwd / ".codex" / "config.toml") in prompts[0][0]
    assert str(isolated_home / ".codex" / "config.toml") in prompts[1][0]
    assert not (project_cwd / "AGENTS.override.md").exists()


# ---------------------------------------------------------------------------
# Lazy tomlkit import (cycle-2 I7).
# ---------------------------------------------------------------------------


def test_install_uses_tomlkit_lazy_import_in_fallback(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """The primary `codex mcp add` path must NOT import tomlkit.

    We pre-trust the project to skip the trust-gate write (which uses
    tomlkit), then pin tomlkit to a sentinel that raises if imported
    during install. The test passes iff `codex mcp add` succeeds
    without any tomlkit import.
    """
    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(list_stdout=b""),  # empty list → add runs
    )

    # Temporarily mask tomlkit: any import attempt raises.
    import sys as _sys

    original = _sys.modules.get("tomlkit")

    class _Sentinel:
        def __getattr__(self, name: str):
            raise RuntimeError(
                f"tomlkit should NOT be imported on the primary "
                f"`codex mcp add` path (attribute access: {name!r})"
            )

    # Insert a sentinel that claims to be tomlkit; if any code does
    # `import tomlkit` + accesses an attr, we blow up loudly.
    _sys.modules["tomlkit"] = _Sentinel()  # type: ignore[assignment]
    try:
        CodexAdapter().install(project_cwd, scope="project")
    finally:
        # Restore original tomlkit binding (or clear).
        if original is None:
            _sys.modules.pop("tomlkit", None)
        else:
            _sys.modules["tomlkit"] = original


def test_structured_edit_fallback_uses_tomlkit(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """When `codex mcp add` fails, the adapter falls back to a
    lazy-imported tomlkit edit of config.toml."""
    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    # codex mcp add returns non-zero → fallback tomlkit edit runs.
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(add_exit=1, list_stdout=b""),
    )

    CodexAdapter().install(project_cwd, scope="project")
    # Project-scope config.toml should exist + contain mcp_servers.perfxpert.
    cfg = project_cwd / ".codex" / "config.toml"
    assert cfg.is_file()
    text = cfg.read_text()
    assert "[mcp_servers.perfxpert]" in text or "perfxpert" in text
    assert "perfxpert-mcp" in text


# ---------------------------------------------------------------------------
# Never mutates tracked AGENTS.md.
# ---------------------------------------------------------------------------


def test_install_never_touches_tracked_agents_md(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """I3: even if the user has a committed AGENTS.md, install must
    leave the committed file byte-identical and write a discovered
    `AGENTS.override.md` that shadows it safely for Codex."""
    tracked = project_cwd / "AGENTS.md"
    tracked.write_text("USER CONTENT — do not touch\n")
    subprocess.run(
        ["git", "add", "AGENTS.md"], cwd=str(project_cwd), check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=str(project_cwd), check=True
    )

    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    CodexAdapter().install(project_cwd, scope="project")

    # Tracked AGENTS.md unchanged.
    assert tracked.read_text() == "USER CONTENT — do not touch\n"
    # Codex-discovered override staged at project root.
    override = project_cwd / "AGENTS.override.md"
    assert override.is_file()
    override_text = override.read_text()
    assert "USER CONTENT — do not touch" in override_text
    assert "<!-- BEGIN perfxpert-managed v1 cache=" in override_text


def test_install_refreshes_generated_shadow_copy_on_rerun(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    tracked = project_cwd / "AGENTS.md"
    tracked.write_text("BASE v1\n")

    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    adapter = CodexAdapter()
    adapter.install(project_cwd, scope="project")
    tracked.write_text("BASE v2\n")
    adapter.install(project_cwd, scope="project")

    override = project_cwd / "AGENTS.override.md"
    text = override.read_text()
    assert "BASE v2" in text
    assert "BASE v1" not in text


def test_install_refuses_recreating_tracked_deleted_override_without_flag(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    override = project_cwd / "AGENTS.override.md"
    override.write_text("tracked override\n")
    subprocess.run(
        ["git", "add", "AGENTS.override.md"],
        cwd=str(project_cwd),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "track override"],
        cwd=str(project_cwd),
        check=True,
    )
    override.unlink()

    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    with pytest.raises(ConfigClobber, match="tracked in git"):
        CodexAdapter().install(project_cwd, scope="project")


def test_install_allows_recreating_tracked_deleted_override_with_flag(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    override = project_cwd / "AGENTS.override.md"
    override.write_text("tracked override\n")
    subprocess.run(
        ["git", "add", "AGENTS.override.md"],
        cwd=str(project_cwd),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "track override"],
        cwd=str(project_cwd),
        check=True,
    )
    override.unlink()

    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    CodexAdapter().install(
        project_cwd,
        scope="project",
        allow_agents_md_append=True,
    )
    assert override.exists()


# ---------------------------------------------------------------------------
# spawn.
# ---------------------------------------------------------------------------


def test_spawn_uses_execvpe_not_subprocess_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """I1: codex is a TUI — spawn must use os.execvpe."""
    called: dict = {}

    def _fake_execvpe(name, argv, env):
        called["name"] = name
        called["argv"] = list(argv)
        called["env"] = dict(env)
        raise RuntimeError("stopped")

    monkeypatch.setattr("os.execvpe", _fake_execvpe)
    monkeypatch.setattr("os.chdir", lambda _p: None)

    adapter = CodexAdapter()
    with pytest.raises(RuntimeError, match="stopped"):
        adapter.spawn(["hello"], {"K": "V"}, tmp_path)

    assert called["name"] == "codex"
    assert called["argv"] == ["codex", "hello"]
    assert called["env"] == {"K": "V"}


# ---------------------------------------------------------------------------
# uninstall — drift detection.
# ---------------------------------------------------------------------------


def test_uninstall_removes_only_managed_block(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """uninstall removes `[mcp_servers.perfxpert]` but preserves other MCP entries."""
    _mark_trusted(isolated_home, project_cwd)
    # Pre-populate a config.toml with TWO MCP servers.
    cfg = project_cwd / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[mcp_servers.other]\n'
        'command = "other-bin"\n'
        'args = []\n'
        '\n'
        '[mcp_servers.perfxpert]\n'
        'command = "perfxpert-mcp"\n'
        'args = []\n'
        'enabled = true\n'
    )

    # Shell-out path absent → use tomlkit fallback.
    monkeypatch.setattr("shutil.which", lambda _: None)

    report = CodexAdapter().uninstall(project_cwd, scope="project")
    assert isinstance(report, UninstallReport)

    text = cfg.read_text()
    assert "[mcp_servers.other]" in text
    assert "other-bin" in text
    assert "[mcp_servers.perfxpert]" not in text


def test_uninstall_refuses_on_marker_drift(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """If `perfxpert` table points at a DIFFERENT command, refuse to remove it."""
    cfg = project_cwd / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[mcp_servers.perfxpert]\n'
        'command = "NOT-ours"\n'
        'args = []\n'
    )

    monkeypatch.setattr("shutil.which", lambda _: None)

    report = CodexAdapter().uninstall(project_cwd, scope="project")
    # The entry stays; the path is recorded as drift.
    text = cfg.read_text()
    assert 'command = "NOT-ours"' in text
    assert cfg in report.skipped_due_to_drift


# ---------------------------------------------------------------------------
# verify_mcp_live — mock-backend probes.
# ---------------------------------------------------------------------------


def test_verify_mcp_live_happy_path(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex mcp list output contains 'perfxpert' → mcp_healthy=True."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(list_stdout=b"perfxpert\nother\n"),
    )
    report = CodexAdapter().verify_mcp_live(project_cwd)
    assert isinstance(report, LiveCheckReport)
    assert report.mcp_listed is True
    assert report.mcp_healthy is True
    # Codex hook surface is prompt-layer-only on current Codex — False
    # is the documented-known-limit state, not a failure.
    assert report.gate_hook_installed is False


def test_verify_mcp_live_missing_perfxpert(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No perfxpert token in `codex mcp list` → error surfaced."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(list_stdout=b"only-other-server\n"),
    )
    # Tight retry budget so the test doesn't wait 14s on exponential
    # backoff.
    monkeypatch.setenv("PERFXPERT_MCP_RETRY_BUDGET_S", "0.1")

    report = CodexAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "perfxpert" in (report.error or "").lower()


def test_verify_mcp_live_binary_missing(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex not on PATH → mcp_healthy=False with binary-missing error."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    report = CodexAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "PATH" in (report.error or "")


# ---------------------------------------------------------------------------
# Idempotency + clobber.
# ---------------------------------------------------------------------------


def test_install_is_idempotent_when_mcp_list_already_has_perfxpert(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """If `codex mcp list` already shows perfxpert, skip the add."""
    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")

    calls: list[list] = []

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        calls.append(list(cmd))

        class _R:
            returncode = 0
            stdout = b"perfxpert\n"
            stderr = b""

        return _R()

    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run", _run
    )
    CodexAdapter().install(project_cwd, scope="project")
    # `mcp add` must NOT appear among the calls once listing already
    # shows perfxpert.
    add_calls = [c for c in calls if "add" in c and "perfxpert" in c]
    assert not add_calls, (
        f"expected no `codex mcp add` call on idempotent path; got {calls}"
    )


def test_install_raises_config_clobber_on_conflicting_entry(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Existing perfxpert entry pointing at a different command →
    ConfigClobber raised from the fallback tomlkit edit."""
    _mark_trusted(isolated_home, project_cwd)
    cfg = project_cwd / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[mcp_servers.perfxpert]\n'
        'command = "someone-elses-mcp"\n'
        'args = []\n'
    )
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    # Force fallback path: `codex mcp add` fails.
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(add_exit=1, list_stdout=b""),
    )

    with pytest.raises(ConfigClobber):
        CodexAdapter().install(project_cwd, scope="project")


# ---------------------------------------------------------------------------
# 32 KiB cap precheck.
# ---------------------------------------------------------------------------


def test_install_rejects_oversized_prompt(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Rendered prompt > 32 KiB → PartialInstall (plan practical §3.4)."""
    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )
    # Monkey-patch _render_prompt_for_codex to return a 33 KiB string.
    oversize = "A" * (33 * 1024)
    monkeypatch.setattr(
        CodexAdapter, "_render_prompt_for_codex", lambda self: oversize
    )
    from perfxpert.cli._backend.protocol import PartialInstall

    with pytest.raises(PartialInstall, match="32 KiB"):
        CodexAdapter().install(project_cwd, scope="project")


def test_install_rejects_oversized_shadow_copy_prompt(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    _mark_trusted(isolated_home, project_cwd)
    (project_cwd / "AGENTS.md").write_text("A" * (33 * 1024))
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    with pytest.raises(PartialInstall, match="32 KiB"):
        CodexAdapter().install(project_cwd, scope="project")


# ---------------------------------------------------------------------------
# Sanity: plan / install preserve duration_s + report shape.
# ---------------------------------------------------------------------------


def test_install_report_shape(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )
    r = CodexAdapter().install(project_cwd, scope="project")
    assert isinstance(r, InstallReport)
    assert r.backend == "codex"
    assert r.duration_s >= 0.0
    # MCP config path + discovered prompt file present.
    names = {p.name for p in r.paths_written}
    assert "config.toml" in names
    assert "AGENTS.override.md" in names


# ---------------------------------------------------------------------------
# Finding #2: refuse git-tracked ~/.codex/config.toml (dotfiles-style repos).
# ---------------------------------------------------------------------------


def test_mark_trusted_refuses_git_tracked_config(
    isolated_home: Path, project_cwd: Path
) -> None:
    """Dotfiles-style: `~/.codex/config.toml` is git-tracked → ConfigClobber.

    Initialise a repo at $HOME, commit an empty ~/.codex/config.toml, then
    ask `_mark_trusted` to rewrite it. The adapter must refuse rather
    than silently overwrite a versioned file.
    """
    cfg = isolated_home / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")  # empty but present

    # Make $HOME a git repo, track the file.
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"],
        cwd=str(isolated_home),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@e.com"],
        cwd=str(isolated_home),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=str(isolated_home),
        check=True,
    )
    subprocess.run(
        ["git", "add", ".codex/config.toml"],
        cwd=str(isolated_home),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=str(isolated_home),
        check=True,
    )

    with pytest.raises(ConfigClobber, match="tracked in a git repository"):
        CodexAdapter()._mark_trusted(project_cwd, home=isolated_home)


def test_structured_edit_refuses_git_tracked_project_config(
    isolated_home: Path, project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project-scope `.codex/config.toml` is git-tracked → ConfigClobber.

    The project_cwd fixture already initialises a git repo. Add and
    commit a `.codex/config.toml`, then force the install to fall
    back to the structured TOML edit by making `codex mcp add` fail.
    The fallback must refuse the tracked file.
    """
    _mark_trusted(isolated_home, project_cwd)

    # Pre-create + commit a .codex/config.toml in the project repo.
    cfg = project_cwd / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("# project-scoped codex config\n")
    subprocess.run(
        ["git", "add", ".codex/config.toml"],
        cwd=str(project_cwd),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "add codex config"],
        cwd=str(project_cwd),
        check=True,
    )

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    # `codex mcp add` returns non-zero → fallback structured edit runs.
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(add_exit=1, list_stdout=b""),
    )

    with pytest.raises(ConfigClobber, match="tracked in a git repository"):
        CodexAdapter().install(project_cwd, scope="project")


# ---------------------------------------------------------------------------
# Finding #3: ConsentDenied leaves ~/.codex/ untouched.
# ---------------------------------------------------------------------------


def test_install_consent_denied_leaves_codex_home_untouched(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Declined consent prompt → ConsentDenied, and nothing under
    ``~/.codex/`` is created or modified.

    The isolated_home fixture sets PERFXPERT_ASSUME_CONSENT=1 — we
    clear it here so the prompt actually runs, then patch
    prompt_consent_interactive to return False. Must raise
    ConsentDenied and must NOT create ``~/.codex/``.
    """
    monkeypatch.delenv("PERFXPERT_ASSUME_CONSENT", raising=False)
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.prompt_consent_interactive",
        lambda *a, **kw: False,
    )
    # No binary → adapter would have taken the tomlkit fallback, but we
    # never reach it because consent is denied first.
    monkeypatch.setattr("shutil.which", lambda _: None)

    codex_home = isolated_home / ".codex"
    assert not codex_home.exists()  # baseline

    with pytest.raises(ConsentDenied, match="declined"):
        CodexAdapter().install(project_cwd, scope="project")

    # Nothing under ~/.codex/ was created.
    assert not codex_home.exists(), (
        f"ConsentDenied path must not create ~/.codex/; found: "
        f"{list(codex_home.rglob('*')) if codex_home.exists() else []}"
    )
    # Project-scope .codex/ also must not be created.
    assert not (project_cwd / ".codex").exists()
    assert not (project_cwd / ".perfxpert").exists()
    assert not (project_cwd / "AGENTS.override.md").exists()


# ---------------------------------------------------------------------------
# Finding #4: malformed TOML raises a clear user-facing error
# (not a raw tomllib.TOMLDecodeError traceback).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_toml",
    [
        "[bad\nno_close_bracket = true\n",   # unclosed table header
        "bare_key_no_equals\n",              # missing `=`
        '[projects."x"]\ntrust_level = \n',  # value missing
    ],
    ids=["unclosed-bracket", "bare-key", "missing-value"],
)
def test_mark_trusted_on_malformed_toml_raises_config_clobber(
    isolated_home: Path, project_cwd: Path, bad_toml: str
) -> None:
    """Malformed ~/.codex/config.toml → ConfigClobber with a clear
    message, not a raw tomllib/tomlkit traceback."""
    _write_user_codex_config(isolated_home, bad_toml)

    with pytest.raises(ConfigClobber) as excinfo:
        CodexAdapter()._mark_trusted(project_cwd, home=isolated_home)

    msg = str(excinfo.value)
    assert "not valid TOML" in msg
    assert "config.toml" in msg
    # Must not surface the internal exception class name.
    assert "TOMLDecodeError" not in msg
    assert "Traceback" not in msg


def test_structured_remove_on_malformed_toml_raises_config_clobber(
    isolated_home: Path, project_cwd: Path
) -> None:
    """Malformed project-scope config.toml → _structured_remove raises
    ConfigClobber (caller translates to drift + action message)."""
    cfg = project_cwd / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("[mcp_servers.perfxpert\ncommand = \"perfxpert-mcp\"\n")

    drifted: list[Path] = []
    with pytest.raises(ConfigClobber) as excinfo:
        CodexAdapter()._structured_remove(cfg, drifted)

    msg = str(excinfo.value)
    assert "not valid TOML" in msg
    assert str(cfg) in msg
    assert "TOMLDecodeError" not in msg


# ---------------------------------------------------------------------------
# Finding #5: PERFXPERT_AUTO_TRUST warning bypasses --quiet.
# ---------------------------------------------------------------------------


def test_auto_trust_warning_prints_even_with_quiet(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Auto-trust is a security decision; the warning must print to
    stderr unconditionally — NOT be suppressed by `--quiet`."""
    monkeypatch.setenv(AUTO_TRUST_ENV, "1")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    CodexAdapter().install(project_cwd, scope="project", quiet=True)

    err = capsys.readouterr().err
    # Warning is present and explicit about the bypass.
    assert AUTO_TRUST_ENV in err
    assert "trust" in err.lower()
    assert "[WARN]" in err or "bypass" in err.lower()


def test_uninstall_on_malformed_project_config_records_drift_without_traceback(
    project_cwd: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a user-edited malformed config.toml at uninstall time
    lands in `skipped_due_to_drift` with a clear action message — no
    raw traceback escapes to the caller."""
    cfg = project_cwd / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("[bad no close bracket\nkey = 1\n")

    monkeypatch.setattr("shutil.which", lambda _: None)  # tomlkit fallback

    report = CodexAdapter().uninstall(project_cwd, scope="project")
    assert cfg in report.skipped_due_to_drift
    joined = "\n".join(report.actions)
    assert "not valid TOML" in joined
    assert "refused to edit" in joined


def test_install_appends_managed_block_to_existing_override(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Existing project override stays intact; perfxpert appends only its block."""
    _mark_trusted(isolated_home, project_cwd)
    override = project_cwd / "AGENTS.override.md"
    override.write_text("EXISTING OVERRIDE\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    CodexAdapter().install(project_cwd, scope="project")

    text = override.read_text()
    assert "EXISTING OVERRIDE" in text
    assert "<!-- BEGIN perfxpert-managed v1 cache=" in text


def test_uninstall_removes_only_perfxpert_block_from_existing_override(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Uninstall preserves unrelated AGENTS.override.md content."""
    _mark_trusted(isolated_home, project_cwd)
    override = project_cwd / "AGENTS.override.md"
    override.write_text("EXISTING OVERRIDE\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )
    CodexAdapter().install(project_cwd, scope="project")

    report = CodexAdapter().uninstall(project_cwd, scope="project")

    assert isinstance(report, UninstallReport)
    assert override.is_file()
    text = override.read_text()
    assert text == "EXISTING OVERRIDE\n"


def test_uninstall_removes_generated_override_and_legacy_cache(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    """Generated AGENTS.override.md is removed entirely on uninstall.

    Legacy hidden caches from older installs are removed too.
    """
    _mark_trusted(isolated_home, project_cwd)
    tracked = project_cwd / "AGENTS.md"
    tracked.write_text("ROOT AGENTS\n")
    legacy = project_cwd / ".perfxpert" / "AGENTS.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy prompt cache\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )
    CodexAdapter().install(project_cwd, scope="project")

    report = CodexAdapter().uninstall(project_cwd, scope="project")

    assert isinstance(report, UninstallReport)
    assert not (project_cwd / "AGENTS.override.md").exists()
    assert not legacy.exists()
    assert tracked.read_text() == "ROOT AGENTS\n"


def test_uninstall_preserves_user_edited_generated_override(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    _mark_trusted(isolated_home, project_cwd)
    tracked = project_cwd / "AGENTS.md"
    tracked.write_text("ROOT AGENTS\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )
    adapter = CodexAdapter()
    adapter.install(project_cwd, scope="project")

    override = project_cwd / "AGENTS.override.md"
    override.write_text(override.read_text().replace("ROOT AGENTS", "ROOT AGENTS\nUSER EDIT"))

    report = adapter.uninstall(project_cwd, scope="project")

    assert isinstance(report, UninstallReport)
    assert override.exists()
    text = override.read_text()
    assert "USER EDIT" in text
    assert "<!-- BEGIN perfxpert-managed v1 cache=" not in text


def test_uninstall_removes_project_trust_entry(
    project_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
) -> None:
    _mark_trusted(isolated_home, project_cwd)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    adapter = CodexAdapter()
    adapter.install(project_cwd, scope="project")
    adapter.uninstall(project_cwd, scope="project")

    cfg = isolated_home / ".codex" / "config.toml"
    assert str(project_cwd.resolve()) not in cfg.read_text()


def test_split_managed_prompt_block_ignores_shadow_copy_end_marker_text() -> None:
    adapter = CodexAdapter()
    managed = adapter._make_managed_prompt_block("PROMPT")
    text = adapter._make_shadow_copy_prompt(
        "before\n<!-- END perfxpert-managed v1 -->\nafter\n",
        managed,
    )

    before, after = adapter._split_managed_prompt_block(text)

    assert "<!-- END perfxpert-managed v1 -->" in before
    assert "PROMPT" not in before
    assert after.strip() == ""
