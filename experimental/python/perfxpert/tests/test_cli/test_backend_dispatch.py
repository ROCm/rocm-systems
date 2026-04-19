"""Tests for `perfxpert.cli._backend_dispatch` (Task 2).

Covers:

* `route_subcommand` recognizes `claude` / `codex` / `gemini` as
  `"backend"` kinds.
* The stub dispatcher returns rc=42 until real adapters land (Tasks
  4b/5/10).
* Recursion guard refuses when `PERFXPERT_IN_AGENT_SESSION` is set
  (unless `--force`).
* The bare `perfxpert-code` path (no args) still reaches opencode —
  regression guard for the default behavior.
"""

from __future__ import annotations

import pytest

from perfxpert.cli import _backend_dispatch, opencode_launcher
from perfxpert.cli._backend_dispatch import RECURSION_GUARD_ENV, is_help_request
from perfxpert.cli.opencode_launcher import main, route_subcommand


# ---------------------------------------------------------------------------
# route_subcommand — third-party backend recognition.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["claude", "codex", "gemini"])
def test_route_recognizes_backend(name: str) -> None:
    kind, out = route_subcommand([name])
    assert kind == "backend"
    assert out == [name]


@pytest.mark.parametrize("name", ["claude", "codex", "gemini"])
def test_route_backend_with_args(name: str) -> None:
    kind, out = route_subcommand([name, "hello", "world"])
    assert kind == "backend"
    assert out == [name, "hello", "world"]


def test_route_backend_with_leading_flag() -> None:
    """`perfxpert-code --verbose claude hello` — flags skipped, claude found."""
    kind, out = route_subcommand(["--verbose", "claude", "hello"])
    assert kind == "backend"
    assert out == ["--verbose", "claude", "hello"]


def test_route_default_unchanged_for_empty_argv() -> None:
    """Regression: bare `perfxpert-code` still goes to opencode_default."""
    kind, out = route_subcommand([])
    assert kind == "opencode_default"
    assert out == []


def test_route_doctor_still_perfxpert_owned() -> None:
    """Regression: adding backend routing did not break existing dispatch."""
    kind, _out = route_subcommand(["doctor"])
    assert kind == "perfxpert"


# ---------------------------------------------------------------------------
# Stub dispatcher — returns rc=42, prints helpful message.
# ---------------------------------------------------------------------------


def test_codex_stub_returns_42(capsys, monkeypatch) -> None:
    """Codex ships in PR 2 (Task 10); in PR 1 its stub returns 42."""
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)
    rc = _backend_dispatch._exec_backend("codex", [])
    assert rc == 42
    err = capsys.readouterr().err
    assert "codex" in err
    assert "PR 2" in err


@pytest.mark.parametrize("name", ["claude", "gemini"])
def test_real_adapters_registered_for_claude_and_gemini(
    name: str, monkeypatch
) -> None:
    """Task 6: real adapter runners replace the Task-2 stubs for
    claude + gemini. Invoking the handler reaches the install flow
    (we don't verify the full install here — other test files do — but
    we DO verify the handler is not the stub-42 path)."""
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)
    # Replace the adapter's install method with a sentinel so we
    # short-circuit without real file writes.
    import perfxpert.cli._backend.claude as claude_mod
    import perfxpert.cli._backend.gemini as gemini_mod

    calls = {"install": 0}

    def _fake_install(self, cwd, **kw):
        calls["install"] += 1
        from perfxpert.cli._backend.protocol import InstallReport

        return InstallReport(backend=self.name)

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "install", _fake_install)
    monkeypatch.setattr(gemini_mod.GeminiAdapter, "install", _fake_install)

    # Also stub spawn so we don't try to execvpe.
    def _fake_spawn(self, argv, env, cwd):
        return 0

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "spawn", _fake_spawn)
    monkeypatch.setattr(gemini_mod.GeminiAdapter, "spawn", _fake_spawn)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = _backend_dispatch._exec_backend(name, ["--quiet", "hello"])
    assert rc == 0
    assert calls["install"] == 1


# ---------------------------------------------------------------------------
# Recursion guard.
# ---------------------------------------------------------------------------


def test_recursion_guard_refuses_when_env_set(
    capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RECURSION_GUARD_ENV, "claude")
    rc = _backend_dispatch._exec_backend("claude", [])
    assert rc == 3
    err = capsys.readouterr().err
    assert "already inside" in err
    assert "claude" in err


def test_recursion_guard_force_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--force` in the argv bypasses the recursion refusal."""
    monkeypatch.setenv(RECURSION_GUARD_ENV, "claude")
    # codex stub returns 42 — use it to verify the guard is bypassed.
    rc = _backend_dispatch._exec_backend("codex", ["--force", "hello"])
    assert rc == 42


def test_recursion_guard_empty_env_does_not_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)
    # codex stub returns 42 when guard permits dispatch.
    rc = _backend_dispatch._exec_backend("codex", [])
    assert rc == 42


# ---------------------------------------------------------------------------
# is_help_request helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_is_help_request_true_for_leading_flag(flag: str) -> None:
    assert is_help_request([flag]) is True
    assert is_help_request([flag, "extra"]) is True


def test_is_help_request_false_when_flag_not_first() -> None:
    assert is_help_request(["hello", "--help"]) is False


def test_is_help_request_false_for_empty() -> None:
    assert is_help_request([]) is False


# ---------------------------------------------------------------------------
# main() end-to-end — backend dispatch path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["claude", "codex", "gemini"])
def test_main_dispatches_backend_to_stub(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`perfxpert-code <backend>` reaches _exec_backend (not opencode)."""
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)
    calls: list[tuple[str, list[str]]] = []

    def _fake_exec_backend(backend_name: str, remaining_argv: list[str]) -> int:
        calls.append((backend_name, remaining_argv))
        return 0

    monkeypatch.setattr(
        "perfxpert.cli._backend_dispatch._exec_backend", _fake_exec_backend
    )
    rc = main([name, "arg1", "arg2"])
    assert rc == 0
    assert calls == [(name, ["arg1", "arg2"])]


def test_main_backend_path_does_not_resolve_opencode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short-circuit: backend dispatch must not require an opencode binary."""
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)

    def _fake_exec_backend(_name, _argv):
        return 0

    monkeypatch.setattr(
        "perfxpert.cli._backend_dispatch._exec_backend", _fake_exec_backend
    )

    def _no_binary():
        raise AssertionError("resolve_opencode_binary should NOT be called")

    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", _no_binary)

    # If resolve_opencode_binary was called, _no_binary would AssertionError
    # which pytest surfaces as a test failure, so a clean rc=0 proves
    # dispatch short-circuits before binary resolution.
    rc = main(["claude", "hello"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Dispatcher flag parser (Task 6).
# ---------------------------------------------------------------------------


def test_parse_dispatcher_flags_empty() -> None:
    f = _backend_dispatch.parse_dispatcher_flags([])
    assert f.dry_run is False
    assert f.quiet is False
    assert f.force is False
    assert f.allow_agents_md_append is False
    assert f.remaining == []


def test_parse_dispatcher_flags_all() -> None:
    f = _backend_dispatch.parse_dispatcher_flags(
        ["--dry-run", "--quiet", "--force", "--allow-agents-md-append", "hello"]
    )
    assert f.dry_run is True
    assert f.quiet is True
    assert f.force is True
    assert f.allow_agents_md_append is True
    assert f.remaining == ["hello"]


def test_parse_dispatcher_flags_stops_at_first_non_flag() -> None:
    """Anything after the first non-dispatcher token stays intact for the backend."""
    f = _backend_dispatch.parse_dispatcher_flags(
        ["--quiet", "hello", "--dry-run"]
    )
    assert f.quiet is True
    assert f.remaining == ["hello", "--dry-run"]


def test_dry_run_does_not_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--dry-run` → install(dry_run=True); skip spawn."""
    import perfxpert.cli._backend.claude as claude_mod

    install_calls: list = []
    spawn_calls: list = []

    def _fake_install(self, cwd, **kw):
        install_calls.append(kw)
        from perfxpert.cli._backend.protocol import InstallReport

        return InstallReport(backend=self.name)

    def _fake_spawn(self, argv, env, cwd):
        spawn_calls.append(argv)
        return 0

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "install", _fake_install)
    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "spawn", _fake_spawn)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)

    rc = _backend_dispatch._exec_backend("claude", ["--dry-run", "hello"])
    assert rc == 0
    assert len(install_calls) == 1
    assert install_calls[0]["dry_run"] is True
    assert spawn_calls == []  # --dry-run means NO spawn


def test_quiet_flag_forwarded_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import perfxpert.cli._backend.claude as claude_mod

    captured = {}

    def _fake_install(self, cwd, **kw):
        captured.update(kw)
        from perfxpert.cli._backend.protocol import InstallReport

        return InstallReport(backend=self.name)

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "install", _fake_install)
    monkeypatch.setattr(
        claude_mod.ClaudeCodeAdapter, "spawn", lambda self, a, e, c: 0
    )
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)

    _backend_dispatch._exec_backend("claude", ["--quiet", "--dry-run"])
    assert captured["quiet"] is True


def test_allow_agents_md_append_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    import perfxpert.cli._backend.claude as claude_mod

    captured = {}

    def _fake_install(self, cwd, **kw):
        captured.update(kw)
        from perfxpert.cli._backend.protocol import InstallReport

        return InstallReport(backend=self.name)

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "install", _fake_install)
    monkeypatch.setattr(
        claude_mod.ClaudeCodeAdapter, "spawn", lambda self, a, e, c: 0
    )
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)

    _backend_dispatch._exec_backend(
        "claude", ["--allow-agents-md-append", "--dry-run"]
    )
    assert captured["allow_agents_md_append"] is True


def test_recursion_guard_env_set_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child's env MUST have PERFXPERT_IN_AGENT_SESSION=<backend>."""
    import perfxpert.cli._backend.claude as claude_mod

    captured_env: dict = {}

    def _fake_install(self, cwd, **kw):
        from perfxpert.cli._backend.protocol import InstallReport

        return InstallReport(backend=self.name)

    def _fake_spawn(self, argv, env, cwd):
        captured_env.update(env)
        return 0

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "install", _fake_install)
    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "spawn", _fake_spawn)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)

    _backend_dispatch._exec_backend("claude", ["--quiet", "hello"])
    assert captured_env.get(RECURSION_GUARD_ENV) == "claude"


def test_help_passthrough_does_not_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`perfxpert-code claude --help` must NOT run the installer."""
    import perfxpert.cli._backend.claude as claude_mod

    install_calls = []

    def _fake_install(self, cwd, **kw):
        install_calls.append(kw)
        from perfxpert.cli._backend.protocol import InstallReport

        return InstallReport(backend=self.name)

    def _fake_spawn(self, argv, env, cwd):
        # spawn IS called for help passthrough.
        assert "--help" in argv
        return 0

    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "install", _fake_install)
    monkeypatch.setattr(claude_mod.ClaudeCodeAdapter, "spawn", _fake_spawn)
    monkeypatch.delenv(RECURSION_GUARD_ENV, raising=False)

    rc = _backend_dispatch._exec_backend("claude", ["--help"])
    assert rc == 0
    assert install_calls == []


def test_main_default_still_uses_opencode(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression: bare `perfxpert-code` still stages the runtime cfg dir
    and launches the opencode binary (Task 2 must not regress this path)."""
    fake_bin = tmp_path / "opencode"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    fake_cfg = tmp_path / "cfg"
    fake_cfg.mkdir()
    (fake_cfg / "opencode.json").write_text("{}")

    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_opencode_binary", lambda: fake_bin
    )
    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_config_dir", lambda: fake_cfg
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    class _FakeProc:
        returncode = 0

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeProc()

    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.subprocess.run", _fake_run
    )
    rc = main([])
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0][0] == str(fake_bin)
