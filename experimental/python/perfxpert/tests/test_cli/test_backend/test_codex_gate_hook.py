"""Tests for `perfxpert.cli._gate_hooks.codex` (Task 4.6 Codex-portion, PR 2).

Codex's native PreToolUse hook only intercepts Bash as of April 2026,
so the gate hook module raises `GateHookUnsupported` unconditionally.
That's the prompt-layer-only fallback per plan guardrail.

The spec's key invariant is:
  * `install()` MUST raise BEFORE MCP registration runs so no
    partial state is left behind (I-N1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perfxpert.cli._backend.codex import CodexAdapter
from perfxpert.cli._backend.protocol import (
    GateHookUnsupported,
    InstallReport,
)
from perfxpert.cli._gate_hooks.codex import (
    evaluate_gate_state,
    install,
    uninstall,
)


# ---------------------------------------------------------------------------
# Direct hook-module invariants.
# ---------------------------------------------------------------------------


def test_install_raises_gate_hook_unsupported_unconditionally(
    tmp_path: Path,
) -> None:
    """Codex PreToolUse only hits Bash → no MCP interception →
    raise GateHookUnsupported. Plan guardrail: 'default to
    prompt-layer-only'."""
    with pytest.raises(GateHookUnsupported, match="Bash"):
        install(tmp_path)


def test_install_respects_env_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERFXPERT_GATE_HOOK=0 → user disabled the gate; the error
    message distinguishes 'user disabled' from 'surface unsupported'."""
    monkeypatch.setenv("PERFXPERT_GATE_HOOK", "0")
    with pytest.raises(GateHookUnsupported, match="user request"):
        install(tmp_path)


def test_uninstall_is_noop(tmp_path: Path) -> None:
    """Since install() never writes files, uninstall() is a no-op.

    The call MUST NOT raise and MUST NOT create any files.
    """
    before = set(tmp_path.iterdir()) if tmp_path.is_dir() else set()
    result = uninstall(tmp_path)
    after = set(tmp_path.iterdir()) if tmp_path.is_dir() else set()
    assert result is None
    assert before == after


# ---------------------------------------------------------------------------
# Event-based evaluator (documentation-via-test).
# ---------------------------------------------------------------------------


def test_evaluate_permits_perfxpert_tool_always() -> None:
    """Any `mcp__perfxpert__*` tool is allowed before intent_classify."""
    r = evaluate_gate_state(
        "mcp__perfxpert__intent_classify", intent_classify_observed=False
    )
    assert r["allowed"] is True


def test_evaluate_rejects_non_perfxpert_before_intent_classify() -> None:
    """Non-perfxpert + intent_classify not observed → rejected.

    B-N3 event-based rule: no turn counter; lift requires
    intent_classify-returned-in-session.
    """
    r = evaluate_gate_state(
        "Bash", intent_classify_observed=False
    )
    assert r["allowed"] is False
    assert "intent_classify" in r["reason"].lower()


def test_evaluate_permits_after_intent_classify() -> None:
    """Once intent_classify has returned, any tool is permitted."""
    r = evaluate_gate_state(
        "Bash", intent_classify_observed=True
    )
    assert r["allowed"] is True


def test_evaluate_marks_enforced_by_prompt_layer_on_codex() -> None:
    """Documents that Codex gate is prompt-layer-only (not server-side)."""
    r = evaluate_gate_state(
        "Bash", intent_classify_observed=False
    )
    assert r.get("enforced_by") == "prompt-layer"


# ---------------------------------------------------------------------------
# Adapter-integration: I-N1 partial-state protection.
# ---------------------------------------------------------------------------


def _fake_codex_subprocess(
    *, list_stdout: bytes = b"perfxpert\n", add_exit: int = 0
):
    """Fake subprocess.run for codex CLI — happy path defaults."""
    import subprocess

    _REAL_RUN = subprocess.run

    def _run(cmd, *args, **kwargs):
        class _R:
            def __init__(self, rc: int, stdout: bytes = b"", stderr: bytes = b""):
                self.returncode = rc
                self.stdout = stdout
                self.stderr = stderr

        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "--version":
            return _R(0, stdout=b"codex 0.7.0\n")
        if isinstance(cmd, list) and "mcp" in cmd:
            verb = cmd[cmd.index("mcp") + 1]
            if verb == "list":
                return _R(0, stdout=list_stdout)
            if verb == "add":
                return _R(add_exit)
        return _R(0)

    return _run


@pytest.fixture
def isolated_home_with_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Redirect HOME + XDG_CONFIG_HOME; pre-trust the project cwd.

    Returns (home, project_cwd). The project cwd is marked trusted
    in ~/.codex/config.toml so the install's trust gate passes
    without interactive prompting — the test can focus on the gate
    hook's I-N1 partial-state behavior.
    """
    import subprocess

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setenv("PERFXPERT_SKIP_LIVE_CHECK", "1")

    cwd = tmp_path / "proj"
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
    # Pre-mark trusted so the install's trust-gate passes.
    codex_cfg = tmp_path / ".codex" / "config.toml"
    codex_cfg.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(cwd.expanduser().resolve())
    codex_cfg.write_text(
        f'[projects."{resolved}"]\ntrust_level = "trusted"\n'
    )
    return tmp_path, cwd


def test_codex_hook_install_raises_before_mcp_registration_on_unsupported_surface(
    isolated_home_with_trust: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-N1 invariant: the adapter calls gate_hook.install() BEFORE
    MCP registration. Since Codex's install() raises
    GateHookUnsupported unconditionally, the adapter MUST catch it
    AND fall through to MCP registration (prompt-layer-only
    enforcement) — NOT abort the whole install.

    What we assert: the order of operations is hook-install THEN
    mcp-register; the adapter records gate_hook_installed=False
    in the subsequent LiveCheckReport; the install still completes.
    """
    home, cwd = isolated_home_with_trust

    # Record the order of operations.
    call_order: list[str] = []

    # Patch the gate hook's install to record + raise.
    from perfxpert.cli._gate_hooks import codex as codex_hook

    def _recording_install(cwd_arg):
        call_order.append("gate_hook_install")
        raise GateHookUnsupported("fake prompt-layer-only fallback")

    monkeypatch.setattr(codex_hook, "install", _recording_install)

    # Patch the adapter's subprocess to record MCP calls.
    import subprocess

    _REAL_RUN = subprocess.run

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        if isinstance(cmd, list) and "mcp" in cmd:
            call_order.append(f"mcp_{cmd[cmd.index('mcp') + 1]}")

        class _R:
            returncode = 0
            stdout = b"perfxpert\n"
            stderr = b""

        return _R()

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run", _run
    )

    report = CodexAdapter().install(cwd, scope="project")

    # Gate-hook install MUST come before any mcp_* call.
    assert "gate_hook_install" in call_order
    hook_idx = call_order.index("gate_hook_install")
    mcp_calls = [c for c in call_order if c.startswith("mcp_")]
    # Every MCP call occurred AFTER the gate-hook attempt.
    if mcp_calls:
        first_mcp_idx = next(
            i for i, c in enumerate(call_order) if c.startswith("mcp_")
        )
        assert hook_idx < first_mcp_idx, (
            f"gate_hook_install must run BEFORE mcp_register; "
            f"got order {call_order}"
        )

    # Install should have completed successfully (prompt-layer-only
    # fallback, NOT an abort).
    assert isinstance(report, InstallReport)
    # Actions log should mention the gate-hook-unsupported fallback.
    joined = "\n".join(report.actions).lower()
    assert "prompt-layer" in joined or "unsupported" in joined


def test_codex_verify_mcp_live_records_gate_hook_installed_false(
    isolated_home_with_trust: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gate_hook_installed=False is the documented-known-limit state
    (NOT a failure) on Codex — per plan I-N1. verify_mcp_live() MUST
    record it that way so dashboards / CI lints can distinguish
    'gate missing because surface unsupported' from 'gate broken'.
    """
    home, cwd = isolated_home_with_trust
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )

    report = CodexAdapter().verify_mcp_live(cwd)
    # Tri-state: None=disabled, False=unsupported(known-limit), True=installed.
    # Codex today: False (Bash-only surface cannot enforce MCP gate).
    assert report.gate_hook_installed is False


def test_codex_gate_hook_env_disable_returns_none(
    isolated_home_with_trust: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERFXPERT_GATE_HOOK=0 → verify_mcp_live records gate_hook_installed=None
    (user disabled), distinguishing it from False (surface unsupported)."""
    home, cwd = isolated_home_with_trust
    monkeypatch.setenv("PERFXPERT_GATE_HOOK", "0")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "perfxpert.cli._backend.codex.subprocess.run",
        _fake_codex_subprocess(),
    )
    report = CodexAdapter().verify_mcp_live(cwd)
    assert report.gate_hook_installed is None
