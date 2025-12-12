"""Tests for `perfxpert.cli._gate_hooks` (Task 4.6, F1, B-N2, B-N3).

Covers per-backend:

* opencode: rejection before intent_classify, lift after, event-based
  permit on turn 2 post intent_classify, fork-only docstring callout.
* claude: native PreToolUse install with settings.json merge;
  GateHookUnsupported when surface unavailable; marker substring
  for uninstall recognition; install-before-MCP invariant (I-N1)
  exercised via `evaluate_gate_state`.
* gemini: native BeforeTool/AfterTool hooks, event-based lift, uninstall
  clears perfxpert entries.

Also asserts the **event-based** rule: a `bash` on turn 2 after
`intent_classify` on turn 1 is permitted (the cycle-4 false-refusal
class regression test).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from perfxpert.cli._backend.protocol import GateHookUnsupported
from perfxpert.cli._gate_hooks import (
    GATE_HOOK_DISABLED_ENV,
    GATE_REJECTION_REASON_TEMPLATE,
    GATE_STATE_LIFTED_SENTINEL,
)
from perfxpert.cli._gate_hooks import claude as claude_hook
from perfxpert.cli._gate_hooks import gemini as gemini_hook
from perfxpert.cli._gate_hooks import opencode as opencode_hook


# ---------------------------------------------------------------------------
# opencode.
# ---------------------------------------------------------------------------


def test_opencode_gate_rejects_bash_before_intent_classify() -> None:
    result = opencode_hook.evaluate(
        "bash", intent_classify_observed=False
    )
    assert result.get("block") is True
    assert "intent_classify" in result["retryWith"]


def test_opencode_gate_lifts_after_intent_classify() -> None:
    result = opencode_hook.evaluate(
        "bash", intent_classify_observed=True
    )
    assert result == {}


def test_opencode_permits_bash_on_turn_2_after_intent_classify_on_turn_1() -> None:
    """B-N3 false-refusal regression: turn number is NOT consulted."""
    turn1 = opencode_hook.evaluate(
        "perfxpert_intent_classify", intent_classify_observed=False
    )
    assert turn1 == {}  # perfxpert-prefixed tool is always allowed.
    # Simulate intent_classify having returned — session-state flipped.
    turn2 = opencode_hook.evaluate("bash", intent_classify_observed=True)
    assert turn2 == {}


def test_opencode_perfxpert_tools_always_allowed() -> None:
    """Even before intent_classify returns, the perfxpert_* namespace is open."""
    for tool in ("perfxpert_intent_classify", "mcp__perfxpert__report"):
        assert (
            opencode_hook.evaluate(tool, intent_classify_observed=False)
            == {}
        )


def test_opencode_documents_fork_only_dependency_in_docstring() -> None:
    """B-N2: callout must appear in module docstring."""
    assert "fork-only" in opencode_hook.__doc__.lower()
    assert "0020" in opencode_hook.__doc__
    # Docstring mentions "upstream" somewhere explaining the
    # degradation path.
    assert "upstream" in opencode_hook.__doc__.lower()


def test_opencode_install_is_idempotent() -> None:
    """The opencode install is a no-op; calling it twice returns consistent results."""
    r1 = opencode_hook.install(Path("/tmp"))
    r2 = opencode_hook.install(Path("/tmp"))
    assert r1.installed is True
    assert r2.installed is True


# ---------------------------------------------------------------------------
# claude.
# ---------------------------------------------------------------------------


def test_claude_evaluate_rejects_non_perfxpert_before_intent_classify() -> None:
    out = claude_hook.evaluate_gate_state("Bash", intent_classify_observed=False)
    decision = out["hookSpecificOutput"]["permissionDecision"]
    assert decision == "deny"
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "intent_classify" in reason


def test_claude_evaluate_lifts_after_intent_classify() -> None:
    out = claude_hook.evaluate_gate_state(
        "Bash", intent_classify_observed=True
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_claude_evaluate_always_allows_mcp_perfxpert() -> None:
    out = claude_hook.evaluate_gate_state(
        "mcp__perfxpert__intent_classify", intent_classify_observed=False
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_claude_gate_install_writes_script_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(GATE_HOOK_DISABLED_ENV, raising=False)
    result = claude_hook.ClaudeGateHook().install(tmp_path)
    assert result.hook_script.is_file()
    assert result.post_script.is_file()
    assert result.settings_path.is_file()
    # Settings must contain our perfxpert entry on BOTH hooks.
    data = json.loads(result.settings_path.read_text())
    hooks = data["hooks"]
    assert "PreToolUse" in hooks and "PostToolUse" in hooks
    pre_blob = json.dumps(hooks["PreToolUse"])
    post_blob = json.dumps(hooks["PostToolUse"])
    assert "perfxpert-gate" in pre_blob
    assert "perfxpert-gate" in post_blob


def test_claude_gate_install_idempotent_replaces_prior_perfxpert_entry(
    tmp_path: Path,
) -> None:
    """Second install replaces the prior perfxpert entry, not duplicates it."""
    h = claude_hook.ClaudeGateHook()
    h.install(tmp_path)
    h.install(tmp_path)
    data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    pre = data["hooks"]["PreToolUse"]
    # Count entries whose JSON contains 'perfxpert-gate'.
    perfxpert_entries = [e for e in pre if "perfxpert-gate" in json.dumps(e)]
    assert len(perfxpert_entries) == 1


def test_claude_gate_install_preserves_existing_pre_tool_use_hooks(
    tmp_path: Path,
) -> None:
    """User's other hooks must survive."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"hooks": [{"type": "command", "command": "/user/other.sh"}]}
                    ]
                }
            }
        )
    )
    claude_hook.ClaudeGateHook().install(tmp_path)
    data = json.loads(settings.read_text())
    commands = [e["hooks"][0]["command"] for e in data["hooks"]["PreToolUse"]]
    assert "/user/other.sh" in commands
    assert any("perfxpert-gate" in c for c in commands)


def test_claude_gate_install_raises_on_invalid_existing_json(
    tmp_path: Path,
) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("not valid json {{}")
    with pytest.raises(GateHookUnsupported):
        claude_hook.ClaudeGateHook().install(tmp_path)
    # I-N1: no partial state → the gate script should NOT have been
    # left behind.
    assert not (tmp_path / ".claude" / "hooks" / "perfxpert-gate.sh").exists()


def test_claude_gate_install_raises_when_env_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GATE_HOOK_DISABLED_ENV, "0")
    with pytest.raises(GateHookUnsupported, match="0"):
        claude_hook.ClaudeGateHook().install(tmp_path)


def test_claude_gate_render_script_has_substitutions() -> None:
    script = claude_hook.render_gate_script()
    assert "__REJECTION_REASON__" not in script
    assert "__GATE_SENTINEL__" not in script
    assert GATE_STATE_LIFTED_SENTINEL in script
    assert "intent_classify" in script


def _run_shell_hook(script_path: Path, payload: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script_path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def test_claude_gate_pretool_rejects_invalid_session_id(tmp_path: Path) -> None:
    hooks_dir = tmp_path / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True)
    script_path = hooks_dir / "perfxpert-gate.sh"
    script_path.write_text(claude_hook.render_gate_script())
    script_path.chmod(0o755)

    result = _run_shell_hook(
        script_path,
        {"tool_name": "Bash", "session_id": "../escape"},
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    decision = payload["hookSpecificOutput"]["permissionDecision"]
    assert decision == "deny"
    assert list((tmp_path / ".claude").glob(".perfxpert-gate-state.*.json")) == []


def test_claude_gate_valid_session_id_lifts_gate(tmp_path: Path) -> None:
    hooks_dir = tmp_path / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True)
    pre_path = hooks_dir / "perfxpert-gate.sh"
    post_path = hooks_dir / "perfxpert-gate-post.sh"
    pre_path.write_text(claude_hook.render_gate_script())
    post_path.write_text(claude_hook.render_post_script())
    pre_path.chmod(0o755)
    post_path.chmod(0o755)

    session_id = "safe_sid-123"
    post_result = _run_shell_hook(
        post_path,
        {
            "tool_name": "mcp__perfxpert__intent_classify",
            "session_id": session_id,
        },
    )
    assert post_result.returncode == 0
    state_file = tmp_path / ".claude" / f".perfxpert-gate-state.{session_id}.json"
    assert state_file.is_file()

    pre_result = _run_shell_hook(
        pre_path,
        {"tool_name": "Bash", "session_id": session_id},
    )
    assert pre_result.returncode == 0
    payload = json.loads(pre_result.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_claude_gate_posttool_ignores_invalid_session_id(tmp_path: Path) -> None:
    hooks_dir = tmp_path / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True)
    script_path = hooks_dir / "perfxpert-gate-post.sh"
    script_path.write_text(claude_hook.render_post_script())
    script_path.chmod(0o755)

    result = _run_shell_hook(
        script_path,
        {
            "tool_name": "mcp__perfxpert__intent_classify",
            "session_id": "../../escape",
        },
    )
    assert result.returncode == 0
    assert list((tmp_path / ".claude").glob(".perfxpert-gate-state.*.json")) == []


def test_claude_gate_uninstall_removes_perfxpert_entries(tmp_path: Path) -> None:
    h = claude_hook.ClaudeGateHook()
    h.install(tmp_path)
    ok = h.uninstall(tmp_path)
    assert ok is True
    data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert "hooks" not in data or "perfxpert-gate" not in json.dumps(data["hooks"])


# ---------------------------------------------------------------------------
# gemini.
# ---------------------------------------------------------------------------


def test_gemini_evaluate_rejects_before_intent_classify() -> None:
    out = gemini_hook.evaluate_gate_state("Bash", intent_classify_observed=False)
    assert out["allowed"] is False
    assert "intent_classify" in out["reason"]


def test_gemini_evaluate_lifts_after_intent_classify() -> None:
    out = gemini_hook.evaluate_gate_state(
        "Bash", intent_classify_observed=True
    )
    assert out["allowed"] is True


def test_gemini_evaluate_perfxpert_prefix_always_allowed() -> None:
    out = gemini_hook.evaluate_gate_state(
        "mcp_perfxpert_intent_classify", intent_classify_observed=False
    )
    assert out["allowed"] is True


def test_gemini_gate_install_writes_settings_and_scripts(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    result = gemini_hook.GeminiGateHook().install(proj)
    assert result.settings_path.is_file()
    assert result.hook_script.is_file()
    assert result.post_script.is_file()
    data = json.loads(result.settings_path.read_text())
    hooks = data["hooks"]
    assert "BeforeTool" in hooks and "AfterTool" in hooks
    assert "perfxpert-gate" in json.dumps(hooks)


def test_gemini_gate_install_preserves_existing_beforetool_hooks(
    tmp_path: Path,
) -> None:
    proj = tmp_path / "proj"
    (proj / ".gemini").mkdir(parents=True)
    settings = proj / ".gemini" / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "BeforeTool": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/user/other.sh",
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    gemini_hook.GeminiGateHook().install(proj)
    data = json.loads(settings.read_text())
    commands = [
        entry["hooks"][0]["command"] for entry in data["hooks"]["BeforeTool"]
    ]
    assert "/user/other.sh" in commands
    assert any("perfxpert-gate" in command for command in commands)


def test_gemini_gate_install_raises_when_env_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GATE_HOOK_DISABLED_ENV, "0")
    with pytest.raises(GateHookUnsupported):
        gemini_hook.GeminiGateHook().install(tmp_path / "proj")


@pytest.mark.parametrize(
    "existing",
    [
        "not valid json {{",
        "[]",
        json.dumps({"hooks": []}),
    ],
    ids=["invalid-json", "top-level-list", "hooks-not-object"],
)
def test_gemini_gate_install_rolls_back_scripts_on_invalid_settings(
    tmp_path: Path, existing: str
) -> None:
    proj = tmp_path / "proj"
    settings = proj / ".gemini" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(existing)

    with pytest.raises(GateHookUnsupported):
        gemini_hook.GeminiGateHook().install(proj)

    assert not (proj / ".gemini" / "hooks" / "perfxpert-gate.sh").exists()
    assert not (proj / ".gemini" / "hooks" / "perfxpert-gate-post.sh").exists()


def test_gemini_gate_uninstall_removes_perfxpert_entries(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    h = gemini_hook.GeminiGateHook()
    h.install(proj)
    assert h.uninstall(proj) is True
    data = json.loads((proj / ".gemini" / "settings.json").read_text())
    assert "hooks" not in data or "perfxpert-gate" not in json.dumps(data["hooks"])


def test_gemini_gate_valid_session_id_lifts_gate(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    gemini_hook.GeminiGateHook().install(proj)

    session_id = "safe.sid-123"
    post_result = _run_shell_hook(
        proj / ".gemini" / "hooks" / "perfxpert-gate-post.sh",
        {
            "tool_name": "mcp_perfxpert_intent_classify",
            "session_id": session_id,
        },
    )
    assert post_result.returncode == 0
    state_file = proj / ".gemini" / "runtime" / f"perfxpert-gate-{session_id}.json"
    assert state_file.is_file()
    assert GATE_STATE_LIFTED_SENTINEL in state_file.read_text()

    pre_result = _run_shell_hook(
        proj / ".gemini" / "hooks" / "perfxpert-gate.sh",
        {
            "tool_name": "Bash",
            "session_id": session_id,
        },
    )
    assert pre_result.returncode == 0
    payload = json.loads(pre_result.stdout)
    assert payload["decision"] == "allow"


def test_gemini_gate_posttool_ignores_invalid_session_id(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    gemini_hook.GeminiGateHook().install(proj)

    result = _run_shell_hook(
        proj / ".gemini" / "hooks" / "perfxpert-gate-post.sh",
        {
            "tool_name": "mcp_perfxpert_intent_classify",
            "session_id": "../../escape",
        },
    )
    assert result.returncode == 0
    assert list((proj / ".gemini" / "runtime").glob("perfxpert-gate-*.json")) == []


# ---------------------------------------------------------------------------
# Cross-backend: new session always starts gate engaged.
# ---------------------------------------------------------------------------


def test_new_session_always_starts_gate_engaged_claude(tmp_path: Path) -> None:
    """I-N3: new session (no state file yet) → gate engaged."""
    # Simulate "new session, no state observed yet".
    out = claude_hook.evaluate_gate_state("Bash", intent_classify_observed=False)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_new_session_always_starts_gate_engaged_gemini(tmp_path: Path) -> None:
    out = gemini_hook.evaluate_gate_state("Bash", intent_classify_observed=False)
    assert out["allowed"] is False


def test_new_session_always_starts_gate_engaged_opencode(tmp_path: Path) -> None:
    out = opencode_hook.evaluate("bash", intent_classify_observed=False)
    assert out.get("block") is True


def test_gate_rejection_template_shared_across_backends() -> None:
    """The rejection reason text is sourced from ONE template; drift
    between prompt-layer and hook messaging is a bug (I-N3)."""
    reason = GATE_REJECTION_REASON_TEMPLATE.format(classify_tool="X")
    claude_reason = claude_hook.evaluate_gate_state(
        "Bash", intent_classify_observed=False, classify_tool="X"
    )["hookSpecificOutput"]["permissionDecisionReason"]
    assert claude_reason == reason
