"""Tests for `ClaudeCodeAdapter.verify_mcp_live` (Task 4c).

Covers:

* `verify_mcp_live` returns a healthy report when `claude mcp list`
  lists perfxpert.
* Detects an unhealthy/missing entry (B1).
* Telemetry probe (I11) when `PERFXPERT_TELEMETRY=1`.
* Retries handshake 3× with exponential backoff 2/4/8 (F2, I-N5).
* Early-exits when `PERFXPERT_MCP_RETRY_BUDGET_S` budget exhausted
  (I-N5).
* `gate_hook_installed` tristate:
   - `None` when `PERFXPERT_GATE_HOOK=0`.
   - `False` when settings.json has no hook entry (documented-known-limit).
   - `True` when settings.json contains the perfxpert hook marker.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from perfxpert.cli._backend import _prompt_adapter as pa
from perfxpert.cli._backend.claude import ClaudeCodeAdapter
from perfxpert.cli._backend.protocol import LiveCheckReport


_REAL_RUN = subprocess.run


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# mcp_listed / healthy.
# ---------------------------------------------------------------------------


def _fake_mcp_list_response(
    payload: dict,
    *,
    returncode: int = 0,
    status: str = "✓ Connected",
) -> "subprocess.CompletedProcess":
    """Simulate `claude mcp list` plain-text output from a server dict.

    Accepts the legacy `{"mcpServers": {<name>: {...}}}` shape used
    across tests, plus a flat `{<name>: {...}}`. Each listed server
    becomes a line of the form:

        <name>: <command-or-endpoint> - <status>

    `status` defaults to "✓ Connected" (healthy). Pass `status="✘ failed"`
    to simulate an unhealthy entry.
    """
    class _R:
        pass

    # Normalise the two legacy shapes.
    servers = payload.get("mcpServers", payload) if isinstance(payload, dict) else {}

    lines = ["Checking MCP server health…", ""]
    for name, spec in servers.items():
        endpoint = (
            (isinstance(spec, dict) and spec.get("command"))
            or (isinstance(spec, dict) and spec.get("url"))
            or "/usr/bin/perfxpert-mcp"
        )
        lines.append(f"{name}: {endpoint} - {status}")

    r = _R()
    r.returncode = returncode
    r.stdout = ("\n".join(lines) + "\n").encode("utf-8")
    r.stderr = b""
    return r


def test_verify_mcp_live_happy_path(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert isinstance(report, LiveCheckReport)
    assert report.mcp_listed is True
    assert report.mcp_healthy is True
    assert report.error is None


def test_verify_mcp_live_detects_unhealthy_entry(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: perfxpert entry missing from list → healthy=False."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"other": {"command": "other-bin"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    # Force the retry helper to sleep 0 so tests run fast.
    monkeypatch.setattr(pa.time, "sleep", lambda _s: None)
    monkeypatch.setenv("PERFXPERT_MCP_RETRY_BUDGET_S", "100")
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.mcp_healthy is False
    assert report.error is not None
    assert "perfxpert" in report.error


# ---------------------------------------------------------------------------
# Retry + backoff.
# ---------------------------------------------------------------------------


def test_verify_mcp_live_retries_with_backoff_2_4_8(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-N5: on transient failure, retry 3× with backoff 2s / 4s / 8s."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    attempts = {"n": 0}

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        attempts["n"] += 1
        if attempts["n"] < 3:
            # Simulate the F2 race.
            class _R:
                returncode = 1
                stdout = b""
                stderr = b"transient bootstrap failure"

            return _R()
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    sleeps: list[float] = []
    monkeypatch.setattr(pa.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("PERFXPERT_MCP_RETRY_BUDGET_S", "100")
    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )

    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.mcp_healthy is True
    # 2 retries happened → 2 sleeps.
    assert sleeps == [2.0, 4.0]
    assert attempts["n"] == 3


def test_verify_mcp_live_exits_early_when_budget_exhausted(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-N5: respect PERFXPERT_MCP_RETRY_BUDGET_S early-exit."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    attempts = {"n": 0}

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        attempts["n"] += 1

        class _R:
            returncode = 1
            stdout = b""
            stderr = b"always fails"

        return _R()

    # Fake clock + sleep that advance together so retry_mcp_handshake
    # computes elapsed correctly.
    fake_time = {"t": 0.0}
    monkeypatch.setattr(pa.time, "sleep", lambda s: fake_time.update(t=fake_time["t"] + s))
    monkeypatch.setattr(pa.time, "monotonic", lambda: fake_time["t"])

    monkeypatch.setenv("PERFXPERT_MCP_RETRY_BUDGET_S", "5")
    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.mcp_healthy is False
    # With 5s budget: attempt 1 fails, sleep 2s (elapsed=2, <=5),
    # attempt 2 fails, would sleep 4s → elapsed 6 > 5 → bail. So
    # exactly 2 attempts.
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# gate_hook_installed tri-state (I-N1).
# ---------------------------------------------------------------------------


def test_gate_hook_none_when_env_disables(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERFXPERT_GATE_HOOK", "0")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.gate_hook_installed is None


def test_gate_hook_false_when_settings_missing(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-N1: documented-known-limit — surface unsupported / not installed."""
    monkeypatch.delenv("PERFXPERT_GATE_HOOK", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.gate_hook_installed is False


def test_gate_hook_true_when_settings_has_hook(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = cwd / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": ".claude/hooks/perfxpert-gate.sh",
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )

    monkeypatch.delenv("PERFXPERT_GATE_HOOK", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.gate_hook_installed is True


# ---------------------------------------------------------------------------
# Telemetry probe (I11).
# ---------------------------------------------------------------------------


def test_telemetry_probe_reads_log_for_intent_classify(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    xdg = tmp_path / "xdg_cache"
    xdg.mkdir()
    log = xdg / "perfxpert" / "mcp-telemetry.log"
    log.parent.mkdir()
    log.write_text("[1234] called: mcp__perfxpert__intent_classify\n")

    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setenv("PERFXPERT_TELEMETRY", "1")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd, telemetry=True)
    assert report.mcp_healthy is True


def test_telemetry_probe_flags_missing_intent_classify(
    cwd: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    xdg = tmp_path / "xdg_cache"
    xdg.mkdir()
    log = xdg / "perfxpert" / "mcp-telemetry.log"
    log.parent.mkdir()
    # No intent_classify entry.
    log.write_text("[1234] started\n")

    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setenv("PERFXPERT_TELEMETRY", "1")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

    def _run(cmd, *a, **kw):
        if cmd[:1] == ["git"]:
            return _REAL_RUN(cmd, *a, **kw)
        return _fake_mcp_list_response(
            {"mcpServers": {"perfxpert": {"command": "perfxpert-mcp"}}}
        )

    monkeypatch.setattr(
        "perfxpert.cli._backend.claude.subprocess.run", _run
    )
    report = ClaudeCodeAdapter().verify_mcp_live(cwd, telemetry=True)
    assert report.mcp_healthy is False
    assert report.error and "telemetry" in report.error


# ---------------------------------------------------------------------------
# Missing binary at verify time (rare but possible).
# ---------------------------------------------------------------------------


def test_verify_mcp_live_missing_binary(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    report = ClaudeCodeAdapter().verify_mcp_live(cwd)
    assert report.mcp_healthy is False
    assert report.gate_hook_installed is None
    assert "not on PATH" in (report.error or "")
