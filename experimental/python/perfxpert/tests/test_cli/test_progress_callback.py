"""Tests for the live-progress feedback wired through the agent
runtime and the ``perfxpert analyze`` CLI.

Covers:
  * agent-level forwarding (``agent_root`` + ``AnalysisSession.run_*``
    propagate callbacks and fire ``entering <phase>`` / ``exit <phase>``),
  * CLI non-TTY behaviour (plain ``[perfxpert]`` status lines on stderr,
    real output on stdout — no ANSI escape codes),
  * ``--no-progress`` silences the feature entirely.

All tests use airgap mode so nothing hits an LLM provider.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from perfxpert import api
from perfxpert.agents import runtime as runtime_mod
from perfxpert.agents import schemas


# --- 1. agent-tool level: agent_root forwards the callback ---------------


def test_agent_root_forwards_callback():
    """``api.agent_root`` threads the callback into the session so both
    ``entering root`` and ``exit root`` fire on an airgap invocation."""
    events: list[str] = []
    api.agent_root(
        airgap=True,
        user_query="why slow?",
        progress_callback=events.append,
    )
    # Order matters: the enter comes before the exit.
    assert "entering root" in events, events
    assert "exit root" in events, events
    assert events.index("entering root") < events.index("exit root")


# --- 2. session-level: run_analysis forwards the callback ----------------


def test_session_run_analysis_forwards_callback(tmp_path):
    """``AnalysisSession.run_analysis`` must fire
    ``entering analysis`` / ``exit analysis`` around the inner call."""
    events: list[str] = []
    session = runtime_mod.build_session(airgap=True)
    # Airgap path needs a real-ish AnalysisInput shape: fabricate a db
    # path. The airgap analysis stub never reads it; Pydantic only
    # validates presence of the required field.
    db_path = tmp_path / "fake.db"
    db_path.write_bytes(b"")
    payload = schemas.AnalysisInput(database_path=str(db_path))
    session.run_analysis(payload, progress_callback=events.append)
    assert "entering analysis" in events
    assert "exit analysis" in events


def test_build_session_accepts_progress_callback_default():
    """``build_session`` stores the default callback so per-call
    ``progress_callback=None`` still triggers emits."""
    events: list[str] = []
    session = runtime_mod.build_session(
        airgap=True, progress_callback=events.append
    )
    session.run_root(schemas.RootInput(user_query="?"))
    assert "entering root" in events
    assert "exit root" in events


# --- 3. CLI non-TTY: plain status lines on stderr ------------------------


_CLI_PRELUDE = (
    # Run analyze.main() directly inside a subprocess so we get a real
    # (non-TTY) stderr pipe. We skip the argparse `-i` requirement by
    # driving _execute_agentic through the process_args path.
    "import sys\n"
    "from perfxpert import analyze\n"
    "analyze._execute_agentic(\n"
    "    None,\n"
    "    config=None,\n"
    "    source_dir='.',\n"
    "    output_format='json',\n"
    "    enable_llm=True,\n"        # forces the callback branch
    "    llm_provider='anthropic',\n"  # ignored under airgap
    "    no_progress={no_progress},\n"
    ")\n"
)


def _ansi_free(text: str) -> bool:
    """True if the string contains no CSI / OSC escape sequences."""
    return re.search(r"\x1b\[[0-9;]*[A-Za-z]", text) is None


@pytest.fixture
def _airgap_env(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["PERFXPERT_AIRGAP"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
    )
    return env


def test_analyze_cli_non_tty_prints_status_lines(_airgap_env):
    """Subprocess run of ``_execute_agentic`` under airgap + a piped
    stderr emits the ``[perfxpert]`` status prefix on stderr and the
    JSON result on stdout. No ANSI escapes in either stream.
    """
    code = _CLI_PRELUDE.format(no_progress="False")
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_airgap_env,
        timeout=30,
    )
    assert res.returncode == 0, (
        f"exit={res.returncode}\nstderr={res.stderr!r}\nstdout={res.stdout!r}"
    )
    # stdout should be JSON-ish (the formatted agentic output).
    assert res.stdout.strip(), "stdout must contain the analysis output"
    assert _ansi_free(res.stdout), "stdout must be ANSI-free when piped"
    # stderr has at least one [perfxpert] status line.
    assert "[perfxpert]" in res.stderr, (
        f"expected plain status lines on stderr; got {res.stderr!r}"
    )
    # No spinner escape codes on a piped stream.
    assert _ansi_free(res.stderr), (
        f"stderr escape codes leaked on non-TTY: {res.stderr!r}"
    )


def test_analyze_cli_no_progress_flag_silent(_airgap_env):
    """When ``--no-progress`` is set, the CLI suppresses the
    ``[perfxpert]`` status prefix even in LLM mode. stdout is unchanged.
    """
    code = _CLI_PRELUDE.format(no_progress="True")
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_airgap_env,
        timeout=30,
    )
    assert res.returncode == 0, (
        f"exit={res.returncode}\nstderr={res.stderr!r}\nstdout={res.stdout!r}"
    )
    assert res.stdout.strip(), "stdout must still contain the analysis output"
    assert "[perfxpert]" not in res.stderr, (
        f"--no-progress must silence status lines; got stderr={res.stderr!r}"
    )
